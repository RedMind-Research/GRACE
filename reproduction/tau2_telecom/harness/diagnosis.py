# Copyright 2026 Dan C. Hsu and Luke Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequential, fail-closed two-stage diagnosis with safe deterministic resume."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from grace.providers.base import (
    LLMProvider,
    PromptRequest,
    ProviderCallResult,
    UsageRecord,
    make_logical_call_id,
)

from .diagnosis_prompts import (
    DIAGNOSIS_PROMPT_SUITE_HASH,
    DIAGNOSIS_SYNTHESIS_SYSTEM,
    DIAGNOSIS_SYNTHESIS_USER,
    PER_FAILURE_REFLECTION_SYSTEM,
    PER_FAILURE_REFLECTION_USER,
)
from .identity import (
    canonical_json,
    canonical_sha256,
    make_diagnosis_fingerprint,
    make_reflection_id,
    text_sha256,
    validate_sha256,
)
from .models import (
    DiagnosisErrorRecord,
    DiagnosisManifest,
    DiagnosisResult,
    DiagnosisStatus,
    DiagnosisSynthesis,
    EligibilityDecision,
    EligibilityStatus,
    EpisodeCell,
    ExpectedMatrix,
    FailureReflection,
    ObservedToolErrorEvidence,
    ReflectionArtifact,
    SynthesisArtifact,
    SynthesisResponse,
    ToolErrorEvidenceItem,
    TrajectoryRecord,
)


TOOL_NOT_FOUND_RE = re.compile(
    r"Error:\s*Tool ['`]([^'`]+)['`] not found",
    re.IGNORECASE,
)
DIAGNOSIS_REFLECTION_MAX_TOKENS = 65_536
DIAGNOSIS_SYNTHESIS_MAX_TOKENS = 65_536


class DiagnosisError(RuntimeError):
    """Base error for deterministic diagnosis orchestration."""


class DiagnosisFingerprintMismatch(DiagnosisError):
    """Raised when resume inputs differ from the frozen diagnosis contract."""


class DiagnosisArtifactError(DiagnosisError):
    """Raised when a persisted diagnosis artifact is corrupt or inconsistent."""


def classify_trajectory(
    value: TrajectoryRecord | Mapping[str, Any],
) -> tuple[TrajectoryRecord | None, EligibilityDecision]:
    """Classify one trajectory before any model call.

    Infrastructure termination takes precedence over a contradictory success
    flag.  Malformed records never get silently dropped.
    """

    if isinstance(value, TrajectoryRecord):
        record = value
    else:
        try:
            record = TrajectoryRecord.model_validate(value)
        except ValidationError:
            raw_episode_id = value.get("episode_id") if isinstance(value, Mapping) else None
            raw_split_slot = value.get("split_slot") if isinstance(value, Mapping) else None
            return None, EligibilityDecision(
                episode_id=raw_episode_id if isinstance(raw_episode_id, str) else None,
                split_slot=raw_split_slot if isinstance(raw_split_slot, str) else None,
                status=EligibilityStatus.INVALID_INPUT,
                reason_code="trajectory_schema_invalid",
                detail="trajectory failed the frozen public schema",
            )

    if record.termination_reason.strip().lower() in {"error", "infrastructure_error"}:
        status = EligibilityStatus.INFRASTRUCTURE_ERROR
        code = "infrastructure_termination"
        detail = "trajectory terminated with an infrastructure error"
    elif record.success:
        status = EligibilityStatus.SUCCESSFUL_CONTROL
        code = "successful_episode"
        detail = "successful episode is counted but excluded from reflection"
    elif not record.substantive:
        status = EligibilityStatus.INVALID_INPUT
        code = "trajectory_not_substantive"
        detail = "failed trajectory contains no substantive message or tool call"
    else:
        status = EligibilityStatus.ELIGIBLE
        code = "behavioral_failure"
        detail = "non-infrastructure behavioral failure is eligible for reflection"

    return record, EligibilityDecision(
        episode_id=record.episode_id,
        split_slot=record.split_slot,
        status=status,
        reason_code=code,
        detail=detail,
    )


def _record_matches_episode(record: TrajectoryRecord, episode: EpisodeCell) -> bool:
    return all(
        (
            record.episode_id == episode.episode_id,
            record.task_uid == episode.task_uid,
            record.task_definition_hash == episode.task_definition_hash,
            record.benchmark_task_id == episode.benchmark_task_id,
            record.split_slot == episode.split_slot,
            record.trial == episode.trial,
            record.protocol_seed == episode.protocol_seed,
        )
    )


def _prepare_inputs(
    expected: ExpectedMatrix,
    trajectories: Sequence[TrajectoryRecord | Mapping[str, Any]],
) -> tuple[tuple[TrajectoryRecord, ...], tuple[EligibilityDecision, ...]]:
    """Bind raw inputs to the exact matrix and preserve canonical episode order."""

    raw_by_id: dict[str, list[TrajectoryRecord | Mapping[str, Any]]] = defaultdict(list)
    idless_count = 0
    for value in trajectories:
        if isinstance(value, TrajectoryRecord):
            episode_id: Any = value.episode_id
        elif isinstance(value, Mapping):
            episode_id = value.get("episode_id")
        else:
            episode_id = None
        if isinstance(episode_id, str) and episode_id:
            try:
                validate_sha256(episode_id, field="episode_id")
            except ValueError:
                idless_count += 1
            else:
                raw_by_id[episode_id].append(value)
        else:
            idless_count += 1

    expected_by_id = {episode.episode_id: episode for episode in expected.episodes}
    records: list[TrajectoryRecord] = []
    decisions: list[EligibilityDecision] = []
    for episode in expected.episodes:
        candidates = raw_by_id.get(episode.episode_id, [])
        if not candidates:
            decisions.append(
                EligibilityDecision(
                    episode_id=episode.episode_id,
                    split_slot=episode.split_slot,
                    status=EligibilityStatus.INVALID_INPUT,
                    reason_code="expected_trajectory_missing",
                    detail="expected matrix cell has no trajectory",
                )
            )
            continue
        if len(candidates) > 1:
            decisions.append(
                EligibilityDecision(
                    episode_id=episode.episode_id,
                    split_slot=episode.split_slot,
                    status=EligibilityStatus.INVALID_INPUT,
                    reason_code="duplicate_trajectory",
                    detail="expected matrix cell has duplicate trajectories",
                )
            )
            continue
        record, decision = classify_trajectory(candidates[0])
        if record is not None and not _record_matches_episode(record, episode):
            decision = EligibilityDecision(
                episode_id=episode.episode_id,
                split_slot=episode.split_slot,
                status=EligibilityStatus.INVALID_INPUT,
                reason_code="episode_identity_mismatch",
                detail="trajectory identity fields do not match the expected matrix cell",
            )
            record = None
        decisions.append(decision)
        if record is not None:
            records.append(record)

    for episode_id in sorted(set(raw_by_id) - set(expected_by_id)):
        decisions.append(
            EligibilityDecision(
                episode_id=episode_id,
                status=EligibilityStatus.INVALID_INPUT,
                reason_code="unknown_trajectory",
                detail="trajectory does not belong to the expected matrix",
            )
        )
    for _ in range(idless_count):
        decisions.append(
            EligibilityDecision(
                status=EligibilityStatus.INVALID_INPUT,
                reason_code="trajectory_episode_id_missing",
                detail="trajectory does not expose an episode ID",
            )
        )
    return tuple(records), tuple(decisions)


def _parse_subtasks(task_id: str) -> tuple[str, ...]:
    first_close = task_id.find("]")
    persona_start = task_id.rfind("[PERSONA:")
    if first_close < 0 or persona_start <= first_close:
        return ()
    body = task_id[first_close + 1 : persona_start]
    return tuple(part for part in body.split("|") if part)


def _render_trajectory(record: TrajectoryRecord) -> str:
    lines: list[str] = []
    for message in record.messages:
        if message.content is not None and message.content.strip():
            lines.append(f"[{message.role}]: {message.content}")
        for call in message.tool_calls:
            lines.append(
                f"[{message.role}] Tool call: {call.name}({canonical_json(call.arguments)})"
            )
    return "\n".join(lines)


def _reflection_request(
    policy: str,
    record: TrajectoryRecord,
    *,
    reflection_id: str,
    model: str,
) -> PromptRequest:
    subtasks = _parse_subtasks(record.benchmark_task_id)
    subtasks_list = "\n".join(f"- {subtask}" for subtask in subtasks)
    if not subtasks_list:
        subtasks_list = "(No subtask structure available)"
    user_prompt = PER_FAILURE_REFLECTION_USER.format(
        system_prompt=policy,
        task_id=record.benchmark_task_id,
        subtasks_list=subtasks_list,
        trajectory=_render_trajectory(record),
    )
    return PromptRequest(
        system_prompt=PER_FAILURE_REFLECTION_SYSTEM,
        user_prompt=user_prompt,
        stage="diagnosis_reflection",
        logical_call_id=make_logical_call_id(
            stage="diagnosis_reflection",
            system_prompt=PER_FAILURE_REFLECTION_SYSTEM,
            user_prompt=user_prompt,
            model=model,
            binding_id=reflection_id,
        ),
        expect_json=True,
        temperature=0.0,
        max_tokens=DIAGNOSIS_REFLECTION_MAX_TOKENS,
    )


def _synthesis_request(
    reflections: Sequence[ReflectionArtifact],
    *,
    diagnosis_fingerprint: str,
    model: str,
) -> PromptRequest:
    payload = [artifact.reflection.model_dump(mode="json") for artifact in reflections]
    user_prompt = DIAGNOSIS_SYNTHESIS_USER.format(
        num_reflections=len(payload),
        reflections_json=json.dumps(payload, indent=2, ensure_ascii=False),
    )
    return PromptRequest(
        system_prompt=DIAGNOSIS_SYNTHESIS_SYSTEM,
        user_prompt=user_prompt,
        stage="diagnosis_synthesis",
        logical_call_id=make_logical_call_id(
            stage="diagnosis_synthesis",
            system_prompt=DIAGNOSIS_SYNTHESIS_SYSTEM,
            user_prompt=user_prompt,
            model=model,
            binding_id=diagnosis_fingerprint,
        ),
        expect_json=True,
        temperature=0.0,
        max_tokens=DIAGNOSIS_SYNTHESIS_MAX_TOKENS,
    )


def _parsed_object(result: ProviderCallResult) -> Mapping[str, Any]:
    if not isinstance(result.parsed_content, Mapping):
        raise ValueError("provider parsed content must be a JSON object")
    return result.parsed_content


def _tool_error_evidence(records: Sequence[TrajectoryRecord]) -> ObservedToolErrorEvidence:
    tool_counts: Counter[str] = Counter()
    tool_tasks: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for message in record.messages:
            if not message.content:
                continue
            for match in TOOL_NOT_FOUND_RE.finditer(message.content):
                tool = match.group(1)
                tool_counts[tool] += 1
                tool_tasks[tool].add(record.benchmark_task_id)
    tools = tuple(
        ToolErrorEvidenceItem(
            tool=tool,
            count=count,
            task_count=len(tool_tasks[tool]),
            task_ids=tuple(sorted(tool_tasks[tool])),
        )
        for tool, count in sorted(tool_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return ObservedToolErrorEvidence(
        total_tool_not_found_calls=sum(tool_counts.values()),
        unique_tool_not_found_names=len(tool_counts),
        tools=tools,
    )


def render_diagnosis_report(synthesis: DiagnosisSynthesis) -> str:
    """Render a stable report from typed synthesis data."""

    lines = ["# Diagnosis Report", "", synthesis.summary, ""]
    themes = sorted(synthesis.themes, key=lambda theme: (-theme.frequency, theme.theme_id))
    for theme in themes:
        lines.append(f"### {theme.theme_id}: {theme.title} (freq={theme.frequency})")
        if theme.affected_subtask_types:
            lines.append(f"Affected subtasks: {', '.join(theme.affected_subtask_types)}")
        lines.append(f"Root cause: {theme.root_cause}")
        lines.append("Actions:")
        for index, action in enumerate(theme.recommended_actions, 1):
            lines.append(f"  {index}. {action}")
        lines.append("")
    evidence = synthesis.observed_tool_error_evidence
    if evidence.total_tool_not_found_calls:
        lines.extend(["### Observed Tool-Error Evidence", ""])
        for item in evidence.tools:
            lines.append(f"- `{item.tool}`: {item.count} call(s) across {item.task_count} task(s).")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class FileDiagnosisStore:
    """Small diagnosis-specific store; general run storage remains a separate layer."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.reflections_dir = self.root / "reflections"
        self.synthesis_path = self.root / "synthesis.json"
        self.report_path = self.root / "diagnosis_report.md"
        self.complete_path = self.root / "complete.json"

    @staticmethod
    def _json_text(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, text: str, *, replace: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            if replace:
                os.replace(temp_path, path)
            else:
                try:
                    os.link(temp_path, path)
                except FileExistsError:
                    if path.read_text(encoding="utf-8") != text:
                        raise DiagnosisArtifactError(f"immutable artifact differs: {path}")
                temp_path.unlink(missing_ok=True)
            self._fsync_directory(path.parent)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _load_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DiagnosisArtifactError(f"cannot load diagnosis artifact {path}") from exc

    def load_manifest(self) -> DiagnosisManifest | None:
        if not self.manifest_path.exists():
            return None
        try:
            return DiagnosisManifest.model_validate(self._load_json(self.manifest_path))
        except ValidationError as exc:
            raise DiagnosisArtifactError("diagnosis manifest is invalid") from exc

    def save_manifest(self, manifest: DiagnosisManifest) -> None:
        self._atomic_write(self.manifest_path, self._json_text(manifest), replace=True)

    def reflection_path(self, reflection_id: str) -> Path:
        validate_sha256(reflection_id, field="reflection_id")
        return self.reflections_dir / f"{reflection_id}.json"

    def load_reflection(self, reflection_id: str) -> ReflectionArtifact | None:
        path = self.reflection_path(reflection_id)
        if not path.exists():
            return None
        try:
            return ReflectionArtifact.model_validate(self._load_json(path))
        except ValidationError as exc:
            raise DiagnosisArtifactError(f"invalid reflection artifact {reflection_id}") from exc

    def save_reflection(self, artifact: ReflectionArtifact) -> None:
        self._atomic_write(
            self.reflection_path(artifact.reflection_id),
            self._json_text(artifact),
            replace=False,
        )

    def load_synthesis(self) -> SynthesisArtifact | None:
        if not self.synthesis_path.exists():
            return None
        try:
            return SynthesisArtifact.model_validate(self._load_json(self.synthesis_path))
        except ValidationError as exc:
            raise DiagnosisArtifactError("invalid synthesis artifact") from exc

    def save_synthesis(self, artifact: SynthesisArtifact) -> None:
        self._atomic_write(
            self.synthesis_path,
            self._json_text(artifact),
            replace=False,
        )

    def save_report(self, report: str) -> None:
        self._atomic_write(self.report_path, report, replace=False)

    def _accepted_file_hashes(self, manifest: DiagnosisManifest) -> dict[str, str]:
        paths: list[Path] = [
            self.reflection_path(reflection_id) for reflection_id in manifest.ordered_reflection_ids
        ]
        if manifest.status == DiagnosisStatus.COMPLETE:
            paths.append(self.synthesis_path)
        if manifest.status in {DiagnosisStatus.COMPLETE, DiagnosisStatus.NO_UPDATE}:
            paths.append(self.report_path)
        hashes: dict[str, str] = {}
        for path in paths:
            if not path.exists() or not path.is_file():
                raise DiagnosisArtifactError(f"accepted diagnosis artifact is missing: {path}")
            hashes[path.relative_to(self.root).as_posix()] = text_sha256(
                path.read_text(encoding="utf-8")
            )
        return hashes

    def finalize(self, manifest: DiagnosisManifest) -> None:
        marker = {
            "accepted_file_hashes": self._accepted_file_hashes(manifest),
            "diagnosis_fingerprint": manifest.diagnosis_fingerprint,
            "manifest_hash": canonical_sha256(manifest.model_dump(mode="json")),
            "namespace": "grace.diagnosis-complete.v1",
            "ordered_reflection_ids": list(manifest.ordered_reflection_ids),
            "status": manifest.status.value,
        }
        self._atomic_write(self.complete_path, self._json_text(marker), replace=False)

    def validate_complete(self, manifest: DiagnosisManifest) -> None:
        if not self.complete_path.exists():
            raise DiagnosisArtifactError("completion marker is missing")
        marker = self._load_json(self.complete_path)
        expected = {
            "accepted_file_hashes": self._accepted_file_hashes(manifest),
            "diagnosis_fingerprint": manifest.diagnosis_fingerprint,
            "manifest_hash": canonical_sha256(manifest.model_dump(mode="json")),
            "namespace": "grace.diagnosis-complete.v1",
            "ordered_reflection_ids": list(manifest.ordered_reflection_ids),
            "status": manifest.status.value,
        }
        if marker != expected:
            raise DiagnosisArtifactError("completion marker does not match the manifest")


def _safe_error(
    *,
    stage: Literal["input", "reflection", "synthesis"],
    logical_id: str | None,
    error: Exception,
    usage: UsageRecord | None = None,
) -> DiagnosisErrorRecord:
    messages = {
        "input": "diagnosis input failed deterministic validation",
        "reflection": "reflection provider call or typed parsing failed",
        "synthesis": "synthesis provider call or typed parsing failed",
    }
    return DiagnosisErrorRecord(
        stage=stage,
        logical_id=logical_id,
        error_type=type(error).__name__,
        safe_message=messages[stage],
        usage=usage,
    )


def _update_manifest(
    manifest: DiagnosisManifest,
    **updates: Any,
) -> DiagnosisManifest:
    """Apply state transitions through normal validation, never unchecked copy."""

    payload = manifest.model_dump(mode="json")
    payload.update(updates)
    return DiagnosisManifest.model_validate(payload)


def _load_completed_result(
    store: FileDiagnosisStore,
    manifest: DiagnosisManifest,
    reflection_specs: Sequence[tuple[TrajectoryRecord, str, str]],
    *,
    require_marker: bool = True,
) -> DiagnosisResult:
    if require_marker:
        store.validate_complete(manifest)
    expected_specs = {
        reflection_id: (record, input_fingerprint)
        for record, reflection_id, input_fingerprint in reflection_specs
    }
    reflections: list[ReflectionArtifact] = []
    for reflection_id in manifest.ordered_reflection_ids:
        artifact = store.load_reflection(reflection_id)
        if artifact is None:
            raise DiagnosisArtifactError("completed diagnosis is missing a reflection")
        try:
            record, input_fingerprint = expected_specs[reflection_id]
        except KeyError as exc:
            raise DiagnosisArtifactError(
                "completed diagnosis has an unexpected reflection"
            ) from exc
        if (
            artifact.episode_id != record.episode_id
            or artifact.task_uid != record.task_uid
            or artifact.trajectory_hash != record.trajectory_hash
            or artifact.input_fingerprint != input_fingerprint
        ):
            raise DiagnosisArtifactError("completed reflection does not match current inputs")
        reflections.append(artifact)
    synthesis_artifact = store.load_synthesis()
    if manifest.status == DiagnosisStatus.COMPLETE:
        if synthesis_artifact is None or not store.report_path.exists():
            raise DiagnosisArtifactError("completed diagnosis is missing synthesis/report")
        if synthesis_artifact.diagnosis_fingerprint != manifest.diagnosis_fingerprint:
            raise DiagnosisArtifactError("synthesis fingerprint does not match manifest")
        report = store.report_path.read_text(encoding="utf-8")
        if report != render_diagnosis_report(synthesis_artifact.synthesis):
            raise DiagnosisArtifactError("diagnosis report does not render from synthesis")
        usage = tuple(artifact.usage for artifact in reflections) + (synthesis_artifact.usage,)
        return DiagnosisResult(
            status=manifest.status,
            manifest=manifest,
            reflections=tuple(reflections),
            synthesis=synthesis_artifact.synthesis,
            report=report,
            usage=usage,
            artifact_dir=store.root,
        )
    if manifest.status == DiagnosisStatus.NO_UPDATE:
        no_update_report = (
            store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else None
        )
        return DiagnosisResult(
            status=manifest.status,
            manifest=manifest,
            report=no_update_report,
            artifact_dir=store.root,
        )
    raise DiagnosisArtifactError("only complete or no_update manifests have completion markers")


def run_diagnosis(
    *,
    provider: LLMProvider,
    expected: ExpectedMatrix,
    trajectories: Sequence[TrajectoryRecord | Mapping[str, Any]],
    policy: str,
    artifact_dir: str | Path,
    config_hash: str,
    diagnosis_prompt_hash: str = DIAGNOSIS_PROMPT_SUITE_HASH,
) -> DiagnosisResult:
    """Execute or safely resume one exact-set diagnosis batch.

    The function performs no concurrency and never invokes synthesis while an
    input or reflection is unresolved.  Existing complete artifacts are loaded
    without any filesystem mutation or provider call.
    """

    if not policy.strip():
        raise ValueError("policy must not be empty")
    validate_sha256(config_hash, field="config_hash")
    validate_sha256(diagnosis_prompt_hash, field="diagnosis_prompt_hash")
    records, classifications = _prepare_inputs(expected, trajectories)
    records_by_id = {record.episode_id: record for record in records}
    eligible_records = tuple(
        records_by_id[decision.episode_id]
        for decision in classifications
        if decision.status == EligibilityStatus.ELIGIBLE and decision.episode_id in records_by_id
    )
    policy_hash = text_sha256(policy)
    reflection_specs: list[tuple[TrajectoryRecord, str, str]] = []
    for record in eligible_records:
        reflection_id = make_reflection_id(
            episode_id=record.episode_id,
            task_uid=record.task_uid,
            trajectory_hash=record.trajectory_hash,
            policy_hash=policy_hash,
            diagnosis_prompt_hash=diagnosis_prompt_hash,
            model=provider.model,
            config_hash=config_hash,
        )
        input_fingerprint = canonical_sha256(
            {
                "namespace": "grace.reflection-input.v1",
                "reflection_id": reflection_id,
                "rendered_trajectory_hash": text_sha256(_render_trajectory(record)),
            }
        )
        reflection_specs.append((record, reflection_id, input_fingerprint))
    ordered_reflection_ids = tuple(item[1] for item in reflection_specs)
    classification_hash = canonical_sha256(
        {
            "classifications": [item.model_dump(mode="json") for item in classifications],
            "namespace": "grace.diagnosis-classification.v1",
            "trajectory_inputs": [
                {"episode_id": record.episode_id, "trajectory_hash": record.trajectory_hash}
                for record in records
            ],
        }
    )
    fingerprint = make_diagnosis_fingerprint(
        expected_matrix_hash=expected.matrix_hash,
        policy_hash=policy_hash,
        diagnosis_prompt_hash=diagnosis_prompt_hash,
        model=provider.model,
        config_hash=config_hash,
        ordered_reflection_ids=ordered_reflection_ids,
        classification_hash=classification_hash,
    )
    store = FileDiagnosisStore(artifact_dir)
    existing = store.load_manifest()
    if existing is not None and existing.diagnosis_fingerprint != fingerprint:
        raise DiagnosisFingerprintMismatch(
            "diagnosis resume fingerprint changed; use a new artifact directory"
        )
    if existing is not None and existing.status in {
        DiagnosisStatus.COMPLETE,
        DiagnosisStatus.NO_UPDATE,
    }:
        if not store.complete_path.exists():
            _load_completed_result(
                store,
                existing,
                reflection_specs,
                require_marker=False,
            )
            store.finalize(existing)
        return _load_completed_result(store, existing, reflection_specs)

    errors = list(existing.errors) if existing is not None else []
    base_manifest = DiagnosisManifest(
        status=DiagnosisStatus.RUNNING,
        diagnosis_fingerprint=fingerprint,
        expected_matrix_hash=expected.matrix_hash,
        policy_hash=policy_hash,
        diagnosis_prompt_hash=diagnosis_prompt_hash,
        config_hash=config_hash,
        model=provider.model,
        classifications=classifications,
        ordered_eligible_episode_ids=tuple(record.episode_id for record in eligible_records),
        ordered_reflection_ids=ordered_reflection_ids,
        errors=tuple(errors),
    )
    blockers = tuple(
        decision
        for decision in classifications
        if decision.status
        in {
            EligibilityStatus.INFRASTRUCTURE_ERROR,
            EligibilityStatus.INVALID_INPUT,
        }
    )
    if blockers:
        for blocker in blockers:
            errors.append(
                DiagnosisErrorRecord(
                    stage="input",
                    logical_id=blocker.episode_id,
                    error_type=blocker.status.value,
                    safe_message=blocker.detail,
                )
            )
        manifest = _update_manifest(
            base_manifest,
            status=DiagnosisStatus.INCOMPLETE,
            errors=tuple(errors),
        )
        if existing != manifest:
            store.save_manifest(manifest)
        return DiagnosisResult(
            status=manifest.status,
            manifest=manifest,
            artifact_dir=store.root,
        )

    if not eligible_records:
        manifest = _update_manifest(base_manifest, status=DiagnosisStatus.NO_UPDATE)
        report = "No eligible behavioral failures; no update is required.\n"
        store.save_report(report)
        store.save_manifest(manifest)
        store.finalize(manifest)
        return DiagnosisResult(
            status=manifest.status,
            manifest=manifest,
            report=report,
            artifact_dir=store.root,
        )

    store.save_manifest(base_manifest)
    completed: list[str] = []
    failed: list[str] = []
    reflections: list[ReflectionArtifact] = []
    all_usage: list[UsageRecord] = [error.usage for error in errors if error.usage is not None]
    for record, reflection_id, input_fingerprint in reflection_specs:
        cached = store.load_reflection(reflection_id)
        if cached is not None:
            if (
                cached.episode_id != record.episode_id
                or cached.trajectory_hash != record.trajectory_hash
                or cached.input_fingerprint != input_fingerprint
            ):
                raise DiagnosisArtifactError("cached reflection fingerprint mismatch")
            reflections.append(cached)
            completed.append(reflection_id)
            all_usage.append(cached.usage)
            continue
        result: ProviderCallResult | None = None
        try:
            result = provider.complete(
                _reflection_request(
                    policy,
                    record,
                    reflection_id=reflection_id,
                    model=provider.model,
                )
            )
            reflection = FailureReflection.model_validate(_parsed_object(result))
            if reflection.task_id != record.benchmark_task_id:
                raise ValueError("reflection task_id does not match the requested task")
            artifact = ReflectionArtifact(
                reflection_id=reflection_id,
                episode_id=record.episode_id,
                task_uid=record.task_uid,
                benchmark_task_id=record.benchmark_task_id,
                trajectory_hash=record.trajectory_hash,
                input_fingerprint=input_fingerprint,
                reflection=reflection,
                usage=result.usage,
            )
            store.save_reflection(artifact)
            reflections.append(artifact)
            completed.append(reflection_id)
            all_usage.append(result.usage)
        except Exception as exc:  # provider and typed-parse errors are both fail-closed
            failed.append(reflection_id)
            usage = result.usage if result is not None else None
            if usage is not None:
                all_usage.append(usage)
            errors.append(
                _safe_error(
                    stage="reflection",
                    logical_id=reflection_id,
                    error=exc,
                    usage=usage,
                )
            )
        progress = _update_manifest(
            base_manifest,
            status=DiagnosisStatus.RUNNING,
            completed_reflection_ids=tuple(completed),
            failed_reflection_ids=tuple(failed),
            errors=tuple(errors),
        )
        store.save_manifest(progress)

    if failed:
        manifest = _update_manifest(
            base_manifest,
            status=DiagnosisStatus.INCOMPLETE,
            completed_reflection_ids=tuple(completed),
            failed_reflection_ids=tuple(failed),
            errors=tuple(errors),
        )
        store.save_manifest(manifest)
        return DiagnosisResult(
            status=manifest.status,
            manifest=manifest,
            reflections=tuple(reflections),
            usage=tuple(all_usage),
            artifact_dir=store.root,
        )

    synthesis_artifact = store.load_synthesis()
    if synthesis_artifact is not None:
        if synthesis_artifact.diagnosis_fingerprint != fingerprint:
            raise DiagnosisArtifactError("cached synthesis fingerprint mismatch")
        synthesis = synthesis_artifact.synthesis
        all_usage.append(synthesis_artifact.usage)
    else:
        synthesis_result: ProviderCallResult | None = None
        try:
            synthesis_result = provider.complete(
                _synthesis_request(
                    reflections,
                    diagnosis_fingerprint=fingerprint,
                    model=provider.model,
                )
            )
            parsed = SynthesisResponse.model_validate(_parsed_object(synthesis_result))
            synthesis = DiagnosisSynthesis(
                themes=parsed.themes,
                summary=parsed.summary,
                observed_tool_error_evidence=_tool_error_evidence(records),
            )
            synthesis_artifact = SynthesisArtifact(
                diagnosis_fingerprint=fingerprint,
                synthesis=synthesis,
                usage=synthesis_result.usage,
            )
            store.save_synthesis(synthesis_artifact)
            all_usage.append(synthesis_result.usage)
        except Exception as exc:
            usage = synthesis_result.usage if synthesis_result is not None else None
            if usage is not None:
                all_usage.append(usage)
            errors.append(
                _safe_error(
                    stage="synthesis",
                    logical_id=fingerprint,
                    error=exc,
                    usage=usage,
                )
            )
            manifest = _update_manifest(
                base_manifest,
                status=DiagnosisStatus.INCOMPLETE,
                completed_reflection_ids=tuple(completed),
                errors=tuple(errors),
            )
            store.save_manifest(manifest)
            return DiagnosisResult(
                status=manifest.status,
                manifest=manifest,
                reflections=tuple(reflections),
                usage=tuple(all_usage),
                artifact_dir=store.root,
            )

    report = render_diagnosis_report(synthesis)
    store.save_report(report)
    manifest = _update_manifest(
        base_manifest,
        status=DiagnosisStatus.COMPLETE,
        completed_reflection_ids=tuple(completed),
        synthesis_accepted=True,
        errors=tuple(errors),
    )
    store.save_manifest(manifest)
    store.finalize(manifest)
    return DiagnosisResult(
        status=manifest.status,
        manifest=manifest,
        reflections=tuple(reflections),
        synthesis=synthesis,
        report=report,
        usage=tuple(all_usage),
        artifact_dir=store.root,
    )


__all__ = [
    "DiagnosisArtifactError",
    "DiagnosisError",
    "DiagnosisFingerprintMismatch",
    "FileDiagnosisStore",
    "classify_trajectory",
    "render_diagnosis_report",
    "run_diagnosis",
]
