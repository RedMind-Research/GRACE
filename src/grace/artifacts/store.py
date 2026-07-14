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

"""Atomic, immutable product artifact persistence.

This store is deliberately engine-neutral.  Engines hand it an accepted
``GraceState``, typed result payloads, and provenance; the store owns only
safe serialization, atomic finalization, integrity verification, and reload.
Reproduction-level per-provider-attempt storage belongs to the optional
telecom harness and is not implemented here.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from grace._version import __version__ as SOURCE_VERSION
from grace.artifacts.models import (
    ArtifactManifest,
    ArtifactRecord,
    EvolutionResult,
    InitializationResult,
    JsonObject,
    PromptProvenance,
    ReconstructionReport,
    ResultStatus,
    ValidationReport,
)
from grace.artifacts.provenance import (
    assert_artifact_safe,
    canonical_json_bytes,
    capture_prompt_provenance,
    manifest_prompt_hashes,
    sha256_bytes,
    to_json_object,
)
from grace.errors import ArtifactError
from grace.graph.models import GraceState, GraphState, STATE_FORMAT_VERSION
from grace.graph.validation import validate_graph
from grace.providers.base import ProviderAttemptRecord, UsageRecord
from grace.schemas.models import NetworkSchema


ArtifactMode = Literal["checkpoint", "audit", "none"]
ResultKind = Literal["initialization", "evolution"]

ARTIFACT_FORMAT_VERSION = "1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_ID_RE = _SHA256_RE
_SCHEMA_FILENAME = "schema.json"
_SCHEMA_RELATIVE_FROM_STATE = "../../schema.json"
_CORE_FILES = frozenset(
    {
        "graph.json",
        "instruction.txt",
        "result.json",
        "validation_report.json",
    }
)
_EVOLUTION_FILES = frozenset({"change_log.json", "reconstruction_report.json"})
_AUDIT_FILES = frozenset({"audit.json", "prompt_provenance.json", "prompt_snapshots.json"})
_RESULT_FIELDS = {
    "initialization": frozenset({"validation_report"}),
    "evolution": frozenset(
        {"change_log", "validation_report", "reconstruction_report", "provenance"}
    ),
}


def _package_version() -> str:
    try:
        return version("redmind-grace")
    except PackageNotFoundError:
        return SOURCE_VERSION


def _require_sha256(value: str, *, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _load_json_object(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact {path.name} must contain an object")
    return value


class ArtifactStore:
    """Persist and verify one immutable GRACE run.

    ``run_id`` identifies the run-level schema namespace.  Omitting it creates
    a new in-memory identifier; constructing another store to resume an
    existing run therefore requires passing the original ``run_id``.
    """

    def __init__(
        self,
        root: str | Path = "./grace_runs",
        *,
        mode: ArtifactMode = "checkpoint",
        run_id: str | None = None,
        grace_version: str | None = None,
        sensitive_values: Sequence[str] = (),
        allow_raw_model_outputs: bool = False,
    ) -> None:
        if mode not in {"checkpoint", "audit", "none"}:
            raise ArtifactError(f"unsupported artifact mode: {mode!r}")
        resolved_run_id = run_id or f"run-{uuid.uuid4().hex}"
        if not _RUN_ID_RE.fullmatch(resolved_run_id):
            raise ArtifactError("run_id must contain only letters, digits, '.', '_', or '-'")
        resolved_version = grace_version or _package_version()
        if not resolved_version.strip():
            raise ArtifactError("grace_version cannot be empty")

        self.root = Path(root).expanduser().absolute()
        self.mode: ArtifactMode = mode
        self.run_id = resolved_run_id
        self.grace_version = resolved_version
        self.sensitive_values = tuple(item for item in sensitive_values if item)
        self.allow_raw_model_outputs = allow_raw_model_outputs

    @property
    def run_path(self) -> Path:
        return self.root / self.run_id

    @property
    def states_path(self) -> Path:
        return self.run_path / "states"

    @property
    def schema_path(self) -> Path:
        return self.run_path / _SCHEMA_FILENAME

    def write_state(
        self,
        *,
        state: GraceState,
        schema: NetworkSchema,
        result_kind: ResultKind,
        result_payload: Mapping[str, Any],
        prompt_snapshot: Mapping[str, Any] | None = None,
        input_hashes: Mapping[str, str] | None = None,
        config_hash: str | None = None,
        models: Sequence[str] = (),
        usage: Sequence[UsageRecord | Mapping[str, Any]] = (),
        attempts: Sequence[ProviderAttemptRecord | Mapping[str, Any]] = (),
        status: ResultStatus | str = ResultStatus.COMPLETED,
        warnings: Sequence[str] = (),
        audit_payload: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Atomically create an immutable state record.

        Repeating the same logical write returns the existing verified path
        without mutation.  A conflicting record for the same content-derived
        state ID raises instead of overwriting it.
        """

        typed_status = self._coerce_status(status)
        typed_usage = self._coerce_usage(usage)
        typed_attempts = self._coerce_attempts(attempts)
        typed_warnings = tuple(str(item) for item in warnings)
        normalized_inputs = dict(input_hashes or {})
        for name, digest in normalized_inputs.items():
            if not name:
                raise ArtifactError("input hash names cannot be empty")
            _require_sha256(digest, field=f"input_hashes[{name!r}]")
        if config_hash is not None:
            _require_sha256(config_hash, field="config_hash")

        self._validate_state_schema(state=state, schema=schema, result_kind=result_kind)
        full_result, checkpoint_result = self._validate_result_payload(
            state=state,
            result_kind=result_kind,
            result_payload=result_payload,
            usage=typed_usage,
            attempts=typed_attempts,
            status=typed_status,
            warnings=typed_warnings,
        )
        prompt_provenance, normalized_prompts = capture_prompt_provenance(prompt_snapshot)
        normalized_audit = (
            None if audit_payload is None else to_json_object(audit_payload, path="$.audit_payload")
        )
        normalized_models = self._normalize_models(models, typed_usage)

        safety_payload = {
            "state": state,
            "schema": schema,
            "result": full_result,
            "prompt_snapshot": normalized_prompts,
            "audit": normalized_audit,
            "models": normalized_models,
            "usage": typed_usage,
            "attempts": typed_attempts,
        }
        assert_artifact_safe(
            safety_payload,
            sensitive_values=self.sensitive_values,
            allow_raw_model_outputs=self.allow_raw_model_outputs,
        )

        if self.mode == "none":
            return None

        self._prepare_run_directories()
        schema_bytes = canonical_json_bytes(schema.canonical_data())
        self._atomic_create_file(self.schema_path, schema_bytes)
        self._verify_schema_snapshot(schema)

        parent_state: GraceState | None = None
        if state.parent_state_id is not None:
            parent_path = self.find_state(state.parent_state_id)
            if parent_path is None:
                raise ArtifactError(
                    f"parent state {state.parent_state_id} is absent from run {self.run_id}"
                )
            parent_record = self.verify_record(parent_path, expected_schema=schema)
            if parent_record.manifest.mode != self.mode:
                raise ArtifactError(
                    "artifact mode cannot change within one persisted state lineage"
                )
            parent_state = parent_record.state
            state.validate_integrity(parent_state=parent_state)

        if result_kind == "evolution":
            reconstruction = ReconstructionReport.model_validate(
                full_result["reconstruction_report"]
            )
            assert parent_state is not None
            if reconstruction.input_instruction_hash != parent_state.instruction_hash:
                raise ArtifactError(
                    "reconstruction input hash does not match the persisted parent instruction"
                )

        files = self._record_file_payloads(
            state=state,
            checkpoint_result=checkpoint_result,
            full_result=full_result,
            prompt_provenance=prompt_provenance,
            normalized_prompts=normalized_prompts,
            normalized_audit=normalized_audit,
            result_kind=result_kind,
        )
        file_hashes = {name: sha256_bytes(data) for name, data in files.items()}
        file_hashes[_SCHEMA_RELATIVE_FROM_STATE] = sha256_bytes(schema_bytes)
        result_hash = file_hashes["result.json"]
        manifest = ArtifactManifest(
            artifact_format_version=ARTIFACT_FORMAT_VERSION,
            grace_version=self.grace_version,
            state_format_version=state.format_version,
            mode=self.mode,
            result_kind=result_kind,
            run_id=self.run_id,
            state_id=state.state_id,
            parent_state_id=state.parent_state_id,
            step=state.step,
            schema_id=state.schema_id,
            schema_hash=state.schema_hash,
            schema_file=_SCHEMA_RELATIVE_FROM_STATE,
            instruction_hash=state.instruction_hash,
            graph_hash=state.graph_hash,
            config_hash=config_hash,
            input_hashes=normalized_inputs,
            output_hashes={
                "graph": state.graph_hash,
                "instruction": state.instruction_hash,
                "result": result_hash,
                "state": state.state_id,
            },
            prompt_hashes=manifest_prompt_hashes(prompt_provenance),
            models=normalized_models,
            usage=typed_usage,
            attempts=typed_attempts,
            status=typed_status,
            warnings=typed_warnings,
            files=file_hashes,
            created_at=datetime.now(timezone.utc),
        ).with_integrity_hash()

        target = self.states_path / state.state_id
        if target.exists():
            return self._accept_existing_record(
                target=target,
                proposed_manifest=manifest,
                expected_schema=schema,
            )
        temporary = self.states_path / f".tmp-{state.state_id}-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(mode=0o700)
            for name in sorted(files):
                self._write_bytes(temporary / name, files[name])
            self._write_bytes(temporary / "manifest.json", canonical_json_bytes(manifest))
            self._fsync_directory(temporary)
            return self._finalize_record(
                temporary=temporary,
                target=target,
                proposed_manifest=manifest,
                expected_schema=schema,
            )
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError(f"cannot write artifact state {state.state_id}: {exc}") from exc
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def find_state(self, state_id: str) -> Path | None:
        """Return the verified path for ``state_id`` in this run, if present."""

        if not _STATE_ID_RE.fullmatch(state_id):
            raise ArtifactError("state_id must be a lowercase SHA-256 digest")
        if self.mode == "none":
            return None
        path = self.states_path / state_id
        if not path.exists():
            return None
        self.verify_record(path)
        return path

    def verify_record(
        self,
        path_or_state_id: str | Path,
        *,
        expected_schema: NetworkSchema | None = None,
        parent_state: GraceState | None = None,
    ) -> ArtifactRecord:
        """Fully verify a finalized record without modifying it."""

        if self.mode == "none":
            raise ArtifactError("artifact persistence is disabled")
        record_path = self._record_path(path_or_state_id)
        self._require_real_directory(record_path, label="state record")
        manifest_path = record_path / "manifest.json"
        self._require_regular_file(manifest_path)

        try:
            manifest = ArtifactManifest.model_validate(_load_json_object(manifest_path))
            manifest.validate_integrity_hash()
        except (ValidationError, ValueError) as exc:
            raise ArtifactError(f"invalid artifact manifest: {exc}") from exc

        if manifest.artifact_format_version != ARTIFACT_FORMAT_VERSION:
            raise ArtifactError(
                "unsupported artifact format version "
                f"{manifest.artifact_format_version!r}; expected {ARTIFACT_FORMAT_VERSION!r}"
            )
        if manifest.state_format_version != STATE_FORMAT_VERSION:
            raise ArtifactError(
                "unsupported state format version "
                f"{manifest.state_format_version!r}; expected {STATE_FORMAT_VERSION!r}"
            )
        if record_path.name != manifest.state_id:
            raise ArtifactError("state directory name does not match manifest state_id")
        if manifest.config_hash is not None:
            _require_sha256(manifest.config_hash, field="manifest.config_hash")
        for mapping_name, digests in (
            ("input_hashes", manifest.input_hashes),
            ("output_hashes", manifest.output_hashes),
            ("prompt_hashes", manifest.prompt_hashes),
        ):
            for name, digest in digests.items():
                if not name:
                    raise ArtifactError(f"manifest {mapping_name} contains an empty name")
                _require_sha256(digest, field=f"manifest.{mapping_name}[{name!r}]")

        run_path = record_path.parent.parent
        if run_path.name != manifest.run_id:
            raise ArtifactError("run directory name does not match manifest run_id")
        expected_schema_path = run_path / _SCHEMA_FILENAME
        actual_schema_path = Path(os.path.abspath(record_path / manifest.schema_file))
        if actual_schema_path != Path(os.path.abspath(expected_schema_path)):
            raise ArtifactError("manifest schema_file does not reference the run schema")

        expected_local_files = self._expected_local_files(
            result_kind=manifest.result_kind,
            mode=manifest.mode,
        )
        actual_local_files: set[str] = set()
        for child in record_path.iterdir():
            if child.name == "manifest.json":
                continue
            self._require_regular_file(child)
            actual_local_files.add(child.name)
        if actual_local_files != expected_local_files:
            raise ArtifactError(
                "state record file set mismatch: expected "
                f"{sorted(expected_local_files)}, found {sorted(actual_local_files)}"
            )
        expected_manifest_files = expected_local_files | {_SCHEMA_RELATIVE_FROM_STATE}
        if set(manifest.files) != expected_manifest_files:
            raise ArtifactError("manifest files mapping does not match the mode allowlist")

        for relative_name, expected_hash in manifest.files.items():
            _require_sha256(expected_hash, field=f"manifest.files[{relative_name!r}]")
            file_path = (
                expected_schema_path
                if relative_name == _SCHEMA_RELATIVE_FROM_STATE
                else record_path / relative_name
            )
            self._require_regular_file(file_path)
            try:
                actual_hash = sha256_bytes(file_path.read_bytes())
            except OSError as exc:
                raise ArtifactError(f"cannot hash artifact {relative_name}: {exc}") from exc
            if actual_hash != expected_hash:
                raise ArtifactError(f"artifact hash mismatch for {relative_name}")

        schema = self._load_schema(expected_schema_path)
        if schema.id != manifest.schema_id or schema.schema_hash != manifest.schema_hash:
            raise ArtifactError("schema snapshot identity does not match manifest")
        if expected_schema is not None and (
            schema.id != expected_schema.id or schema.schema_hash != expected_schema.schema_hash
        ):
            raise ArtifactError("schema snapshot does not match expected_schema")

        graph = self._load_graph(record_path / "graph.json")
        try:
            instruction = (record_path / "instruction.txt").read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ArtifactError(f"cannot read instruction artifact: {exc}") from exc
        try:
            state = GraceState.from_parts(
                graph=graph,
                instruction=instruction,
                schema_id=manifest.schema_id,
                schema_hash=manifest.schema_hash,
                step=manifest.step,
                parent_state_id=manifest.parent_state_id,
                format_version=manifest.state_format_version,
            )
            state.validate_integrity(
                schema_id=schema.id,
                schema_hash=schema.schema_hash,
                parent_state=parent_state,
            )
        except Exception as exc:
            raise ArtifactError(f"persisted state failed integrity validation: {exc}") from exc
        if state.state_id != manifest.state_id:
            raise ArtifactError("reconstructed state_id does not match manifest")
        if state.graph_hash != manifest.graph_hash:
            raise ArtifactError("reconstructed graph_hash does not match manifest")
        if state.instruction_hash != manifest.instruction_hash:
            raise ArtifactError("reconstructed instruction_hash does not match manifest")
        graph_report = validate_graph(state.graph, schema)
        if not graph_report.valid:
            raise ArtifactError("persisted graph is invalid under the schema snapshot")

        checkpoint_result = _load_json_object(record_path / "result.json")
        audit_payload: JsonObject | None = None
        full_result = checkpoint_result
        prompt_provenance: PromptProvenance | None = None
        if manifest.mode == "audit":
            audit_wrapper = _load_json_object(record_path / "audit.json")
            if set(audit_wrapper) != {"audit_payload", "result_payload"}:
                raise ArtifactError("audit.json has an unexpected shape")
            raw_result = audit_wrapper["result_payload"]
            raw_audit = audit_wrapper["audit_payload"]
            if not isinstance(raw_result, dict) or not isinstance(raw_audit, dict):
                raise ArtifactError("audit result_payload and audit_payload must be objects")
            full_result = raw_result
            audit_payload = raw_audit
            regenerated_projection = self._checkpoint_projection(
                result_kind=manifest.result_kind,
                full_result=full_result,
            )
            if regenerated_projection != checkpoint_result:
                raise ArtifactError("audit result payload does not match checkpoint projection")

            try:
                prompt_provenance = PromptProvenance.model_validate(
                    _load_json_object(record_path / "prompt_provenance.json")
                )
            except ValidationError as exc:
                raise ArtifactError(f"invalid prompt provenance: {exc}") from exc
            snapshots = _load_json_object(record_path / "prompt_snapshots.json")
            if prompt_provenance.captured:
                if set(snapshots) != set(prompt_provenance.entry_order):
                    raise ArtifactError("prompt snapshot keys do not match prompt provenance")
                supplied_snapshot = {key: snapshots[key] for key in prompt_provenance.entry_order}
            else:
                supplied_snapshot = None
            regenerated, normalized = capture_prompt_provenance(supplied_snapshot)
            if regenerated != prompt_provenance:
                raise ArtifactError("prompt snapshot does not match prompt provenance")
            if normalized != supplied_snapshot:
                raise ArtifactError("prompt snapshot normalization mismatch")
            if manifest.prompt_hashes != manifest_prompt_hashes(prompt_provenance):
                raise ArtifactError("prompt provenance does not match manifest prompt hashes")

        validated_full, regenerated_checkpoint = self._validate_result_payload(
            state=state,
            result_kind=manifest.result_kind,
            result_payload=full_result,
            usage=manifest.usage,
            attempts=manifest.attempts,
            status=manifest.status,
            warnings=manifest.warnings,
        )
        if regenerated_checkpoint != checkpoint_result:
            raise ArtifactError("persisted result payload is not canonical")
        expected_outputs = {
            "graph": state.graph_hash,
            "instruction": state.instruction_hash,
            "result": manifest.files["result.json"],
            "state": state.state_id,
        }
        if manifest.output_hashes != expected_outputs:
            raise ArtifactError("manifest output_hashes do not match persisted outputs")

        assert_artifact_safe(
            {
                "manifest": manifest,
                "state": state,
                "schema": schema,
                "result": validated_full,
                "audit": audit_payload,
            },
            sensitive_values=self.sensitive_values,
            allow_raw_model_outputs=self.allow_raw_model_outputs,
        )
        return ArtifactRecord(
            path=record_path,
            manifest=manifest,
            state=state,
            schema_snapshot=schema,
            result_payload=validated_full,
            prompt_provenance=prompt_provenance,
            audit_payload=audit_payload,
        )

    def load_state(
        self,
        path_or_state_id: str | Path,
        *,
        expected_schema: NetworkSchema | None = None,
        parent_state: GraceState | None = None,
    ) -> GraceState:
        """Load one state only after complete artifact verification."""

        return self.verify_record(
            path_or_state_id,
            expected_schema=expected_schema,
            parent_state=parent_state,
        ).state

    def resume(
        self,
        path_or_state_id: str | Path,
        *,
        expected_schema: NetworkSchema | None = None,
        parent_state: GraceState | None = None,
    ) -> GraceState:
        """Compatibility alias emphasizing zero-mutation checkpoint resume."""

        return self.load_state(
            path_or_state_id,
            expected_schema=expected_schema,
            parent_state=parent_state,
        )

    def load_parent(
        self,
        path_or_state_id: str | Path,
        *,
        expected_schema: NetworkSchema | None = None,
    ) -> GraceState | None:
        """Load the verified parent checkpoint for rollback or inspection."""

        record = self.verify_record(path_or_state_id, expected_schema=expected_schema)
        parent_id = record.state.parent_state_id
        if parent_id is None:
            return None
        parent = self.load_state(parent_id, expected_schema=record.schema_snapshot)
        record.state.validate_integrity(parent_state=parent)
        return parent

    def _validate_state_schema(
        self,
        *,
        state: GraceState,
        schema: NetworkSchema,
        result_kind: ResultKind,
    ) -> None:
        try:
            state.validate_integrity(schema_id=schema.id, schema_hash=schema.schema_hash)
        except Exception as exc:
            raise ArtifactError(f"state/schema integrity validation failed: {exc}") from exc
        report = validate_graph(state.graph, schema)
        if not report.valid:
            raise ArtifactError("cannot persist a graph invalid under its schema")
        if result_kind == "initialization" and state.step != 0:
            raise ArtifactError("initialization artifacts require state step 0")
        if result_kind == "evolution" and state.step == 0:
            raise ArtifactError("evolution artifacts require a state after step 0")

    def _validate_result_payload(
        self,
        *,
        state: GraceState,
        result_kind: ResultKind,
        result_payload: Mapping[str, Any],
        usage: tuple[UsageRecord, ...],
        attempts: tuple[ProviderAttemptRecord, ...],
        status: ResultStatus,
        warnings: tuple[str, ...],
    ) -> tuple[JsonObject, JsonObject]:
        if result_kind not in _RESULT_FIELDS:
            raise ArtifactError(f"unsupported result_kind: {result_kind!r}")
        full_result = to_json_object(result_payload, path="$.result_payload")
        expected = _RESULT_FIELDS[result_kind]
        actual = set(full_result)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ArtifactError(
                f"{result_kind} result_payload is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ArtifactError(
                f"{result_kind} result_payload contains non-public fields "
                f"{sorted(extra)}; place audit-only telemetry in audit_payload"
            )
        try:
            validation = ValidationReport.model_validate(full_result["validation_report"])
            if result_kind == "initialization":
                InitializationResult(
                    state=state,
                    validation_report=validation,
                    usage=usage,
                    attempts=attempts,
                    status=status,
                    warnings=warnings,
                )
            else:
                raw_change_log_value = full_result["change_log"]
                if not isinstance(raw_change_log_value, list) or not all(
                    isinstance(item, dict) for item in raw_change_log_value
                ):
                    raise ValueError("change_log must be a sequence of JSON objects")
                raw_change_log = cast(list[JsonObject], raw_change_log_value)
                reconstruction = ReconstructionReport.model_validate(
                    full_result["reconstruction_report"]
                )
                provenance_value = full_result["provenance"]
                if not isinstance(provenance_value, dict):
                    raise ValueError("provenance must be a JSON object")
                provenance = cast(JsonObject, provenance_value)
                EvolutionResult(
                    state=state,
                    change_log=tuple(raw_change_log),
                    validation_report=validation,
                    reconstruction_report=reconstruction,
                    provenance=provenance,
                    usage=usage,
                    attempts=attempts,
                    status=status,
                    warnings=warnings,
                )
        except (ValidationError, ValueError) as exc:
            raise ArtifactError(f"invalid {result_kind} result payload: {exc}") from exc
        return full_result, self._checkpoint_projection(
            result_kind=result_kind,
            full_result=full_result,
        )

    @staticmethod
    def _checkpoint_projection(*, result_kind: ResultKind, full_result: JsonObject) -> JsonObject:
        validation_value = full_result["validation_report"]
        if not isinstance(validation_value, dict):
            raise ArtifactError("validation_report must be a JSON object")
        validation = dict(cast(JsonObject, validation_value))
        validation.pop("history", None)
        validation.pop("examined_node_ids", None)
        projected: JsonObject = {"validation_report": validation}
        if result_kind == "evolution":
            projected["change_log"] = full_result["change_log"]
            projected["reconstruction_report"] = full_result["reconstruction_report"]
            projected["provenance"] = full_result["provenance"]
        return projected

    def _record_file_payloads(
        self,
        *,
        state: GraceState,
        checkpoint_result: JsonObject,
        full_result: JsonObject,
        prompt_provenance: PromptProvenance,
        normalized_prompts: JsonObject | None,
        normalized_audit: JsonObject | None,
        result_kind: ResultKind,
    ) -> dict[str, bytes]:
        files = {
            # Preserve the accepted node/edge tuple order for an exact state
            # round trip.  ``state.graph_hash`` remains order-independent;
            # the per-file hash additionally binds this serialized ordering.
            "graph.json": canonical_json_bytes(state.graph.model_dump(mode="json")),
            "instruction.txt": state.instruction.encode("utf-8"),
            "result.json": canonical_json_bytes(checkpoint_result),
            "validation_report.json": canonical_json_bytes(checkpoint_result["validation_report"]),
        }
        if result_kind == "evolution":
            files["change_log.json"] = canonical_json_bytes(checkpoint_result["change_log"])
            files["reconstruction_report.json"] = canonical_json_bytes(
                checkpoint_result["reconstruction_report"]
            )
        if self.mode == "audit":
            files["audit.json"] = canonical_json_bytes(
                {
                    "audit_payload": normalized_audit or {},
                    "result_payload": full_result,
                }
            )
            files["prompt_provenance.json"] = canonical_json_bytes(prompt_provenance)
            files["prompt_snapshots.json"] = canonical_json_bytes(normalized_prompts or {})
        return files

    @staticmethod
    def _coerce_status(status: ResultStatus | str) -> ResultStatus:
        try:
            return status if isinstance(status, ResultStatus) else ResultStatus(status)
        except ValueError as exc:
            raise ArtifactError(f"unsupported result status: {status!r}") from exc

    @staticmethod
    def _coerce_usage(
        usage: Sequence[UsageRecord | Mapping[str, Any]],
    ) -> tuple[UsageRecord, ...]:
        try:
            return tuple(
                item if isinstance(item, UsageRecord) else UsageRecord.model_validate(item)
                for item in usage
            )
        except ValidationError as exc:
            raise ArtifactError(f"invalid usage record: {exc}") from exc

    @staticmethod
    def _coerce_attempts(
        attempts: Sequence[ProviderAttemptRecord | Mapping[str, Any]],
    ) -> tuple[ProviderAttemptRecord, ...]:
        try:
            return tuple(
                item
                if isinstance(item, ProviderAttemptRecord)
                else ProviderAttemptRecord.model_validate(item)
                for item in attempts
            )
        except ValidationError as exc:
            raise ArtifactError(f"invalid provider attempt record: {exc}") from exc

    @staticmethod
    def _normalize_models(models: Sequence[str], usage: tuple[UsageRecord, ...]) -> tuple[str, ...]:
        ordered: list[str] = []
        for model in (*models, *(record.model for record in usage)):
            normalized = str(model).strip()
            if not normalized:
                raise ArtifactError("model identifiers cannot be empty")
            if normalized not in ordered:
                ordered.append(normalized)
        declared = set(ordered)
        missing = {record.model for record in usage} - declared
        if missing:  # defensive; usage models are added above
            raise ArtifactError(f"usage contains undeclared models: {sorted(missing)}")
        return tuple(ordered)

    def _prepare_run_directories(self) -> None:
        self._ensure_directory(self.root)
        self._ensure_directory(self.run_path)
        self._ensure_directory(self.states_path)

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(f"cannot create artifact directory {path}: {exc}") from exc
        if path.is_symlink() or not path.is_dir():
            raise ArtifactError(f"artifact directory is not a real directory: {path}")

    @staticmethod
    def _require_real_directory(path: Path, *, label: str) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ArtifactError(f"{label} is missing or is not a real directory: {path}")

    @staticmethod
    def _require_regular_file(path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"artifact file is missing or not regular: {path}")

    def _atomic_create_file(self, target: Path, data: bytes) -> None:
        """Create ``target`` without replacement, using a durable temp + link."""

        if target.exists():
            self._require_regular_file(target)
            if target.read_bytes() != data:
                raise ArtifactError(
                    f"immutable artifact file already exists with different content: {target}"
                )
            return
        temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        try:
            self._write_bytes(temporary, data)
            try:
                os.link(temporary, target)
            except FileExistsError:
                self._require_regular_file(target)
                if target.read_bytes() != data:
                    raise ArtifactError(
                        f"immutable artifact file already exists with different content: {target}"
                    )
            else:
                self._fsync_directory(target.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short artifact write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _finalize_record(
        self,
        *,
        temporary: Path,
        target: Path,
        proposed_manifest: ArtifactManifest,
        expected_schema: NetworkSchema,
    ) -> Path:
        collision = target.exists()
        if not collision:
            try:
                os.rename(temporary, target)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                collision = True
            else:
                self._fsync_directory(target.parent)
                self.verify_record(target, expected_schema=expected_schema)
                return target

        return self._accept_existing_record(
            target=target,
            proposed_manifest=proposed_manifest,
            expected_schema=expected_schema,
        )

    def _accept_existing_record(
        self,
        *,
        target: Path,
        proposed_manifest: ArtifactManifest,
        expected_schema: NetworkSchema,
    ) -> Path:
        existing = self.verify_record(target, expected_schema=expected_schema)
        existing_payload = existing.manifest.model_dump(
            mode="json", exclude={"created_at", "manifest_hash"}
        )
        proposed_payload = proposed_manifest.model_dump(
            mode="json", exclude={"created_at", "manifest_hash"}
        )
        if existing_payload != proposed_payload:
            raise ArtifactError(
                f"immutable state {proposed_manifest.state_id} already exists with "
                "different artifact content"
            )
        return target

    def _verify_schema_snapshot(self, expected: NetworkSchema) -> None:
        actual = self._load_schema(self.schema_path)
        if actual.id != expected.id or actual.schema_hash != expected.schema_hash:
            raise ArtifactError("run schema snapshot differs from the requested schema")

    @staticmethod
    def _load_schema(path: Path) -> NetworkSchema:
        try:
            schema = NetworkSchema.model_validate(_load_json_object(path))
        except ValidationError as exc:
            raise ArtifactError(f"invalid schema snapshot: {exc}") from exc
        canonical = canonical_json_bytes(schema.canonical_data())
        try:
            persisted = path.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"cannot read schema snapshot: {exc}") from exc
        if persisted != canonical:
            raise ArtifactError("schema snapshot is not in canonical form")
        return schema

    @staticmethod
    def _load_graph(path: Path) -> GraphState:
        try:
            graph = GraphState.model_validate(_load_json_object(path))
        except ValidationError as exc:
            raise ArtifactError(f"invalid graph artifact: {exc}") from exc
        if path.read_bytes() != canonical_json_bytes(graph.model_dump(mode="json")):
            raise ArtifactError("graph artifact is not in canonical form")
        return graph

    @staticmethod
    def _expected_local_files(
        *, result_kind: ResultKind, mode: Literal["checkpoint", "audit"]
    ) -> set[str]:
        expected = set(_CORE_FILES)
        if result_kind == "evolution":
            expected.update(_EVOLUTION_FILES)
        if mode == "audit":
            expected.update(_AUDIT_FILES)
        return expected

    def _record_path(self, path_or_state_id: str | Path) -> Path:
        raw = Path(path_or_state_id)
        if isinstance(path_or_state_id, str) and _STATE_ID_RE.fullmatch(path_or_state_id):
            return self.states_path / path_or_state_id
        path = raw.expanduser().absolute()
        if path.name == "manifest.json":
            path = path.parent
        return path


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ArtifactMode",
    "ArtifactStore",
    "ResultKind",
]
