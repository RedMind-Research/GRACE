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

import hashlib
from pathlib import Path

import pytest

from grace.providers.base import PromptRequest, ProviderCallResult, UsageRecord
from reproduction.tau2_telecom.harness.completeness import build_expected_matrix
from reproduction.tau2_telecom.harness.diagnosis import (
    DiagnosisArtifactError,
    DiagnosisFingerprintMismatch,
    run_diagnosis,
)
from reproduction.tau2_telecom.harness.identity import canonical_sha256
from reproduction.tau2_telecom.harness.models import (
    DiagnosisStatus,
    TaskSlotSpec,
    TaskSelectionManifest,
    TrajectoryMessage,
    TrajectoryRecord,
)
from reproduction.tau2_telecom.harness.tasks import resolve_task_selection


CONFIG_HASH = canonical_sha256({"profile": "diagnosis-test"})
POLICY = "Inspect available state, use supported tools, and give verified next steps."


class ScriptedProvider:
    def __init__(self, script, *, model="vertex_ai/gemini-2.5-flash"):
        self.script = list(script)
        self.calls: list[PromptRequest] = []
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        self.calls.append(request)
        if not self.script:
            raise AssertionError("unexpected provider call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ProviderCallResult(
            parsed_content=item,
            usage=UsageRecord(
                model=self.model,
                input_tokens=10 + len(self.calls),
                output_tokens=5,
                latency_seconds=0.01,
            ),
        )


def _bundle():
    slots = tuple(
        TaskSlotSpec(
            split="experience",
            batch=0,
            slot=index,
            phase="A",
            benchmark_task_id=f"synthetic-telecom-task-{index}",
        )
        for index in range(4)
    )
    selection = TaskSelectionManifest(
        benchmark_revision="c5b2d228d850c59b749b93cf32c4745d3aa53967",
        seed=1024,
        tasks=slots,
    )
    source = tuple(
        {"id": slot.benchmark_task_id, "scenario": f"synthetic-{slot.slot}"}
        for slot in reversed(slots)
    )
    task_map = resolve_task_selection(selection, source, publication_status="public").task_map
    hashes = {
        name: canonical_sha256({"name": name}) for name in ("protocol", "models", "evaluator")
    }
    matrix = build_expected_matrix(
        task_map,
        trials=(0,),
        seed=1024,
        checkpoint_state_id="historical-batch1-policy",
        protocol_hash=hashes["protocol"],
        config_hash=CONFIG_HASH,
        model_roles_hash=hashes["models"],
        evaluator_hash=hashes["evaluator"],
    )
    tool_names = ("toggle_airplane_mode", None, "run_speed_test", "reseat_sim_card")
    records = tuple(
        TrajectoryRecord(
            episode_id=cell.episode_id,
            task_uid=cell.task_uid,
            task_definition_hash=cell.task_definition_hash,
            benchmark_task_id=cell.benchmark_task_id,
            split_slot=cell.split_slot,
            trial=cell.trial,
            protocol_seed=cell.protocol_seed,
            success=tool_name is None,
            termination_reason="agent_stop",
            messages=(
                TrajectoryMessage(
                    role="assistant",
                    content=(
                        "Resolved using supported evidence."
                        if tool_name is None
                        else f"Error: Tool '{tool_name}' not found"
                    ),
                ),
            ),
        )
        for cell, tool_name in zip(matrix.episodes, tool_names, strict=True)
    )
    return POLICY, matrix, records


def _reflection(task_id: str, index: int):
    return {
        "task_id": task_id,
        "subtasks_attempted": ["inspect_state"],
        "subtasks_missed": ["complete_resolution"],
        "findings": [
            {
                "finding_id": f"F{index}",
                "affected_subtasks": ["complete_resolution"],
                "description": "The trajectory lacks a supported completion path.",
                "proposed_instruction": "Use an available tool or give a verified manual step.",
            }
        ],
    }


def _synthesis():
    return {
        "themes": [
            {
                "theme_id": "T1",
                "title": "Unsupported completion actions",
                "frequency": 3,
                "affected_subtask_types": ["complete_resolution"],
                "root_cause": "The policy does not define a supported fallback after unavailable tools.",
                "recommended_actions": [
                    "When a tool is unavailable, use an available diagnostic or give a safe manual action."
                ],
            }
        ],
        "summary": "The failures share an unsupported completion path after diagnosis.",
    }


def _success_script(records):
    failures = [record for record in records if not record.success]
    return [
        *[_reflection(record.benchmark_task_id, index) for index, record in enumerate(failures, 1)],
        _synthesis(),
    ]


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_synthetic_diagnosis_runs_three_reflections_then_one_synthesis(tmp_path) -> None:
    policy, matrix, records = _bundle()
    provider = ScriptedProvider(_success_script(records))

    result = run_diagnosis(
        provider=provider,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=tmp_path / "diagnosis",
        config_hash=CONFIG_HASH,
    )

    assert result.status == DiagnosisStatus.COMPLETE
    assert len(provider.calls) == 4
    assert [request.stage for request in provider.calls] == [
        "diagnosis_reflection",
        "diagnosis_reflection",
        "diagnosis_reflection",
        "diagnosis_synthesis",
    ]
    assert all(request.logical_call_id is not None for request in provider.calls)
    assert len({request.logical_call_id for request in provider.calls}) == 4
    assert [request.max_tokens for request in provider.calls] == [65536, 65536, 65536, 65536]
    assert len(result.reflections) == 3
    assert [item.benchmark_task_id for item in result.reflections] == [
        records[0].benchmark_task_id,
        records[2].benchmark_task_id,
        records[3].benchmark_task_id,
    ]
    assert result.manifest.classifications[1].status.value == "successful_control"
    assert result.synthesis is not None
    assert result.synthesis.observed_tool_error_evidence.total_tool_not_found_calls == 3
    assert {item.tool for item in result.synthesis.observed_tool_error_evidence.tools} == {
        "toggle_airplane_mode",
        "run_speed_test",
        "reseat_sim_card",
    }
    reflection_files = sorted((tmp_path / "diagnosis" / "reflections").glob("*.json"))
    assert len(reflection_files) == 3
    assert all(len(path.stem) == 64 for path in reflection_files)
    assert (tmp_path / "diagnosis" / "complete.json").exists()


def test_complete_resume_makes_zero_calls_and_zero_mutation(tmp_path) -> None:
    policy, matrix, records = _bundle()
    artifact_dir = tmp_path / "diagnosis"
    first = ScriptedProvider(_success_script(records))
    run_diagnosis(
        provider=first,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )
    before = _tree_hash(artifact_dir)
    resumed = ScriptedProvider([])

    result = run_diagnosis(
        provider=resumed,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )

    assert result.status == DiagnosisStatus.COMPLETE
    assert resumed.calls == []
    assert _tree_hash(artifact_dir) == before


def test_partial_reflection_failure_never_synthesizes_and_resume_retries_only_missing(
    tmp_path,
) -> None:
    policy, matrix, records = _bundle()
    failures = [record for record in records if not record.success]
    artifact_dir = tmp_path / "diagnosis"
    first = ScriptedProvider(
        [
            _reflection(failures[0].benchmark_task_id, 1),
            RuntimeError("secret provider detail must not persist"),
            _reflection(failures[2].benchmark_task_id, 3),
        ]
    )

    incomplete = run_diagnosis(
        provider=first,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )

    assert incomplete.status == DiagnosisStatus.INCOMPLETE
    assert len(first.calls) == 3
    assert len(incomplete.reflections) == 2
    assert not (artifact_dir / "synthesis.json").exists()
    assert not (artifact_dir / "complete.json").exists()
    assert "secret provider detail" not in (artifact_dir / "manifest.json").read_text()

    resumed = ScriptedProvider([_reflection(failures[1].benchmark_task_id, 2), _synthesis()])
    complete = run_diagnosis(
        provider=resumed,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )

    assert complete.status == DiagnosisStatus.COMPLETE
    assert len(resumed.calls) == 2
    assert len(complete.reflections) == 3


def test_synthesis_failure_preserves_reflections_and_resume_retries_only_synthesis(
    tmp_path,
) -> None:
    policy, matrix, records = _bundle()
    artifact_dir = tmp_path / "diagnosis"
    first = ScriptedProvider([*_success_script(records)[:-1], RuntimeError("failed")])

    incomplete = run_diagnosis(
        provider=first,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )
    assert incomplete.status == DiagnosisStatus.INCOMPLETE
    assert len(list((artifact_dir / "reflections").glob("*.json"))) == 3

    resumed = ScriptedProvider([_synthesis()])
    complete = run_diagnosis(
        provider=resumed,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )
    assert complete.status == DiagnosisStatus.COMPLETE
    assert len(resumed.calls) == 1


@pytest.mark.parametrize("mutation", ["missing", "infrastructure"])
def test_invalid_or_infrastructure_input_blocks_every_provider_call(tmp_path, mutation) -> None:
    policy, matrix, records = _bundle()
    if mutation == "missing":
        changed = records[:-1]
    else:
        changed = (
            records[0].model_copy(update={"termination_reason": "error"}),
            *records[1:],
        )
    provider = ScriptedProvider([])

    result = run_diagnosis(
        provider=provider,
        expected=matrix,
        trajectories=changed,
        policy=policy,
        artifact_dir=tmp_path / mutation,
        config_hash=CONFIG_HASH,
    )

    assert result.status == DiagnosisStatus.INCOMPLETE
    assert provider.calls == []
    assert not (tmp_path / mutation / "synthesis.json").exists()


def test_all_success_is_explicit_zero_call_no_update(tmp_path) -> None:
    policy, matrix, records = _bundle()
    successful = tuple(record.model_copy(update={"success": True}) for record in records)
    artifact_dir = tmp_path / "no-update"
    provider = ScriptedProvider([])

    result = run_diagnosis(
        provider=provider,
        expected=matrix,
        trajectories=successful,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )

    assert result.status == DiagnosisStatus.NO_UPDATE
    assert provider.calls == []
    assert result.manifest.ordered_reflection_ids == ()
    assert (artifact_dir / "complete.json").exists()


def test_resume_rejects_changed_policy_fingerprint(tmp_path) -> None:
    policy, matrix, records = _bundle()
    artifact_dir = tmp_path / "diagnosis"
    run_diagnosis(
        provider=ScriptedProvider(_success_script(records)),
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )

    with pytest.raises(DiagnosisFingerprintMismatch):
        run_diagnosis(
            provider=ScriptedProvider([]),
            expected=matrix,
            trajectories=records,
            policy=policy + " Changed.",
            artifact_dir=artifact_dir,
            config_hash=CONFIG_HASH,
        )


def test_resume_fingerprint_binds_success_control_trajectory_content(tmp_path) -> None:
    policy, matrix, records = _bundle()
    artifact_dir = tmp_path / "diagnosis"
    run_diagnosis(
        provider=ScriptedProvider(_success_script(records)),
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )
    control = records[1]
    changed_messages = tuple(
        message.model_copy(update={"content": "Changed successful-control evidence."})
        if index == 0
        else message
        for index, message in enumerate(control.messages)
    )
    changed = (
        records[0],
        control.model_copy(update={"messages": changed_messages}),
        *records[2:],
    )

    with pytest.raises(DiagnosisFingerprintMismatch):
        run_diagnosis(
            provider=ScriptedProvider([]),
            expected=matrix,
            trajectories=changed,
            policy=policy,
            artifact_dir=artifact_dir,
            config_hash=CONFIG_HASH,
        )


def test_completion_marker_binds_report_and_detects_tampering(tmp_path) -> None:
    policy, matrix, records = _bundle()
    artifact_dir = tmp_path / "diagnosis"
    run_diagnosis(
        provider=ScriptedProvider(_success_script(records)),
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )
    (artifact_dir / "diagnosis_report.md").write_text("tampered\n")

    with pytest.raises(DiagnosisArtifactError, match="completion marker"):
        run_diagnosis(
            provider=ScriptedProvider([]),
            expected=matrix,
            trajectories=records,
            policy=policy,
            artifact_dir=artifact_dir,
            config_hash=CONFIG_HASH,
        )


def test_interrupted_finalization_recreates_only_completion_marker(tmp_path) -> None:
    policy, matrix, records = _bundle()
    artifact_dir = tmp_path / "diagnosis"
    run_diagnosis(
        provider=ScriptedProvider(_success_script(records)),
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )
    marker = artifact_dir / "complete.json"
    marker.unlink()
    provider = ScriptedProvider([])

    result = run_diagnosis(
        provider=provider,
        expected=matrix,
        trajectories=records,
        policy=policy,
        artifact_dir=artifact_dir,
        config_hash=CONFIG_HASH,
    )

    assert result.status == DiagnosisStatus.COMPLETE
    assert provider.calls == []
    assert marker.exists()
