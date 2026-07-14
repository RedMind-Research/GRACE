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

from __future__ import annotations

import json
from collections import deque

import pytest

from grace.config import GraceConfig
from grace.engine import GraceEngine
from grace.errors import ArtifactError, SchemaValidationError
from grace.providers.base import PromptRequest, ProviderCallResult, UsageRecord
from grace.schemas.default import DefaultGraceSchema


class ScriptedProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.requests: list[PromptRequest] = []

    @property
    def model(self) -> str:
        return "scripted/grace-engine-test"

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        self.requests.append(request)
        response = self.responses.popleft()
        return ProviderCallResult(
            parsed_content=response,
            usage=UsageRecord(
                model=self.model,
                input_tokens=10,
                output_tokens=5,
                latency_seconds=0.01,
            ),
        )


def _initial_graph() -> dict:
    return {
        "nodes": [
            {"id": "N001", "type": "identity", "content": "You are an agent."},
            {"id": "N002", "type": "norm", "content": "Verify first."},
        ],
        "edges": [
            {"source": "N001", "target": "N002", "relation": "supports"},
        ],
    }


def _fidelity_pass() -> dict:
    return {
        "missing_node": [],
        "unfaithful": [],
        "mistyped": [],
        "wrong_relation": [],
        "missing_relation": [],
    }


def test_public_engine_persists_and_resumes_across_process_boundary(tmp_path) -> None:
    initializer = ScriptedProvider([_initial_graph(), _fidelity_pass()])
    first = GraceEngine(
        provider=initializer,
        artifact_dir=tmp_path,
        artifact_mode="audit",
        run_id="engine-integration",
    )
    initialized = first.initialize(instruction="You are an agent.\nVerify first.")

    assert initialized.artifact_location is not None
    init_record = first.artifacts.verify_record(initialized.artifact_location)
    assert init_record.manifest.input_hashes["instruction"] == (initialized.state.instruction_hash)
    assert init_record.prompt_provenance is not None
    assert init_record.prompt_provenance.entry_order == ("call_001", "call_002")

    evolver = ScriptedProvider(
        [
            {
                "operations": [
                    {
                        "op": "AddNode",
                        "type": "knowledge",
                        "content": "Inspect the account before acting.",
                    }
                ]
            },
            {"operations": []},
            {"contradictions": [], "redundancies": []},
            {
                "operations": [
                    {
                        "op": "InsertAfter",
                        "anchor": "Verify first.",
                        "new_text": "Inspect the account before acting.",
                    }
                ]
            },
        ]
    )
    resumed = GraceEngine(
        provider=evolver,
        schema=DefaultGraceSchema(),
        artifact_dir=tmp_path,
        artifact_mode="audit",
        run_id=first.run_id,
    )
    evolved = resumed.evolve(
        state=initialized.state,
        diagnosis_report="The agent acted without inspecting the account.",
    )

    assert evolved.artifact_location is not None
    record = resumed.artifacts.verify_record(evolved.artifact_location)
    assert record.state.parent_state_id == initialized.state.state_id
    assert record.manifest.input_hashes["parent_state"] == initialized.state.state_id
    snapshots = json.loads(
        (evolved.artifact_location / "prompt_snapshots.json").read_text(encoding="utf-8")
    )
    assert [entry["stage"] for entry in snapshots.values()] == [
        "operation_planning",
        "relation_maintenance",
        "structural_analysis_1",
        "patch_reconstruction",
    ]


def test_public_engine_forwards_configured_max_output_tokens_to_p2g(tmp_path) -> None:
    provider = ScriptedProvider([_initial_graph(), _fidelity_pass()])
    engine = GraceEngine(
        provider=provider,
        config=GraceConfig(max_output_tokens=4096),
        artifact_dir=tmp_path,
        artifact_mode="none",
    )

    engine.initialize(instruction="You are an agent.\nVerify first.")

    assert [request.max_tokens for request in provider.requests] == [4096, 4096]


def test_max_output_tokens_participates_in_persisted_method_config_hash(tmp_path) -> None:
    default_engine = GraceEngine(
        provider=ScriptedProvider([]),
        artifact_dir=tmp_path / "default",
        run_id="default-token-limit",
    )
    custom_engine = GraceEngine(
        provider=ScriptedProvider([]),
        config=GraceConfig(max_output_tokens=4096),
        artifact_dir=tmp_path / "custom",
        run_id="custom-token-limit",
    )

    default_result = default_engine.initialize_from_graph(
        graph=_initial_graph(),
        instruction="You are an agent.\nVerify first.",
    )
    custom_result = custom_engine.initialize_from_graph(
        graph=_initial_graph(),
        instruction="You are an agent.\nVerify first.",
    )

    assert default_result.state.state_id == custom_result.state.state_id
    assert default_result.artifact_location is not None
    assert custom_result.artifact_location is not None
    default_record = default_engine.artifacts.verify_record(default_result.artifact_location)
    custom_record = custom_engine.artifacts.verify_record(custom_result.artifact_location)
    assert default_record.manifest.config_hash != custom_record.manifest.config_hash


def test_public_engine_persists_no_update_as_zero_call_child(tmp_path) -> None:
    provider = ScriptedProvider([])
    engine = GraceEngine(
        provider=provider,
        artifact_dir=tmp_path,
        artifact_mode="audit",
        run_id="no-update-lineage",
    )
    initialized = engine.initialize_from_graph(
        graph=_initial_graph(),
        instruction="You are an agent.\nVerify first.",
    )

    report = "No failed trajectories were observed in this batch."
    advanced = engine.advance_no_update(
        state=initialized.state,
        diagnosis_report=report,
    )

    assert provider.requests == []
    assert advanced.state.step == 1
    assert advanced.state.parent_state_id == initialized.state.state_id
    assert advanced.state.graph_hash == initialized.state.graph_hash
    assert advanced.state.instruction_hash == initialized.state.instruction_hash
    assert advanced.artifact_location is not None
    record = engine.artifacts.verify_record(advanced.artifact_location)
    assert record.manifest.input_hashes == {
        "diagnosis_report": advanced.provenance["diagnosis_sha256"],
        "parent_state": initialized.state.state_id,
    }
    assert record.manifest.models == ()
    assert record.manifest.usage == ()
    assert record.prompt_provenance is not None
    assert record.prompt_provenance.captured is False


def test_public_engine_rejects_unknown_parent_before_provider_dispatch(tmp_path) -> None:
    initializer = ScriptedProvider([_initial_graph(), _fidelity_pass()])
    source = GraceEngine(
        provider=initializer,
        artifact_dir=tmp_path / "source",
        artifact_mode="checkpoint",
        run_id="source-run",
    )
    state = source.initialize(instruction="You are an agent.\nVerify first.").state

    unused = ScriptedProvider([])
    wrong_run = GraceEngine(
        provider=unused,
        artifact_dir=tmp_path / "other",
        artifact_mode="checkpoint",
        run_id="other-run",
    )
    with pytest.raises(ArtifactError, match="not present"):
        wrong_run.evolve(state=state, diagnosis_report="A valid diagnosis.")
    assert unused.requests == []


def test_public_engine_adopts_provided_graph_without_provider_call(tmp_path) -> None:
    unused = ScriptedProvider([])
    engine = GraceEngine(
        provider=unused,
        artifact_dir=tmp_path,
        artifact_mode="audit",
        run_id="provided-graph",
    )

    initialized = engine.initialize_from_graph(
        graph=_initial_graph(),
        instruction="You are an agent.\nVerify first.",
    )

    assert unused.requests == []
    assert initialized.usage == ()
    assert initialized.attempts == ()
    assert initialized.artifact_location is not None
    record = engine.artifacts.verify_record(initialized.artifact_location)
    assert record.state == initialized.state
    assert record.manifest.models == ()
    assert record.manifest.input_hashes == {
        "instruction": initialized.state.instruction_hash,
        "provided_graph": initialized.state.graph_hash,
    }
    assert record.prompt_provenance is not None
    assert record.prompt_provenance.captured is False


def test_public_engine_rejects_invalid_provided_graph_before_artifact_write(tmp_path) -> None:
    unused = ScriptedProvider([])
    engine = GraceEngine(
        provider=unused,
        artifact_dir=tmp_path,
        artifact_mode="checkpoint",
        run_id="invalid-provided-graph",
    )
    invalid = {
        "nodes": [{"id": "N001", "type": "unknown", "content": "Invalid."}],
        "edges": [],
    }

    with pytest.raises(SchemaValidationError, match="provided graph is invalid"):
        engine.initialize_from_graph(graph=invalid, instruction="Instruction.")

    assert unused.requests == []
    assert not engine.artifacts.run_path.exists()
