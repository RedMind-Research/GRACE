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

from collections import deque
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from grace.graph.models import GraceState
from grace.providers.base import PromptRequest, ProviderCallResult, UsageRecord
from reproduction.tau2_telecom.experiment import (
    EvaluationTiming,
    Method,
    Tau2TelecomEvaluationPlan,
    Tau2TelecomExperimentConfig,
)
from reproduction.tau2_telecom.harness.splits import load_tau2_telecom_task_split
from reproduction.tau2_telecom.live_backend import Tau2TelecomLiveBackend


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "reproduction/tau2_telecom/configs/reproduction.yaml"


class ProviderStub:
    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, request: Any) -> Any:
        raise AssertionError("constructor validation must not make a provider call")


class ScriptedProvider:
    def __init__(self, model: str, responses: list[JsonValue]) -> None:
        self.model = model
        self.responses = deque(responses)
        self.requests: list[PromptRequest] = []

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        self.requests.append(request)
        return ProviderCallResult(
            parsed_content=self.responses.popleft(),
            usage=UsageRecord(
                model=self.model,
                input_tokens=1,
                output_tokens=1,
                latency_seconds=0.01,
            ),
        )


def _initial_graph() -> dict[str, JsonValue]:
    return {
        "nodes": [
            {"id": "N001", "type": "identity", "content": "You are an agent."},
            {"id": "N002", "type": "norm", "content": "Verify requests."},
        ],
        "edges": [
            {"source": "N001", "target": "N002", "relation": "supports"},
        ],
    }


def _fidelity_pass() -> dict[str, JsonValue]:
    return {
        "missing_node": [],
        "unfaithful": [],
        "mistyped": [],
        "wrong_relation": [],
        "missing_relation": [],
    }


def test_release_config_is_typed_and_uses_one_public_seed() -> None:
    config = Tau2TelecomExperimentConfig.load(CONFIG)

    assert config.execution.seed == 1024
    assert config.execution.phase_schedule == tuple("AABBAABBAA")
    assert config.method_calls.max_tokens == 65536
    assert config.episode_runtime.max_tokens == 8192
    assert config.benchmark.task_split.is_absolute()
    assert len(config.config_hash) == 64


def test_live_backend_binds_every_method_role_to_the_provider(tmp_path: Path) -> None:
    config = Tau2TelecomExperimentConfig.load(CONFIG)
    split = load_tau2_telecom_task_split(config.benchmark.task_split)

    Tau2TelecomLiveBackend(
        config=config,
        split=split,
        provider=ProviderStub(config.models.task_agent),
        artifact_root=tmp_path,
        run_id="valid-models",
    )
    with pytest.raises(ValueError, match="must use the configured provider"):
        Tau2TelecomLiveBackend(
            config=config,
            split=split,
            provider=ProviderStub("gemini/other"),
            artifact_root=tmp_path,
            run_id="invalid-models",
        )


@pytest.mark.parametrize(
    ("method", "expected_evolution_stages"),
    [
        (
            "grace",
            [
                "operation_planning",
                "relation_maintenance",
                "structural_analysis_1",
                "patch_reconstruction",
            ],
        ),
        (
            "grace-no-sa",
            ["operation_planning", "relation_maintenance", "patch_reconstruction"],
        ),
    ],
)
def test_grace_no_sa_changes_only_the_structural_analysis_call(
    tmp_path: Path,
    method: Method,
    expected_evolution_stages: list[str],
) -> None:
    config = Tau2TelecomExperimentConfig.load(CONFIG)
    split = load_tau2_telecom_task_split(config.benchmark.task_split)
    evolution_responses: list[JsonValue] = [
        {"operations": []},
        {"operations": []},
    ]
    if method == "grace":
        evolution_responses.append({"contradictions": [], "redundancies": []})
    evolution_responses.append({"operations": []})
    provider = ScriptedProvider(
        config.models.task_agent,
        [_initial_graph(), _fidelity_pass(), *evolution_responses],
    )
    backend = Tau2TelecomLiveBackend(
        config=config,
        split=split,
        provider=provider,
        artifact_root=tmp_path,
        run_id=f"test-{method}",
    )

    checkpoint = backend.initialize(
        method=method,
        instruction="You are an agent.\nVerify requests.",
    )
    evolved = backend.evolve(
        method=method,
        checkpoint=checkpoint,
        update_index=1,
        diagnosis={"status": "complete", "report": "Maintain current behavior."},
    )

    assert evolved.method == method
    assert [request.stage for request in provider.requests[:2]] == [
        "p2g_initial",
        "p2g_fidelity_1",
    ]
    assert [request.stage for request in provider.requests[2:]] == expected_evolution_stages
    assert {request.max_tokens for request in provider.requests} == {config.method_calls.max_tokens}
    assert not provider.responses


@pytest.mark.parametrize("method", ["grace", "grace-no-sa"])
def test_no_update_advances_inner_grace_lineage_without_provider_call(
    tmp_path: Path,
    method: Method,
) -> None:
    config = Tau2TelecomExperimentConfig.load(CONFIG)
    split = load_tau2_telecom_task_split(config.benchmark.task_split)
    provider = ScriptedProvider(
        config.models.task_agent,
        [_initial_graph(), _fidelity_pass()],
    )
    backend = Tau2TelecomLiveBackend(
        config=config,
        split=split,
        provider=provider,
        artifact_root=tmp_path,
        run_id=f"no-update-{method}",
    )
    checkpoint = backend.initialize(
        method=method,
        instruction="You are an agent.\nVerify requests.",
    )
    parent = GraceState.model_validate(checkpoint.method_state["grace_state"])

    advanced = backend.evolve(
        method=method,
        checkpoint=checkpoint,
        update_index=1,
        diagnosis={
            "status": "no_update",
            "report": "No failed trajectories were observed in this batch.",
        },
    )
    child = GraceState.model_validate(advanced.method_state["grace_state"])

    assert child.step == parent.step + 1
    assert child.parent_state_id == parent.state_id
    assert child.state_id != parent.state_id
    assert child.graph_hash == parent.graph_hash
    assert child.instruction_hash == parent.instruction_hash
    assert [request.stage for request in provider.requests] == [
        "p2g_initial",
        "p2g_fidelity_1",
    ]
    assert not provider.responses


@pytest.mark.parametrize("timing", ["in-loop", "offline"])
def test_evaluation_plan_combines_timing_and_selection(timing: str) -> None:
    all_plan = Tau2TelecomEvaluationPlan.parse(f"{timing}:all")
    selected = Tau2TelecomEvaluationPlan.parse(f"{timing}:0,6,8,10")

    assert all_plan.checkpoint_indices == tuple(range(11))
    assert selected.checkpoint_indices == (0, 6, 8, 10)
    assert selected.timing == EvaluationTiming(timing)


@pytest.mark.parametrize(
    "value",
    ["deferred:all", "offline:", "in-loop:1,1", "in-loop:2,1", "offline:11"],
)
def test_evaluation_plan_rejects_ambiguous_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        Tau2TelecomEvaluationPlan.parse(value)
