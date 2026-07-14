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

from collections import deque

import pytest
from jsonschema import Draft202012Validator

from grace.artifacts.models import ResultStatus, ValidationStatus
from grace.config import GraceConfig
from grace.errors import StateIntegrityError
from grace.evolution.engine import evolve_graph
from grace.graph.models import GraceState, GraphState
from grace.providers.base import PromptRequest, ProviderCallResult, UsageRecord
from grace.schemas.default import DefaultGraceSchema


class ScriptedProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.requests: list[PromptRequest] = []

    @property
    def model(self) -> str:
        return "scripted/test"

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


def _state() -> GraceState:
    schema = DefaultGraceSchema()
    graph = GraphState.model_validate(
        {
            "nodes": [
                {"id": "N001", "type": "identity", "content": "You are an agent."},
                {"id": "N002", "type": "norm", "content": "Verify requests."},
                {
                    "id": "N003",
                    "type": "norm",
                    "content": "Check requests carefully.",
                },
            ],
            "edges": [
                {"source": "N001", "target": "N002", "relation": "supports"},
                {"source": "N001", "target": "N003", "relation": "supports"},
            ],
        }
    )
    return GraceState.from_parts(
        graph=graph,
        instruction="You are an agent.\nVerify requests.\nCheck requests carefully.",
        schema_id=schema.id,
        schema_hash=schema.schema_hash,
    )


def test_no_sa_skips_only_model_assisted_structural_loop() -> None:
    provider = ScriptedProvider(
        [
            {"operations": []},
            {"operations": []},
            {"operations": []},
        ]
    )
    result = evolve_graph(
        _state(),
        "No policy update is needed.",
        provider=provider,
        schema=DefaultGraceSchema(),
        config=GraceConfig(structural_analysis=False),
    )
    assert [request.stage for request in provider.requests] == [
        "operation_planning",
        "relation_maintenance",
        "patch_reconstruction",
    ]
    assert all(request.logical_call_id is not None for request in provider.requests)
    assert len({request.logical_call_id for request in provider.requests}) == len(provider.requests)
    assert {request.max_tokens for request in provider.requests} == {65536}
    assert result.validation_report.schema_valid
    assert result.state.step == 1


def test_evolution_forwards_max_output_tokens_to_every_provider_call() -> None:
    provider = ScriptedProvider(
        [
            {"operations": []},
            {"operations": []},
            {"operations": []},
        ]
    )

    evolve_graph(
        _state(),
        "No policy update is needed.",
        provider=provider,
        schema=DefaultGraceSchema(),
        config=GraceConfig(structural_analysis=False, max_output_tokens=4096),
    )

    assert [request.max_tokens for request in provider.requests] == [4096, 4096, 4096]


def test_operation_planning_cannot_smuggle_merge() -> None:
    provider = ScriptedProvider(
        [
            {
                "operations": [
                    {
                        "op": "Merge",
                        "u_id": "N002",
                        "v_id": "N003",
                        "new_content": "Verify requests carefully.",
                    }
                ]
            },
            {"operations": []},
            {"contradictions": [], "redundancies": []},
            {"operations": []},
        ]
    )
    parent = _state()
    result = evolve_graph(
        parent,
        "The rules overlap.",
        provider=provider,
        schema=DefaultGraceSchema(),
        config=GraceConfig(),
    )
    assert result.state.graph.nodes == parent.graph.nodes
    assert result.status == ResultStatus.COMPLETED_WITH_WARNINGS
    assert result.validation_report.rejected_operations[0]["code"] == (
        "operation_not_allowed_in_stage"
    )


def test_operation_transport_schema_requires_stage_specific_op_discriminator() -> None:
    provider = ScriptedProvider(
        [
            {"operations": []},
            {"operations": []},
            {"operations": []},
        ]
    )
    evolve_graph(
        _state(),
        "No structural update is needed.",
        provider=provider,
        schema=DefaultGraceSchema(),
        config=GraceConfig(structural_analysis=False),
    )

    examples = [
        {
            "AddNode": {"op": "AddNode", "type": "norm", "content": "Rule."},
            "ModifyNode": {
                "op": "ModifyNode",
                "node_id": "N001",
                "new_content": "Updated.",
            },
            "RemoveNode": {"op": "RemoveNode", "node_id": "N001"},
        },
        {
            "AddEdge": {
                "op": "AddEdge",
                "source": "N001",
                "target": "N002",
                "relation": "supports",
            },
            "RemoveEdge": {
                "op": "RemoveEdge",
                "source": "N001",
                "target": "N002",
                "relation": "supports",
            },
        },
        {
            "ReplaceSpan": {
                "op": "ReplaceSpan",
                "anchor": "old",
                "new_text": "new",
            },
            "InsertAfter": {
                "op": "InsertAfter",
                "anchor": "old",
                "new_text": "new",
            },
            "DeleteSpan": {"op": "DeleteSpan", "anchor": "old"},
        },
    ]
    for request, stage_examples in zip(provider.requests, examples, strict=True):
        assert request.response_schema is not None
        schema = request.response_schema
        variants = schema["properties"]["operations"]["items"]["anyOf"]
        allowed = {variant["properties"]["op"]["enum"][0] for variant in variants}
        assert allowed == set(stage_examples)
        validator = Draft202012Validator(schema)
        for example in stage_examples.values():
            assert not list(validator.iter_errors({"operations": [example]}))
        example_name = next(iter(stage_examples))
        assert list(validator.iter_errors({"operations": [{"operation": example_name}]}))
        assert list(validator.iter_errors({"operations": [{"op": example_name}]}))


def test_sa_detects_repairs_and_rechecks_expanding_radius() -> None:
    provider = ScriptedProvider(
        [
            {
                "operations": [
                    {
                        "op": "ModifyNode",
                        "node_id": "N002",
                        "new_content": "Verify requests before acting.",
                    }
                ]
            },
            {"operations": []},
            {
                "contradictions": [],
                "redundancies": [{"u_id": "N002", "v_id": "N003", "reason": "overlap"}],
            },
            {
                "operations": [
                    {
                        "op": "Merge",
                        "u_id": "N002",
                        "v_id": "N003",
                        "new_content": "Verify requests carefully before acting.",
                    }
                ]
            },
            {"contradictions": [], "redundancies": []},
            {
                "operations": [
                    {
                        "op": "ReplaceSpan",
                        "anchor": "Verify requests.\nCheck requests carefully.",
                        "new_text": "Verify requests carefully before acting.",
                    }
                ]
            },
        ]
    )
    result = evolve_graph(
        _state(),
        "Consolidate overlapping verification rules.",
        provider=provider,
        schema=DefaultGraceSchema(),
        config=GraceConfig(),
    )
    assert result.validation_report.status == ValidationStatus.PASSED
    assert result.validation_report.rounds == 2
    assert len(result.state.graph.nodes) == 2
    assert "carefully before acting" in result.state.instruction
    assert [request.stage for request in provider.requests][2:5] == [
        "structural_analysis_1",
        "structural_repair_1",
        "structural_analysis_2",
    ]


def test_tampered_state_fails_before_first_provider_call() -> None:
    parent = _state()
    tampered = parent.model_copy(update={"instruction": parent.instruction + " tampered"})
    provider = ScriptedProvider([])
    with pytest.raises(StateIntegrityError, match="instruction_hash"):
        evolve_graph(
            tampered,
            "Diagnosis.",
            provider=provider,
            schema=DefaultGraceSchema(),
            config=GraceConfig(),
        )
    assert provider.requests == []
