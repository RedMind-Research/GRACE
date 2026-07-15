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

from grace.artifacts.models import ResultStatus, ValidationStatus
from grace.errors import ProviderError, SchemaValidationError
from grace.initialization.p2g import initialize_graph
from grace.providers.base import PromptRequest, ProviderCallResult, UsageRecord
from grace.schemas.default import DefaultGraceSchema
from grace.schemas.models import NetworkSchema, ObjectType, RelationType


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


def _graph() -> dict:
    return {
        "nodes": [
            {"id": "N001", "type": "identity", "content": "You are an agent."},
            {"id": "N002", "type": "norm", "content": "Verify requests."},
        ],
        "edges": [
            {"source": "N001", "target": "N002", "relation": "supports"},
        ],
    }


def _clean() -> dict:
    return {
        "missing_node": [],
        "unfaithful": [],
        "mistyped": [],
        "wrong_relation": [],
        "missing_relation": [],
    }


def test_p2g_repairs_then_converges() -> None:
    provider = ScriptedProvider(
        [
            _graph(),
            {**_clean(), "missing_node": ["Escalate blocked requests."]},
            {
                "operations": [
                    {
                        "op": "AddNode",
                        "type": "norm",
                        "content": "Escalate blocked requests.",
                    }
                ]
            },
            _clean(),
        ]
    )
    result = initialize_graph(
        "You are an agent. Verify requests. Escalate blocked requests.",
        provider=provider,
        schema=DefaultGraceSchema(),
    )
    assert result.validation_report.status == ValidationStatus.CONVERGED
    assert result.validation_report.rounds == 2
    assert len(result.state.graph.nodes) == 3
    assert len(provider.requests) == 4
    assert all(request.logical_call_id is not None for request in provider.requests)
    assert len({request.logical_call_id for request in provider.requests}) == 4
    assert {request.max_tokens for request in provider.requests} == {65536}


def test_p2g_forwards_max_output_tokens_to_every_provider_call() -> None:
    provider = ScriptedProvider([_graph(), _clean()])

    initialize_graph(
        "You are an agent. Verify requests.",
        provider=provider,
        schema=DefaultGraceSchema(),
        max_output_tokens=4096,
    )

    assert [request.max_tokens for request in provider.requests] == [4096, 4096]


def test_p2g_cap_is_honest_warning_not_false_convergence() -> None:
    provider = ScriptedProvider([_graph(), {**_clean(), "missing_node": ["Still missing."]}])
    result = initialize_graph(
        "You are an agent. Verify requests. Still missing.",
        provider=provider,
        schema=DefaultGraceSchema(),
        max_rounds=1,
    )
    assert result.validation_report.status == ValidationStatus.CAP_REACHED
    assert result.validation_report.fidelity_valid is False
    assert result.status == ResultStatus.COMPLETED_WITH_WARNINGS


def test_p2g_closeout_surfaces_node_id_reference_rewrite() -> None:
    graph = _graph()
    graph["nodes"][0]["content"] = "Follow N002 carefully."
    provider = ScriptedProvider([graph, {**_clean(), "missing_node": ["Still missing."]}])

    result = initialize_graph(
        "Follow carefully. Verify requests. Still missing.",
        provider=provider,
        schema=DefaultGraceSchema(),
        max_rounds=1,
    )

    assert result.state.graph.nodes[0].content == "Follow carefully."
    assert result.validation_report.forced_drops[0]["kind"] == "id_reference_rewrite"
    assert result.validation_report.forced_drops[0]["cleaned_content"] == ("Follow carefully.")
    assert "deterministic closeout changed or dropped 1 item(s)" in result.warnings


def test_p2g_rejects_non_object_provider_response() -> None:
    provider = ScriptedProvider([[{"not": "a graph object"}]])
    with pytest.raises(ProviderError, match="JSON object"):
        initialize_graph(
            "Instruction.",
            provider=provider,
            schema=DefaultGraceSchema(),
        )


def test_p2g_final_invalid_schema_is_fatal() -> None:
    provider = ScriptedProvider(
        [
            {
                "nodes": [{"id": "N001", "type": "unknown", "content": "Content."}],
                "edges": [],
            },
            _clean(),
        ]
    )
    with pytest.raises(SchemaValidationError, match="final graph is invalid"):
        initialize_graph(
            "Content.",
            provider=provider,
            schema=DefaultGraceSchema(),
            max_rounds=1,
        )


def test_custom_ontology_is_rendered_without_default_type_assumptions() -> None:
    schema = NetworkSchema(
        id="legal",
        description="Legal review.",
        object_types=(
            ObjectType(id="role", description="Reviewer mandate."),
            ObjectType(id="rule", description="Review requirement."),
        ),
        relation_types=(
            RelationType(
                id="governs",
                description="Role governs rule.",
                allowed_pairs=(("role", "rule"),),
            ),
        ),
    )
    provider = ScriptedProvider(
        [
            {
                "nodes": [
                    {"id": "N001", "type": "role", "content": "You review contracts."},
                    {"id": "N002", "type": "rule", "content": "Check signatures."},
                ],
                "edges": [
                    {"source": "N001", "target": "N002", "relation": "governs"},
                ],
            },
            _clean(),
        ]
    )
    result = initialize_graph(
        "You review contracts. Check signatures.",
        provider=provider,
        schema=schema,
    )
    assert result.validation_report.schema_valid
    first_system = provider.requests[0].system_prompt
    assert "- role: Reviewer mandate." in first_system
    assert "- governs: Role governs rule." in first_system
    assert "identity|norm|knowledge" not in first_system
