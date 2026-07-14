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

from grace.config import GraceConfig
from grace.evolution.engine import advance_no_update, evolve_graph
from grace.initialization.p2g import initialize_graph
from grace.providers.base import PromptRequest, ProviderCallResult, UsageRecord
from grace.schemas.default import DefaultGraceSchema


class ScriptedProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = deque(responses)
        self.requests: list[PromptRequest] = []

    @property
    def model(self) -> str:
        return "scripted/test-model"

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
            {"id": "N001", "type": "identity", "content": "You are a telecom agent."},
            {"id": "N002", "type": "norm", "content": "Always verify."},
        ],
        "edges": [
            {"source": "N001", "target": "N002", "relation": "supports"},
        ],
    }


def test_scripted_initialize_and_full_evolution() -> None:
    schema = DefaultGraceSchema()
    initializer = ScriptedProvider(
        [
            _initial_graph(),
            {
                "missing_node": [],
                "unfaithful": [],
                "mistyped": [],
                "wrong_relation": [],
                "missing_relation": [],
            },
        ]
    )
    initialized = initialize_graph(
        "You are a telecom agent.\nAlways verify.",
        provider=initializer,
        schema=schema,
        max_rounds=3,
    )
    assert initialized.state.step == 0
    assert initialized.validation_report.schema_valid
    assert initialized.validation_report.fidelity_valid
    assert len(initializer.requests) == 2

    provider = ScriptedProvider(
        [
            {
                "operations": [
                    {
                        "op": "AddNode",
                        "type": "knowledge",
                        "content": "Check account state before troubleshooting.",
                        "attach_edges": [
                            {
                                "source": "NEW",
                                "target": "N002",
                                "relation": "supports",
                            }
                        ],
                    }
                ]
            },
            {"operations": []},
            {"contradictions": [], "redundancies": []},
            {
                "operations": [
                    {
                        "op": "InsertAfter",
                        "anchor": "Always verify.",
                        "new_text": "Check account state before troubleshooting.",
                    }
                ]
            },
        ]
    )
    evolved = evolve_graph(
        initialized.state,
        "Agents skipped account-state checks.",
        provider=provider,
        schema=schema,
        config=GraceConfig(),
    )
    assert evolved.state.step == 1
    assert evolved.state.parent_state_id == initialized.state.state_id
    assert evolved.validation_report.schema_valid
    assert "Check account state" in evolved.state.instruction
    assert len(provider.requests) == 4
    assert len(evolved.usage) == 4


def test_no_update_advances_lineage_without_calls() -> None:
    schema = DefaultGraceSchema()
    provider = ScriptedProvider(
        [
            _initial_graph(),
            {
                "missing_node": [],
                "unfaithful": [],
                "mistyped": [],
                "wrong_relation": [],
                "missing_relation": [],
            },
        ]
    )
    parent = initialize_graph(
        "You are a telecom agent.\nAlways verify.",
        provider=provider,
        schema=schema,
    ).state
    child = advance_no_update(parent, schema=schema)
    assert child.state.step == 1
    assert child.state.parent_state_id == parent.state_id
    assert child.state.graph_hash == parent.graph_hash
    assert child.state.instruction_hash == parent.instruction_hash
    assert child.usage == ()
    assert child.provenance["no_update"] is True
