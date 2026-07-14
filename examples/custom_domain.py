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

"""Apply GRACE to a non-telecom incident-response instruction, without network access."""

from __future__ import annotations

import argparse
import json
import uuid
from collections import deque
from pathlib import Path

from grace import (
    Edge,
    GraceEngine,
    GraphState,
    NetworkSchema,
    Node,
    ObjectType,
    RelationType,
)
from grace.providers import PromptRequest, ProviderCallResult, UsageRecord


class ScriptedProvider:
    """Small deterministic provider used to make this example zero-network."""

    def __init__(self) -> None:
        self._responses = deque(
            [
                {
                    "operations": [
                        {
                            "op": "ModifyNode",
                            "node_id": "N003",
                            "new_content": (
                                "Preserve and record volatile evidence before containment."
                            ),
                        }
                    ]
                },
                {"operations": []},
                {"contradictions": [], "redundancies": []},
                {
                    "operations": [
                        {
                            "op": "ReplaceSpan",
                            "anchor": "Preserve volatile evidence before containment.",
                            "new_text": (
                                "Preserve and record volatile evidence before containment."
                            ),
                        }
                    ]
                },
            ]
        )
        self.calls = 0

    @property
    def model(self) -> str:
        return "scripted/custom-domain"

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        del request
        self.calls += 1
        return ProviderCallResult(
            parsed_content=self._responses.popleft(),
            usage=UsageRecord(
                model=self.model,
                route="scripted",
                input_tokens=10,
                output_tokens=5,
                latency_seconds=0.0,
            ),
        )


def incident_response_schema() -> NetworkSchema:
    """Define only the ontology needed to maintain the instruction."""

    return NetworkSchema(
        id="incident-response",
        version="1",
        description="High-level instruction ontology for an incident-response agent.",
        object_types=(
            ObjectType(id="role", description="The agent's authorized operational role."),
            ObjectType(id="control", description="A constraint that governs response work."),
            ObjectType(id="procedure", description="An operational response procedure."),
        ),
        relation_types=(
            RelationType(
                id="governs",
                description="The source role is accountable for the target control.",
                allowed_pairs=(("role", "control"),),
            ),
            RelationType(
                id="implements",
                description="The source procedure implements the target control.",
                allowed_pairs=(("procedure", "control"),),
            ),
            RelationType(
                id="precedes",
                description="The source procedure must occur before the target procedure.",
                allowed_pairs=(("procedure", "procedure"),),
                acyclic=True,
            ),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("grace_runs"))
    args = parser.parse_args()

    instruction = (
        "You are an incident-response coordinator.\nPreserve volatile evidence before containment."
    )
    graph = GraphState(
        nodes=(
            Node(
                id="N001",
                type="role",
                content="You are an incident-response coordinator.",
            ),
            Node(
                id="N002",
                type="control",
                content="Containment must preserve investigation integrity.",
            ),
            Node(
                id="N003",
                type="procedure",
                content="Preserve volatile evidence before containment.",
            ),
        ),
        edges=(
            Edge(source="N001", target="N002", relation="governs"),
            Edge(source="N003", target="N002", relation="implements"),
        ),
    )
    provider = ScriptedProvider()
    engine = GraceEngine(
        provider=provider,
        schema=incident_response_schema(),
        artifact_dir=args.artifact_dir,
        artifact_mode="audit",
        run_id=f"incident-response-{uuid.uuid4().hex[:12]}",
    )
    initial = engine.initialize_from_graph(graph=graph, instruction=instruction)
    evolved = engine.evolve(
        state=initial.state,
        diagnosis_report=(
            "Observed gap: responders preserved evidence but did not record what was "
            "captured. Require an explicit evidence-recording step before containment."
        ),
    )
    print(
        json.dumps(
            {
                "artifact": str(evolved.artifact_location),
                "provider_calls": provider.calls,
                "schema_id": evolved.state.schema_id,
                "status": evolved.status.value,
                "step": evolved.state.step,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
