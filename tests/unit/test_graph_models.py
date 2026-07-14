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

import math

import pytest
from pydantic import ValidationError

from grace.errors import StateIntegrityError
from grace.graph.models import Edge, GraceState, GraphState, Node


def _graph(*, reverse: bool = False) -> GraphState:
    nodes = [
        Node(id="N001", type="identity", content="You are a telecom agent."),
        Node(id="N002", type="norm", content="Verify the subscriber first."),
    ]
    edges = [Edge(source="N001", target="N002", relation="supports")]
    if reverse:
        nodes.reverse()
        edges.reverse()
    return GraphState(nodes=nodes, edges=edges)


def test_graph_hash_is_independent_of_node_and_edge_order() -> None:
    assert _graph().canonical_json() == _graph(reverse=True).canonical_json()
    assert _graph().content_hash() == _graph(reverse=True).content_hash()


def test_state_binds_instruction_graph_schema_and_lineage() -> None:
    parent = GraceState.from_parts(
        graph=_graph(),
        instruction="You are a telecom agent. Verify the subscriber first.",
        schema_id="grace-default-v1",
        schema_hash="schema-hash",
    )
    child = GraceState.from_parts(
        graph=_graph(),
        instruction="You are a careful telecom agent. Verify the subscriber first.",
        schema_id=parent.schema_id,
        schema_hash=parent.schema_hash,
        step=1,
        parent_state_id=parent.state_id,
    )

    assert parent.validate_integrity(schema_id="grace-default-v1")
    assert child.validate_integrity(parent_state=parent)
    assert child.state_id != parent.state_id
    assert child.instruction_hash != parent.instruction_hash


def test_integrity_check_detects_tampering_after_reload() -> None:
    state = GraceState.from_parts(
        graph=_graph(),
        instruction="Original instruction",
        schema_id="grace-default-v1",
        schema_hash="schema-hash",
    )
    tampered = state.model_copy(update={"instruction": "Tampered instruction"})

    with pytest.raises(StateIntegrityError, match="instruction_hash"):
        tampered.validate_integrity()


def test_normal_deserialization_enforces_integrity_automatically() -> None:
    state = GraceState.from_parts(
        graph=_graph(),
        instruction="Original instruction",
        schema_id="grace-default-v1",
        schema_hash="schema-hash",
    )
    payload = state.model_dump(mode="json")
    payload["graph_hash"] = "tampered"

    with pytest.raises(StateIntegrityError, match="graph_hash"):
        GraceState.model_validate(payload)


def test_graph_containers_and_nested_metadata_are_immutable() -> None:
    graph = GraphState(
        nodes=_graph().nodes,
        metadata={"nested": {"items": [1, 2]}},
    )

    assert isinstance(graph.nodes, tuple)
    with pytest.raises(TypeError, match="immutable"):
        graph.metadata["new"] = True
    with pytest.raises(TypeError, match="immutable"):
        graph.metadata["nested"]["items"].append(3)  # type: ignore[union-attr]

    copied = graph.model_copy(deep=True)
    assert copied == graph
    assert copied.content_hash() == graph.content_hash()


def test_graph_metadata_rejects_non_finite_numbers_before_hashing() -> None:
    with pytest.raises(ValidationError, match="NaN or infinite"):
        GraphState(metadata={"invalid": math.inf})


def test_non_initial_state_requires_parent_id() -> None:
    with pytest.raises(StateIntegrityError, match="parent_state_id"):
        GraceState.from_parts(
            graph=_graph(),
            instruction="Instruction",
            schema_id="grace-default-v1",
            schema_hash="schema-hash",
            step=1,
        )
