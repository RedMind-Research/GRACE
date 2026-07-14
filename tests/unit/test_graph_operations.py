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

from grace.graph.models import Edge, GraphState, Node
from grace.graph.operations import AddEdge, AddNode, EdgeAttachment, Merge, assemble
from grace.schemas.default import DefaultGraceSchema
from grace.schemas.models import NetworkSchema, ObjectType, RelationType


def test_assemble_adds_stable_node_and_attachment_without_mutating_input() -> None:
    graph = GraphState(
        nodes=[
            Node(id="N001", type="identity", content="You are an agent."),
            Node(id="N007", type="norm", content="Verify identity."),
        ],
        edges=[],
    )
    original_hash = graph.content_hash()

    result = assemble(
        graph,
        [
            AddNode(
                type="knowledge",
                content="  Accounts have a status.  ",
                attach_edges=(EdgeAttachment(source="NEW", target="N007", relation="supports"),),
            )
        ],
        DefaultGraceSchema(),
    )

    assert graph.content_hash() == original_hash
    assert {node.id for node in result.graph.nodes} == {"N001", "N007", "N008"}
    assert result.graph.nodes[-1].content == "Accounts have a status."
    assert result.graph.edges == (Edge(source="N008", target="N007", relation="supports"),)
    assert result.touched_node_ids == ("N007", "N008")
    assert result.applied_count == 1
    assert result.rejected_count == 0


def test_assemble_rejects_schema_violations_with_stable_codes() -> None:
    graph = GraphState(
        nodes=[
            Node(id="N001", type="norm", content="First."),
            Node(id="N002", type="norm", content="Second."),
            Node(id="N003", type="knowledge", content="A fact."),
        ],
        edges=[Edge(source="N001", target="N002", relation="sequence")],
    )

    result = assemble(
        graph,
        [
            AddEdge(source="N002", target="N001", relation="sequence"),
            AddEdge(source="N001", target="N003", relation="supports"),
            AddEdge(source="N001", target="N001", relation="supports"),
            AddEdge(source="N001", target="missing", relation="supports"),
            AddEdge(source="N001", target="N002", relation="sequence"),
            {"op": "UnknownOperation"},
        ],
        DefaultGraceSchema(),
    )

    assert result.graph == graph.model_copy(
        update={"metadata": {"total_nodes": 3, "total_edges": 1}}
    )
    assert [rejection.code for rejection in result.rejected_operations] == [
        "acyclic_relation_cycle",
        "type_signature_violation",
        "self_loop",
        "dangling_endpoint",
        "duplicate_edge",
        "invalid_operation",
    ]
    assert result.touched_node_ids == ()


def test_merge_that_would_contract_an_acyclic_relation_into_cycle_is_atomic() -> None:
    graph = GraphState(
        nodes=[
            Node(id="N001", type="norm", content="Representative."),
            Node(id="N002", type="norm", content="Absorbed."),
            Node(id="N003", type="norm", content="Third."),
        ],
        edges=[
            Edge(source="N002", target="N003", relation="refines"),
            Edge(source="N003", target="N001", relation="refines"),
        ],
    )

    result = assemble(
        graph,
        [Merge(u_id="N001", v_id="N002", new_content="Merged.")],
        DefaultGraceSchema(),
    )

    assert result.graph.nodes == graph.nodes
    assert result.graph.edges == graph.edges
    assert result.applied_count == 0
    assert result.rejected_operations[0].code == "acyclic_relation_cycle"


def test_custom_schema_can_allow_self_loops_during_assembly() -> None:
    schema = NetworkSchema(
        id="custom",
        description="Custom self-loop fixture.",
        object_types=(ObjectType(id="unit", description="One unit."),),
        relation_types=(
            RelationType(
                id="links",
                description="A link.",
                allowed_pairs=(("unit", "unit"),),
                allow_self_loops=True,
            ),
        ),
    )
    graph = GraphState(nodes=[Node(id="U1", type="unit", content="Unit")])

    result = assemble(
        graph,
        [AddEdge(source="U1", target="U1", relation="links")],
        schema,
    )

    assert result.rejected_count == 0
    assert result.graph.edges == (Edge(source="U1", target="U1", relation="links"),)
