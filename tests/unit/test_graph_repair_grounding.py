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

from grace.graph.grounding import apply_grounding_policy
from grace.graph.models import Edge, GraphState, Node
from grace.graph.repair import RepairDropKind, repair_graph
from grace.graph.validation import validate_graph
from grace.schemas.default import DefaultGraceSchema
from grace.schemas.models import NetworkSchema, ObjectType, RelationType


def test_explicit_repair_cleans_structural_errors_with_audit_drops() -> None:
    graph = GraphState(
        nodes=[
            Node(id="N001", type="norm", content="First rule."),
            Node(id="N002", type="norm", content="Second rule."),
            Node(id="N002", type="norm", content="Duplicate rule."),
            Node(id="N003", type="knowledge", content="Fact N999 remains contextual."),
            Node(id="N004", type="knowledge", content="N999"),
        ],
        edges=[
            Edge(source="N001", target="N002", relation="sequence"),
            Edge(source="N002", target="N001", relation="sequence"),
            Edge(source="N001", target="N002", relation="sequence"),
            Edge(source="N001", target="N003", relation="supports"),
            Edge(source="N003", target="N002", relation="supports"),
            Edge(source="N004", target="N002", relation="supports"),
            Edge(source="N001", target="N001", relation="supports"),
            Edge(source="N003", target="missing", relation="supports"),
            Edge(source="N001", target="N002", relation="unknown"),
        ],
    )

    result = repair_graph(graph, DefaultGraceSchema())
    drop_kinds = {drop.kind for drop in result.forced_drops}

    assert validate_graph(result.graph, DefaultGraceSchema()).valid
    assert {node.id for node in result.graph.nodes} == {"N001", "N002", "N003"}
    assert next(node for node in result.graph.nodes if node.id == "N003").content == (
        "Fact remains contextual."
    )
    assert {
        RepairDropKind.DUPLICATE_NODE,
        RepairDropKind.EMPTY_AFTER_ID_STRIP,
        RepairDropKind.DANGLING_EDGE,
        RepairDropKind.UNKNOWN_RELATION,
        RepairDropKind.SELF_LOOP,
        RepairDropKind.TYPE_SIGNATURE_VIOLATION,
        RepairDropKind.DUPLICATE_EDGE,
        RepairDropKind.RELATION_CYCLE_BREAK,
    } <= drop_kinds
    assert graph.metadata == {}
    assert result.graph.metadata["total_nodes"] == 3


def test_repair_preserves_schema_allowed_self_loop() -> None:
    schema = NetworkSchema(
        id="loop",
        description="Self-loop fixture.",
        object_types=(ObjectType(id="unit", description="A unit."),),
        relation_types=(
            RelationType(
                id="links",
                description="A reflexive link.",
                allowed_pairs=(("unit", "unit"),),
                allow_self_loops=True,
            ),
        ),
    )
    graph = GraphState(
        nodes=[Node(id="U1", type="unit", content="Unit")],
        edges=[Edge(source="U1", target="U1", relation="links")],
    )

    result = repair_graph(graph, schema)
    assert result.graph.edges == graph.edges
    assert result.forced_drops == ()


def test_default_grounding_is_deterministic_idempotent_and_schema_valid() -> None:
    graph = GraphState(
        nodes=[
            Node(id="N010", type="identity", content="Secondary role."),
            Node(id="N006", type="norm", content="Second rule."),
            Node(id="N002", type="identity", content="Primary role."),
            Node(id="N005", type="norm", content="First rule."),
        ],
    )
    schema = DefaultGraceSchema()

    result = apply_grounding_policy(graph, schema)
    repeated = apply_grounding_policy(result.graph, schema)

    assert result.top_root_id == "N002"
    assert {(edge.source, edge.target, edge.relation) for edge in result.added_edges} == {
        ("N002", "N005", "supports"),
        ("N002", "N006", "supports"),
        ("N010", "N002", "refines"),
    }
    assert validate_graph(result.graph, schema).valid
    assert repeated.graph == result.graph
    assert repeated.added_edges == ()
    assert graph.edges == ()


def test_grounding_hierarchy_rejects_cycle_and_custom_schema_can_opt_out() -> None:
    default_graph = GraphState(
        nodes=[
            Node(id="N001", type="identity", content="Primary role."),
            Node(id="N002", type="identity", content="Secondary role."),
        ],
        edges=[Edge(source="N001", target="N002", relation="refines")],
    )
    guarded = apply_grounding_policy(default_graph, DefaultGraceSchema())
    assert guarded.added_edges == ()
    assert any(skip.code == "acyclic_relation_cycle" for skip in guarded.skipped)

    custom = NetworkSchema(
        id="plain",
        description="No grounding convention.",
        object_types=(ObjectType(id="unit", description="A unit."),),
        relation_types=(
            RelationType(
                id="links",
                description="A link.",
                allowed_pairs=(("unit", "unit"),),
            ),
        ),
    )
    plain_graph = GraphState(nodes=[Node(id="U1", type="unit", content="Unit")])
    no_op = apply_grounding_policy(plain_graph, custom)
    assert no_op.graph is plain_graph
    assert no_op.added_edges == ()
    assert no_op.skipped[0].code == "no_grounding_policy"
