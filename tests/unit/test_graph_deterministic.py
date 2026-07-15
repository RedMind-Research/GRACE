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

from grace.graph.diff import diff_graphs, enrich_changes
from grace.graph.models import Edge, GraphState, Node
from grace.graph.neighborhood import induced_neighborhood
from grace.graph.validation import validate_graph
from grace.schemas.default import DefaultGraceSchema
from grace.schemas.models import NetworkSchema, ObjectType, RelationType


def _default_nodes() -> list[Node]:
    return [
        Node(id="N001", type="identity", content="You are a support agent."),
        Node(id="N002", type="norm", content="Verify identity."),
        Node(id="N003", type="norm", content="Then inspect the account."),
        Node(id="N004", type="knowledge", content="Accounts have a status."),
    ]


def test_validation_accepts_default_schema_graph() -> None:
    graph = GraphState(
        nodes=_default_nodes(),
        edges=[
            Edge(source="N001", target="N002", relation="supports"),
            Edge(source="N002", target="N003", relation="sequence"),
            Edge(source="N004", target="N003", relation="supports"),
        ],
    )

    report = validate_graph(graph, DefaultGraceSchema())
    assert report.valid
    assert report.passed
    assert report.errors == ()


def test_validation_distinguishes_domain_tokens_from_graph_id_references() -> None:
    graph = GraphState(
        nodes=[
            Node(
                id="N001",
                type="identity",
                content="Wear an N95 mask and document the unbound N999 code.",
            ),
            Node(id="N002", type="norm", content="Follow the safety protocol."),
        ],
        edges=[Edge(source="N001", target="N002", relation="supports")],
    )

    assert validate_graph(graph, DefaultGraceSchema()).valid

    for content in (
        "Follow N002 carefully.",
        "Do not retain provider placeholder N_7.",
        "Do not retain provider placeholder N_new_7.",
    ):
        referenced = graph.model_copy(
            update={
                "nodes": (
                    graph.nodes[0].model_copy(update={"content": content}),
                    graph.nodes[1],
                )
            }
        )
        assert any(
            issue.code == "id_reference"
            for issue in validate_graph(referenced, DefaultGraceSchema()).errors
        )

    custom_ids = GraphState(
        nodes=[
            Node(id="identity-root", type="identity", content="Primary role."),
            Node(id="U1", type="norm", content="Follow identity-root guidance."),
        ]
    )
    custom_report = validate_graph(custom_ids, DefaultGraceSchema())
    assert [
        issue.context["references"]
        for issue in custom_report.errors
        if issue.code == "id_reference"
    ] == [["identity-root"]]


def test_validation_accumulates_boundary_and_schema_errors() -> None:
    graph = {
        "nodes": [
            {"id": "N001", "type": "norm", "content": "Mentions N001."},
            {"id": "N001", "type": "unknown", "content": ""},
        ],
        "edges": [
            {"source": "N001", "target": "missing", "relation": "supports"},
            {"source": "N001", "target": "N001", "relation": "sequence"},
            {"source": "N001", "target": "N001", "relation": "sequence"},
        ],
    }

    report = validate_graph(graph, DefaultGraceSchema())
    codes = {issue.code for issue in report.errors}

    assert not report.valid
    assert {
        "id_reference",
        "duplicate_node_id",
        "object_type",
        "empty_content",
        "dangling_target",
        "self_loop",
        "duplicate_edge",
        "relation_cycle",
    } <= codes


def test_validation_honors_custom_self_loop_and_acyclic_constraints() -> None:
    schema = NetworkSchema(
        id="loop-schema",
        description="Custom loop semantics.",
        object_types=(ObjectType(id="unit", description="A unit."),),
        relation_types=(
            RelationType(
                id="links",
                description="May point to itself.",
                allowed_pairs=(("unit", "unit"),),
                allow_self_loops=True,
            ),
            RelationType(
                id="precedes",
                description="Must be acyclic.",
                allowed_pairs=(("unit", "unit"),),
                acyclic=True,
            ),
        ),
    )
    self_loop = GraphState(
        nodes=[Node(id="U1", type="unit", content="One unit")],
        edges=[Edge(source="U1", target="U1", relation="links")],
    )
    cycle = GraphState(
        nodes=[
            Node(id="U1", type="unit", content="One unit"),
            Node(id="U2", type="unit", content="Another unit"),
        ],
        edges=[
            Edge(source="U1", target="U2", relation="precedes"),
            Edge(source="U2", target="U1", relation="precedes"),
        ],
    )

    assert validate_graph(self_loop, schema).valid
    assert any(issue.code == "relation_cycle" for issue in validate_graph(cycle, schema).errors)


def test_default_neighborhood_excludes_identity_grounding_hub() -> None:
    graph = GraphState(
        nodes=_default_nodes()[:3],
        edges=[
            Edge(source="N001", target="N002", relation="supports"),
            Edge(source="N001", target="N003", relation="supports"),
        ],
    )

    local = induced_neighborhood(graph, {"N002"}, 2, DefaultGraceSchema())
    with_backbone = induced_neighborhood(
        graph,
        {"N002"},
        2,
        DefaultGraceSchema(),
        exclude_grounding_backbone=False,
    )

    assert {node.id for node in local.nodes} == {"N002"}
    assert local.edges == ()
    assert {node.id for node in with_backbone.nodes} == {"N001", "N002", "N003"}


def test_diff_is_stable_and_enrichment_removes_only_grounding_edges() -> None:
    previous = GraphState(nodes=_default_nodes(), edges=[])
    current = GraphState(
        nodes=[
            _default_nodes()[0],
            Node(id="N002", type="norm", content="Verify identity carefully."),
            _default_nodes()[2],
            _default_nodes()[3],
        ],
        edges=[
            Edge(source="N001", target="N002", relation="supports"),
            Edge(source="N004", target="N002", relation="supports"),
        ],
    )

    changes = diff_graphs(previous, current)
    enriched = enrich_changes(changes, previous, current, DefaultGraceSchema())

    assert [change.op for change in changes] == ["ModifyNode", "AddEdge", "AddEdge"]
    assert [change.op for change in enriched] == ["ModifyNode", "AddEdge"]
    kept_edge = enriched[1]
    assert kept_edge.relation == "supports"
    assert kept_edge.source_content == "Accounts have a status."
    assert kept_edge.target_content == "Verify identity carefully."
