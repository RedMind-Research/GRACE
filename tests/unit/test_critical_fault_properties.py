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

import copy
import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from grace.artifacts.models import ReconstructionStatus
from grace.errors import ConfigurationError, StateIntegrityError
from grace.evolution.reconstruction import (
    PatchApplicationResult,
    ReplaceSpan,
    apply_patch_operations,
)
from grace.graph.diff import diff_graphs, diff_log, enrich_change_log, enrich_changes
from grace.graph.models import Edge, GraceState, GraphState, Node
from grace.graph.operations import (
    AddEdge,
    AddNode,
    EdgeAttachment,
    Merge,
    ModifyNode,
    RemoveEdge,
    RemoveNode,
    assemble,
)
from grace.graph.repair import ForcedDrop, RepairDropKind, repair_graph, repair_schema
from grace.graph.validation import (
    allowed_pairs,
    relation_has_cycle,
    validate_graph,
    validate_schema,
)
from grace.schemas.loader import load_schema
from grace.schemas.models import GroundingPolicy, NetworkSchema, ObjectType, RelationType


def _unit_schema(*, acyclic: bool = False) -> NetworkSchema:
    return NetworkSchema(
        id="unit-schema",
        description="Schema used to exercise deterministic graph boundaries.",
        object_types=(
            ObjectType(id="unit", description="A unit."),
            ObjectType(id="other", description="A different unit."),
        ),
        relation_types=(
            RelationType(
                id="links",
                description="Links units.",
                allowed_pairs=(("unit", "unit"),),
                acyclic=acyclic,
            ),
        ),
    )


def _unit_graph() -> GraphState:
    return GraphState(
        nodes=(
            Node(id="N001", type="unit", content="First unit."),
            Node(id="N002", type="unit", content="Second unit."),
            Node(id="N003", type="unit", content="Third unit."),
        ),
        edges=(Edge(source="N001", target="N002", relation="links"),),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_assembly_audits_non_json_and_non_finite_malformed_inputs() -> None:
    schema = _unit_schema()
    graph = _unit_graph()
    malformed = (
        ObjectType(id="unit", description="A Pydantic object."),
        float("nan"),
        1.25,
        object(),
    )

    for raw in malformed:
        result = assemble(graph, [raw], schema)  # type: ignore[list-item]
        assert result.applied_count == 0
        assert result.rejected_operations[0].code == "invalid_operation"
        json.dumps(result.rejected_operations[0].operation, allow_nan=False)


def test_assembly_rejects_node_faults_and_removes_incident_edges() -> None:
    result = assemble(
        _unit_graph(),
        [
            AddNode(type="missing", content="Content"),
            AddNode(type="unit", content="  "),
            ModifyNode(node_id="missing", new_content="Content"),
            ModifyNode(node_id="N001", new_content="  "),
            RemoveNode(node_id="missing"),
            RemoveNode(node_id="N002"),
        ],
        _unit_schema(),
    )

    assert [item.code for item in result.rejected_operations] == [
        "invalid_object_type",
        "empty_content",
        "unknown_node",
        "empty_content",
        "unknown_node",
    ]
    assert result.applied_count == 1
    assert result.graph.edges == ()
    assert {node.id for node in result.graph.nodes} == {"N001", "N003"}
    assert result.applied_operations[0].effect["removed_edge_count"] == 1


def test_assembly_exercises_edge_faults_success_and_removal() -> None:
    result = assemble(
        _unit_graph(),
        [
            AddEdge(source="N001", target="N003", relation="missing"),
            AddEdge(source="N003", target="N003", relation="links"),
            AddEdge(source="N001", target="N003", relation="links"),
            RemoveEdge(source="missing", target="N002", relation="links"),
            RemoveEdge(source="N001", target="N002", relation="missing"),
            RemoveEdge(source="N002", target="N003", relation="links"),
            RemoveEdge(source="N001", target="N002", relation="links"),
        ],
        _unit_schema(acyclic=True),
    )

    assert [item.code for item in result.rejected_operations] == [
        "invalid_relation",
        "self_loop",
        "dangling_endpoint",
        "invalid_relation",
        "edge_not_found",
    ]
    assert result.applied_count == 2
    assert result.graph.edges == (Edge(source="N001", target="N003", relation="links"),)


def test_add_node_attachment_records_each_nested_outcome() -> None:
    result = assemble(
        _unit_graph(),
        [
            AddNode(
                type="unit",
                content="Fourth unit.",
                attach_edges=(
                    EdgeAttachment(source="NEW", target="N003", relation="links"),
                    EdgeAttachment(source="NEW", target="N003", relation="missing"),
                ),
            )
        ],
        _unit_schema(),
    )

    assert result.applied_operations[0].effect == {
        "generated_node_id": "N004",
        "attached_edge_count": 1,
        "rejected_attachment_count": 1,
    }
    assert result.rejected_operations[0].path.endswith("attach_edges[1]")
    assert result.rejected_operations[0].code == "invalid_relation"


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        (Merge(u_id="missing", v_id="N002", new_content="Merged"), "unknown_node"),
        (Merge(u_id="N001", v_id="N001", new_content="Merged"), "same_merge_node"),
        (Merge(u_id="N001", v_id="O1", new_content="Merged"), "merge_type_mismatch"),
        (Merge(u_id="N001", v_id="N002", new_content="  "), "empty_content"),
    ],
)
def test_merge_rejects_invalid_preconditions(operation: Merge, code: str) -> None:
    graph = GraphState(
        nodes=(*_unit_graph().nodes, Node(id="O1", type="other", content="Other.")),
        edges=_unit_graph().edges,
    )

    result = assemble(graph, [operation], _unit_schema())

    assert result.applied_count == 0
    assert result.rejected_operations[0].code == code
    assert result.graph.nodes == graph.nodes
    assert result.graph.edges == graph.edges


def test_merge_collapses_self_loops_and_duplicate_reroutes_atomically() -> None:
    graph = GraphState(
        nodes=_unit_graph().nodes,
        edges=(
            Edge(source="N001", target="N002", relation="links"),
            Edge(source="N001", target="N003", relation="links"),
            Edge(source="N002", target="N003", relation="links"),
        ),
    )

    result = assemble(
        graph,
        [Merge(u_id="N001", v_id="N002", new_content="Merged unit.")],
        _unit_schema(),
    )

    assert result.rejected_count == 0
    assert result.graph.edges == (Edge(source="N001", target="N003", relation="links"),)
    assert result.applied_operations[0].effect["collapsed_self_loop_count"] == 1
    assert result.applied_operations[0].effect["collapsed_duplicate_count"] == 1


def test_merge_rejects_invalid_existing_edge_reroute_without_partial_mutation() -> None:
    graph = GraphState(
        nodes=_unit_graph().nodes,
        edges=(Edge(source="N002", target="N003", relation="undeclared"),),
    )

    result = assemble(
        graph,
        [Merge(u_id="N001", v_id="N002", new_content="Merged unit.")],
        _unit_schema(),
    )

    assert result.applied_count == 0
    assert result.rejected_operations[0].code == "merge_reroute_violation"
    assert result.graph.nodes == graph.nodes
    assert result.graph.edges == graph.edges


def test_validation_accumulates_malformed_container_and_field_faults() -> None:
    schema = _unit_schema()

    wrong_type = validate_graph(42, schema)  # type: ignore[arg-type]
    wrong_containers = validate_graph({"nodes": "not-a-sequence", "edges": None}, schema)
    malformed = validate_graph(
        {
            "nodes": [
                1,
                {"id": " ", "type": "unit", "content": "Valid content."},
                {"type": None},
                {"id": "O1", "type": "other", "content": "Other content."},
            ],
            "edges": [
                1,
                {},
                {"source": "O1", "target": " ", "relation": "links"},
                {"source": "O1", "target": "O1", "relation": "unknown"},
            ],
        },
        schema,
    )

    assert {issue.code for issue in wrong_type.errors} == {"graph_type", "required_data"}
    assert {issue.field for issue in wrong_containers.errors} == {"nodes", "edges"}
    malformed_codes = {issue.code for issue in malformed.errors}
    assert {
        "node_type",
        "required_field",
        "empty_id",
        "edge_type",
        "edge_required_field",
        "edge_relation",
        "dangling_source",
        "dangling_target",
        "type_signature",
    } <= malformed_codes
    assert validate_schema(_unit_graph(), schema).valid
    assert allowed_pairs(schema, "unknown") == frozenset()


def test_cycle_detection_ignores_incomplete_and_unrelated_edges() -> None:
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edges": [
            {"source": None, "target": "A", "relation": "links"},
            {"source": "A", "target": None, "relation": "links"},
            {"source": "missing", "target": "A", "relation": "links"},
            {"source": "A", "target": "B", "relation": "other"},
            {"source": "A", "target": "B", "relation": "links"},
            {"source": "C", "target": "B", "relation": "links"},
        ],
    }

    assert not relation_has_cycle(graph, "links")


def test_schema_loader_fails_closed_for_io_yaml_and_document_shape(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot load"):
        load_schema(tmp_path / "missing.yaml")

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("value: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot load"):
        load_schema(invalid_yaml)

    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("just one scalar", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="one YAML mapping"):
        load_schema(scalar_yaml)


def test_schema_models_reject_ambiguous_ids_pairs_and_grounding_references() -> None:
    with pytest.raises(ValidationError, match="empty type id"):
        RelationType(
            id="bad",
            description="Bad empty pair.",
            allowed_pairs=(("", "unit"),),
        )

    unit = ObjectType(id="unit", description="A unit.")
    other = ObjectType(id="other", description="Another unit.")
    links = RelationType(
        id="links",
        description="A link.",
        allowed_pairs=(("unit", "unit"),),
    )
    with pytest.raises(ValidationError, match="object type ids must be unique"):
        NetworkSchema(
            id="duplicate-objects",
            description="Invalid.",
            object_types=(unit, unit),
            relation_types=(links,),
        )
    with pytest.raises(ValidationError, match="relation type ids must be unique"):
        NetworkSchema(
            id="duplicate-relations",
            description="Invalid.",
            object_types=(unit,),
            relation_types=(links, links),
        )

    def schema_with(
        policy: GroundingPolicy,
        *,
        support_pairs: tuple[tuple[str, str], ...] = (("unit", "other"),),
        hierarchy_pairs: tuple[tuple[str, str], ...] = (("unit", "unit"),),
    ) -> NetworkSchema:
        return NetworkSchema(
            id="grounding",
            description="Grounding validation fixture.",
            object_types=(unit, other),
            relation_types=(
                RelationType(
                    id="supports",
                    description="Support.",
                    allowed_pairs=support_pairs,
                ),
                RelationType(
                    id="hierarchy",
                    description="Hierarchy.",
                    allowed_pairs=hierarchy_pairs,
                ),
            ),
            grounding_policy=policy,
        )

    valid_policy = GroundingPolicy(
        root_object_type="unit",
        grounded_object_type="other",
        support_relation="supports",
        hierarchy_relation="hierarchy",
    )
    invalid_cases = (
        (
            valid_policy.model_copy(update={"root_object_type": "missing"}),
            {},
            "root_object_type",
        ),
        (
            valid_policy.model_copy(update={"grounded_object_type": "missing"}),
            {},
            "grounded_object_type",
        ),
        (
            valid_policy.model_copy(
                update={"exclude_incident_object_types_from_neighborhood": ("missing",)}
            ),
            {},
            "excludes unknown",
        ),
        (
            valid_policy.model_copy(update={"support_relation": "missing"}),
            {},
            "support_relation is unknown",
        ),
        (
            valid_policy.model_copy(update={"hierarchy_relation": "missing"}),
            {},
            "hierarchy_relation is unknown",
        ),
        (valid_policy, {"support_pairs": (("other", "unit"),)}, "support relation"),
        (valid_policy, {"hierarchy_pairs": (("other", "other"),)}, "hierarchy relation"),
    )
    for policy, kwargs, message in invalid_cases:
        with pytest.raises(ValidationError, match=message):
            schema_with(policy, **kwargs)

    valid = schema_with(valid_policy)
    assert valid.object_type_map == {"unit": unit, "other": other}


def test_state_helpers_and_integrity_guards_cover_all_lineage_bindings() -> None:
    graph = _unit_graph()
    parent = GraceState.from_parts(
        graph=graph.model_dump(mode="python"),
        instruction="Parent instruction.",
        schema_id="unit-schema",
        schema_hash="schema-hash",
    )
    child = GraceState.from_parts(
        graph=graph,
        instruction="Child instruction.",
        schema_id=parent.schema_id,
        schema_hash=parent.schema_hash,
        step=1,
        parent_state_id=parent.state_id,
    )

    assert graph.canonical_hash() == graph.content_hash()
    assert copy.copy(graph.metadata) is graph.metadata
    with pytest.raises(TypeError, match="immutable"):
        graph.metadata["late mutation"] = True
    nested = GraphState(metadata={"values": [1, 2]}).metadata["values"]
    assert copy.copy(nested) is nested
    assert copy.deepcopy(nested) is nested

    empty_fields = parent.model_copy(
        update={"instruction": " ", "schema_id": " ", "schema_hash": " "}
    )
    with pytest.raises(StateIntegrityError, match="instruction is empty.*schema_id.*schema_hash"):
        empty_fields.validate_integrity()

    with pytest.raises(StateIntegrityError, match="schema_id mismatch.*schema_hash"):
        parent.validate_integrity(schema_id="other", schema_hash="other-hash")

    impossible_root = parent.model_copy(update={"parent_state_id": parent.state_id})
    with pytest.raises(StateIntegrityError, match="step 0.*own parent"):
        impossible_root.validate_integrity()

    unrelated_parent = GraceState.from_parts(
        graph=graph,
        instruction="Unrelated.",
        schema_id="other-schema",
        schema_hash="other-hash",
        step=3,
        parent_state_id="an-earlier-state",
    )
    with pytest.raises(
        StateIntegrityError,
        match="parent_state_id.*step is not exactly.*schema_id differs.*schema_hash differs",
    ):
        child.validate_integrity(parent_state=unrelated_parent)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "node": Node(id="N1", type="unit", content="Node."),
            "edge": Edge(source="N1", target="N1", relation="links"),
        },
    ],
)
def test_forced_drop_requires_exactly_one_payload(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ForcedDrop(
            kind=RepairDropKind.DANGLING_EDGE,
            reason="Invalid payload.",
            original_index=0,
            **kwargs,
        )


def test_forced_drop_validates_payload_kind_specific_fields() -> None:
    node = Node(id="N1", type="unit", content="Node.")
    edge = Edge(source="N1", target="N1", relation="links")

    cases = (
        (
            {"kind": RepairDropKind.DUPLICATE_NODE, "edge": edge, "kept_index": 0},
            "node payload",
        ),
        ({"kind": RepairDropKind.DANGLING_EDGE, "node": node}, "edge payload"),
        ({"kind": RepairDropKind.DUPLICATE_EDGE, "edge": edge}, "kept_index"),
        (
            {"kind": RepairDropKind.EMPTY_AFTER_ID_STRIP, "node": node},
            "cleaned_content",
        ),
    )
    for fields, message in cases:
        with pytest.raises(ValidationError, match=message):
            ForcedDrop(reason="Invalid payload.", original_index=0, **fields)


def test_repair_rejects_wrong_types_and_alias_preserves_result() -> None:
    schema = _unit_schema(acyclic=True)
    graph = _unit_graph()

    with pytest.raises(TypeError, match="GraphState"):
        repair_graph(graph.model_dump(mode="python"), schema)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NetworkSchema"):
        repair_graph(graph, object())  # type: ignore[arg-type]

    assert repair_schema(graph, schema) == repair_graph(graph, schema)


def test_repair_handles_converging_dag_without_false_cycle_drop() -> None:
    schema = _unit_schema(acyclic=True)
    graph = GraphState(
        nodes=_unit_graph().nodes,
        edges=(
            Edge(source="N001", target="N003", relation="links"),
            Edge(source="N002", target="N003", relation="links"),
        ),
    )

    result = repair_graph(graph, schema)

    assert result.forced_drops == ()
    assert result.graph.edges == graph.edges


def test_diff_and_enrichment_cover_every_change_variant_and_mapping_boundary() -> None:
    schema = _unit_schema()
    previous = GraphState(
        nodes=(
            Node(id="N001", type="unit", content="Remove me."),
            Node(id="N002", type="unit", content="Old content."),
            Node(id="N003", type="unit", content="Stable."),
        ),
        edges=(
            Edge(source="N001", target="N003", relation="links"),
            Edge(source="N002", target="N003", relation="links"),
        ),
    )
    current = GraphState(
        nodes=(
            Node(id="N002", type="other", content="New content."),
            Node(id="N003", type="unit", content="Stable."),
            Node(id="N004", type="unit", content="Added."),
        ),
        edges=(
            Edge(source="N002", target="N003", relation="links"),
            Edge(source="N003", target="N004", relation="links"),
        ),
    )

    changes = diff_graphs(previous, current)
    assert [change.op for change in changes] == [
        "AddNode",
        "ModifyNode",
        "RemoveNode",
        "AddEdge",
        "RemoveEdge",
    ]
    raw_changes = [change.model_dump(mode="python") for change in changes]
    enriched = enrich_changes(raw_changes, previous, current, schema)
    assert [change.op for change in enriched] == [
        "AddNode",
        "ModifyNode",
        "RemoveNode",
        "AddEdge",
        "RemoveEdge",
    ]
    assert diff_log(previous, current) == [change.model_dump(mode="json") for change in changes]
    assert enrich_change_log(raw_changes, previous, current, schema) == [
        change.model_dump(mode="json") for change in enriched
    ]


def test_reconstruction_sanitizes_recursive_and_invalid_telemetry() -> None:
    recursive: list[object] = []
    recursive.append(recursive)
    result = apply_patch_operations(
        "Instruction\n",
        [
            {
                "op": "ReplaceSpan",
                "anchor": "Instruction",
                "extra": recursive,
            },
            {
                "op": "ReplaceSpan",
                "anchor": "Instruction",
                "new_text": "Replacement",
                "extra": float("inf"),
            },
        ],
    )

    assert result.status == ReconstructionStatus.COMPLETED_WITH_FALLBACKS
    assert len(result.skipped_operations) == 2
    assert result.skipped_operations[0]["validation_errors"]
    assert "<recursive>" in json.dumps(result.proposed_operations)
    json.dumps(result.proposed_operations, allow_nan=False)


def test_reconstruction_fallback_separator_and_result_integrity_guards() -> None:
    appended = apply_patch_operations(
        "Already terminated.\n",
        [{"op": "InsertAfter", "anchor": "missing", "new_text": "Fallback"}],
    )
    assert appended.instruction == "Already terminated.\nFallback"
    assert appended.appended_operations[0]["separator_inserted"] is False
    assert appended.output_instruction == appended.instruction

    with pytest.raises(ValidationError, match="exactly one outcome"):
        PatchApplicationResult(
            instruction="Instruction",
            status=ReconstructionStatus.NO_OP,
            input_instruction_hash=_sha256("Instruction"),
            output_instruction_hash=_sha256("Instruction"),
            proposed_operations=({"index": 0, "operation": {}},),
        )
    with pytest.raises(ValidationError, match="output_instruction_hash"):
        PatchApplicationResult(
            instruction="Instruction",
            status=ReconstructionStatus.NO_OP,
            input_instruction_hash=_sha256("Instruction"),
            output_instruction_hash="incorrect",
        )


def test_reconstruction_rejects_non_utf8_previous_instruction() -> None:
    with pytest.raises(ValueError, match="instruction text must be valid UTF-8"):
        apply_patch_operations("Previous \ud800 instruction", ())


def test_reconstruction_rejects_non_utf8_replacement_text() -> None:
    anchor = "<UNIQUE-GRACE-ANCHOR>"

    with pytest.raises(ValueError, match="instruction text must be valid UTF-8"):
        apply_patch_operations(
            f"Previous {anchor} instruction",
            [ReplaceSpan(anchor=anchor, new_text="Replacement \ud800 text")],
        )


@settings(max_examples=30, deadline=None)
@given(st.permutations(("N001", "N002", "N003")))
def test_graph_hash_is_a_set_semantic_under_all_node_permutations(
    order: list[str],
) -> None:
    nodes = {
        "N001": Node(id="N001", type="unit", content="First."),
        "N002": Node(id="N002", type="unit", content="Second."),
        "N003": Node(id="N003", type="unit", content="Third."),
    }
    graph = GraphState(nodes=tuple(nodes[node_id] for node_id in order))
    canonical = GraphState(nodes=tuple(nodes.values()))

    assert graph.content_hash() == canonical.content_hash()


@settings(max_examples=40, deadline=None)
@given(
    prefix=st.text(
        alphabet=st.characters(codec="utf-8", blacklist_characters="\x00"),
        max_size=30,
    ),
    suffix=st.text(
        alphabet=st.characters(codec="utf-8", blacklist_characters="\x00"),
        max_size=30,
    ),
    replacement=st.text(
        alphabet=st.characters(codec="utf-8", blacklist_characters="\x00"),
        max_size=30,
    ),
)
def test_unique_anchor_replacement_preserves_all_unedited_bytes(
    prefix: str,
    suffix: str,
    replacement: str,
) -> None:
    anchor = "<UNIQUE-GRACE-ANCHOR>"
    prefix = prefix.replace(anchor, "")
    suffix = suffix.replace(anchor, "")
    instruction = prefix + anchor + suffix

    result = apply_patch_operations(
        instruction,
        [ReplaceSpan(anchor=anchor, new_text=replacement)],
    )

    assert result.instruction == prefix + replacement + suffix
    assert result.status == ReconstructionStatus.APPLIED
