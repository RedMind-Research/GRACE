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

"""Deterministic validation for schema-configured GRACE graphs.

This module deliberately contains no knowledge of the paper's default
ontology.  Every object type, relation signature, and acyclicity constraint is
read from the supplied :class:`~grace.schemas.models.NetworkSchema`.

``validate_graph`` also accepts a raw mapping.  That small concession is
important at serialization and provider boundaries: it lets callers receive a
complete, structured report for missing fields instead of losing the original
failure behind Pydantic's first parse exception.  Valid in-process code should
normally pass a typed :class:`~grace.graph.models.GraphState`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, computed_field

from grace.graph.models import GraphState
from grace.schemas.models import NetworkSchema


_NODE_FIELDS = ("id", "type", "content")
_EDGE_FIELDS = ("source", "target", "relation")
_NODE_ID_REFERENCE_RE = re.compile(r"\bN(?:_new)?_?\d+\b")


class IssueSeverity(str, Enum):
    """Severity of one deterministic graph-validation issue."""

    ERROR = "error"
    WARNING = "warning"


class GraphValidationIssue(BaseModel):
    """One stable, machine-readable graph-validation finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: IssueSeverity = IssueSeverity.ERROR
    node_id: str | None = None
    edge_index: int | None = Field(default=None, ge=0)
    field: str | None = None
    context: dict[str, JsonValue] = Field(default_factory=dict)


class GraphValidationReport(BaseModel):
    """Complete deterministic validation result.

    Findings are tuples so a returned report cannot be mutated after the graph
    has moved on to another evolution step.  ``valid`` and the compatibility
    alias ``passed`` are serialized computed fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    errors: tuple[GraphValidationIssue, ...] = ()
    warnings: tuple[GraphValidationIssue, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valid(self) -> bool:
        return not self.errors

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Compatibility spelling for callers using pass/fail terminology."""

        return self.valid


def object_type_ids(schema: NetworkSchema) -> frozenset[str]:
    """Return all object-type IDs declared by ``schema``."""

    return frozenset(schema.object_type_ids)


def relation_type_ids(schema: NetworkSchema) -> frozenset[str]:
    """Return all relation-type IDs declared by ``schema``."""

    return frozenset(schema.relation_type_ids)


def allowed_pairs(schema: NetworkSchema, relation: str) -> frozenset[tuple[str, str]]:
    """Return the configured source/target signatures for one relation."""

    definition = schema.relation_type_map.get(relation)
    return frozenset(definition.allowed_pairs) if definition is not None else frozenset()


def relation_allows(
    schema: NetworkSchema,
    relation: str,
    source_type: str,
    target_type: str,
) -> bool:
    """Whether an edge type signature is admitted by the configured schema."""

    return schema.edge_type_ok(source_type, target_type, relation)


def acyclic_relation_ids(schema: NetworkSchema) -> frozenset[str]:
    """Return relation IDs whose induced directed subgraphs must be acyclic."""

    return schema.acyclic_relations


def _graph_payload(graph: GraphState | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(graph, GraphState):
        return graph.model_dump(mode="python")
    if isinstance(graph, Mapping):
        return graph
    return {}


def _mapping_sequence(payload: Mapping[str, Any], field: str) -> tuple[Any, ...] | None:
    value = payload.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return tuple(value)


def relation_has_cycle(
    graph: GraphState | Mapping[str, Any],
    relation: str,
) -> bool:
    """Detect a directed cycle in the subgraph induced by ``relation``.

    Dangling endpoints are ignored here because validation reports them
    separately and they cannot participate in a cycle over the graph's nodes.
    """

    payload = _graph_payload(graph)
    raw_nodes = _mapping_sequence(payload, "nodes") or ()
    raw_edges = _mapping_sequence(payload, "edges") or ()
    node_ids = {
        str(node.get("id"))
        for node in raw_nodes
        if isinstance(node, Mapping) and node.get("id") is not None
    }
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in raw_edges:
        if not isinstance(edge, Mapping) or str(edge.get("relation")) != relation:
            continue
        source, target = edge.get("source"), edge.get("target")
        if source is None or target is None:
            continue
        source, target = str(source), str(target)
        if source in node_ids and target in node_ids:
            adjacency[source].add(target)

    white, gray, black = 0, 1, 2
    colors = {node_id: white for node_id in node_ids}

    def visit(node_id: str) -> bool:
        colors[node_id] = gray
        for neighbor in sorted(adjacency[node_id]):
            if colors[neighbor] == gray:
                return True
            if colors[neighbor] == white and visit(neighbor):
                return True
        colors[node_id] = black
        return False

    return any(colors[node_id] == white and visit(node_id) for node_id in sorted(node_ids))


def validate_graph(
    graph: GraphState | Mapping[str, Any],
    schema: NetworkSchema,
) -> GraphValidationReport:
    """Validate all deterministic graph invariants under ``schema``.

    Checks are deliberately accumulated rather than short-circuited: required
    data, duplicate IDs, empty content, dangling endpoints, type signatures,
    self-loops, duplicate edges, and every relation marked ``acyclic``.
    """

    errors: list[GraphValidationIssue] = []
    warnings: list[GraphValidationIssue] = []
    payload = _graph_payload(graph)

    if not isinstance(graph, (GraphState, Mapping)):
        errors.append(
            GraphValidationIssue(
                code="graph_type",
                message="graph must be a GraphState or mapping",
            )
        )

    raw_nodes = _mapping_sequence(payload, "nodes")
    raw_edges = _mapping_sequence(payload, "edges")
    if raw_nodes is None:
        errors.append(
            GraphValidationIssue(
                code="required_data",
                field="nodes",
                message="graph field 'nodes' is required and must be a sequence",
            )
        )
        raw_nodes = ()
    if raw_edges is None:
        errors.append(
            GraphValidationIssue(
                code="required_data",
                field="edges",
                message="graph field 'edges' is required and must be a sequence",
            )
        )
        raw_edges = ()

    declared_object_types = object_type_ids(schema)
    declared_relations = relation_type_ids(schema)
    node_types: dict[str, str] = {}
    seen_node_ids: set[str] = set()

    for node_index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            errors.append(
                GraphValidationIssue(
                    code="node_type",
                    message=f"node at index {node_index} must be a mapping",
                    context={"node_index": node_index},
                )
            )
            continue

        node_id_value = raw_node.get("id")
        node_id = None if node_id_value is None else str(node_id_value)
        issue_node_id = node_id or f"<node-{node_index}>"
        for field in _NODE_FIELDS:
            if field not in raw_node or raw_node.get(field) is None:
                errors.append(
                    GraphValidationIssue(
                        code="required_field",
                        node_id=issue_node_id,
                        field=field,
                        message=f"node is missing required field '{field}'",
                        context={"node_index": node_index},
                    )
                )

        if node_id is not None:
            if not node_id.strip():
                errors.append(
                    GraphValidationIssue(
                        code="empty_id",
                        node_id=issue_node_id,
                        field="id",
                        message="node id is empty",
                        context={"node_index": node_index},
                    )
                )
            if node_id in seen_node_ids:
                errors.append(
                    GraphValidationIssue(
                        code="duplicate_node_id",
                        node_id=issue_node_id,
                        message=f"duplicate node id '{node_id}'",
                        context={"node_index": node_index},
                    )
                )
            else:
                seen_node_ids.add(node_id)
                node_types[node_id] = str(raw_node.get("type", ""))

        object_type = str(raw_node.get("type", ""))
        if object_type not in declared_object_types:
            errors.append(
                GraphValidationIssue(
                    code="object_type",
                    node_id=issue_node_id,
                    field="type",
                    message=f"unknown object type '{object_type}'",
                    context={"allowed": cast(list[JsonValue], sorted(declared_object_types))},
                )
            )

        content = str(raw_node.get("content", ""))
        if not content.strip():
            errors.append(
                GraphValidationIssue(
                    code="empty_content",
                    node_id=issue_node_id,
                    field="content",
                    message="node content is empty",
                )
            )
        elif _NODE_ID_REFERENCE_RE.search(content):
            errors.append(
                GraphValidationIssue(
                    code="id_reference",
                    node_id=issue_node_id,
                    field="content",
                    message="node content contains a graph node-ID token",
                )
            )

    seen_edges: set[tuple[str, str, str]] = set()
    for edge_index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, Mapping):
            errors.append(
                GraphValidationIssue(
                    code="edge_type",
                    edge_index=edge_index,
                    message=f"edge at index {edge_index} must be a mapping",
                )
            )
            continue

        for field in _EDGE_FIELDS:
            if field not in raw_edge or raw_edge.get(field) is None:
                errors.append(
                    GraphValidationIssue(
                        code="edge_required_field",
                        edge_index=edge_index,
                        field=field,
                        message=f"edge is missing required field '{field}'",
                    )
                )

        source = "" if raw_edge.get("source") is None else str(raw_edge.get("source"))
        target = "" if raw_edge.get("target") is None else str(raw_edge.get("target"))
        relation = "" if raw_edge.get("relation") is None else str(raw_edge.get("relation"))
        edge_key = (source, target, relation)

        if relation not in declared_relations:
            errors.append(
                GraphValidationIssue(
                    code="edge_relation",
                    edge_index=edge_index,
                    field="relation",
                    message=f"unknown relation type '{relation}'",
                    context={"allowed": cast(list[JsonValue], sorted(declared_relations))},
                )
            )
        if source not in node_types:
            errors.append(
                GraphValidationIssue(
                    code="dangling_source",
                    edge_index=edge_index,
                    field="source",
                    message=f"edge source '{source}' does not identify a graph node",
                    context={"source": source},
                )
            )
        if target not in node_types:
            errors.append(
                GraphValidationIssue(
                    code="dangling_target",
                    edge_index=edge_index,
                    field="target",
                    message=f"edge target '{target}' does not identify a graph node",
                    context={"target": target},
                )
            )
        if source and source == target and not schema.relation_allows_self_loop(relation):
            errors.append(
                GraphValidationIssue(
                    code="self_loop",
                    edge_index=edge_index,
                    message=f"self-loop on node '{source}' is not allowed",
                    context={"edge": list(edge_key)},
                )
            )
        if edge_key in seen_edges:
            errors.append(
                GraphValidationIssue(
                    code="duplicate_edge",
                    edge_index=edge_index,
                    message="duplicate edge",
                    context={"edge": list(edge_key)},
                )
            )
        else:
            seen_edges.add(edge_key)

        if (
            relation in declared_relations
            and source in node_types
            and target in node_types
            and not relation_allows(
                schema,
                relation,
                node_types[source],
                node_types[target],
            )
        ):
            errors.append(
                GraphValidationIssue(
                    code="type_signature",
                    edge_index=edge_index,
                    message=(
                        f"({node_types[source]}, {node_types[target]}) is not "
                        f"allowed for relation '{relation}'"
                    ),
                    context={
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "source_type": node_types[source],
                        "target_type": node_types[target],
                    },
                )
            )

    for relation in sorted(acyclic_relation_ids(schema)):
        if relation_has_cycle(payload, relation):
            errors.append(
                GraphValidationIssue(
                    code="relation_cycle",
                    message=f"relation '{relation}' contains a directed cycle",
                    context={"relation": relation},
                )
            )

    return GraphValidationReport(errors=tuple(errors), warnings=tuple(warnings))


def validate_schema(
    graph: GraphState | Mapping[str, Any],
    schema: NetworkSchema,
) -> GraphValidationReport:
    """Compatibility alias for callers migrating from ``validate_schema``."""

    return validate_graph(graph, schema)


__all__ = [
    "GraphValidationIssue",
    "GraphValidationReport",
    "IssueSeverity",
    "acyclic_relation_ids",
    "allowed_pairs",
    "object_type_ids",
    "relation_allows",
    "relation_has_cycle",
    "relation_type_ids",
    "validate_graph",
    "validate_schema",
]
