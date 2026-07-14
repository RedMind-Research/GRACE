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

"""Deterministic graph differencing and reconstruction-safe enrichment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from grace.graph.models import Edge, GraphState, Node
from grace.graph.neighborhood import is_grounding_backbone_edge
from grace.schemas.models import NetworkSchema


class _Change(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AddNodeChange(_Change):
    op: Literal["AddNode"] = "AddNode"
    node_id: str
    type: str
    content: str


class ModifyNodeChange(_Change):
    op: Literal["ModifyNode"] = "ModifyNode"
    node_id: str
    type: str
    old_type: str
    new_type: str
    old_content: str
    new_content: str


class RemoveNodeChange(_Change):
    op: Literal["RemoveNode"] = "RemoveNode"
    node_id: str
    type: str
    content: str


class AddEdgeChange(_Change):
    op: Literal["AddEdge"] = "AddEdge"
    source: str
    target: str
    relation: str


class RemoveEdgeChange(_Change):
    op: Literal["RemoveEdge"] = "RemoveEdge"
    source: str
    target: str
    relation: str


GraphChange: TypeAlias = Annotated[
    AddNodeChange | ModifyNodeChange | RemoveNodeChange | AddEdgeChange | RemoveEdgeChange,
    Field(discriminator="op"),
]


class EnrichedAddNodeChange(_Change):
    op: Literal["AddNode"] = "AddNode"
    type: str
    content: str


class EnrichedModifyNodeChange(_Change):
    op: Literal["ModifyNode"] = "ModifyNode"
    type: str
    old_type: str
    new_type: str
    old_content: str
    new_content: str


class EnrichedRemoveNodeChange(_Change):
    op: Literal["RemoveNode"] = "RemoveNode"
    type: str
    content: str


class EnrichedAddEdgeChange(_Change):
    op: Literal["AddEdge"] = "AddEdge"
    relation: str
    source_content: str
    target_content: str


class EnrichedRemoveEdgeChange(_Change):
    op: Literal["RemoveEdge"] = "RemoveEdge"
    relation: str
    source_content: str
    target_content: str


EnrichedGraphChange: TypeAlias = Annotated[
    EnrichedAddNodeChange
    | EnrichedModifyNodeChange
    | EnrichedRemoveNodeChange
    | EnrichedAddEdgeChange
    | EnrichedRemoveEdgeChange,
    Field(discriminator="op"),
]


_CHANGE_ADAPTER: TypeAdapter[GraphChange] = TypeAdapter(GraphChange)


def _edge_keys(graph: GraphState) -> frozenset[tuple[str, str, str]]:
    """Return structural edge identities; edge metadata is non-semantic here."""

    return frozenset((edge.source, edge.target, edge.relation) for edge in graph.edges)


def _node_changed(previous: Node, current: Node) -> bool:
    """Node-level policy changes cover type and deployable content, not metadata."""

    return previous.type != current.type or previous.content != current.content


def diff_graphs(previous: GraphState, current: GraphState) -> tuple[GraphChange, ...]:
    """Return a stable component-wise graph difference from ``previous`` to ``current``.

    A merge naturally appears as ``ModifyNode`` plus ``RemoveNode``.  Results do
    not depend on either graph's list ordering.  Graph/node/edge metadata is
    intentionally excluded because reconstruction operates on deployable policy
    content and typed structural relations only.
    """

    previous_nodes = {node.id: node for node in previous.nodes}
    current_nodes = {node.id: node for node in current.nodes}
    changes: list[GraphChange] = []

    for node_id in sorted(current_nodes.keys() - previous_nodes.keys()):
        node = current_nodes[node_id]
        changes.append(AddNodeChange(node_id=node.id, type=node.type, content=node.content))

    for node_id in sorted(current_nodes.keys() & previous_nodes.keys()):
        old_node = previous_nodes[node_id]
        new_node = current_nodes[node_id]
        if _node_changed(old_node, new_node):
            changes.append(
                ModifyNodeChange(
                    node_id=node_id,
                    type=new_node.type,
                    old_type=old_node.type,
                    new_type=new_node.type,
                    old_content=old_node.content,
                    new_content=new_node.content,
                )
            )

    for node_id in sorted(previous_nodes.keys() - current_nodes.keys()):
        node = previous_nodes[node_id]
        changes.append(RemoveNodeChange(node_id=node.id, type=node.type, content=node.content))

    previous_edges = _edge_keys(previous)
    current_edges = _edge_keys(current)
    for source, target, relation in sorted(current_edges - previous_edges):
        changes.append(AddEdgeChange(source=source, target=target, relation=relation))
    for source, target, relation in sorted(previous_edges - current_edges):
        changes.append(RemoveEdgeChange(source=source, target=target, relation=relation))

    return tuple(changes)


def _coerce_change(change: GraphChange | Mapping[str, Any]) -> GraphChange:
    if isinstance(change, _Change):
        return change
    return _CHANGE_ADAPTER.validate_python(change)


def enrich_changes(
    changes: Iterable[GraphChange | Mapping[str, Any]],
    previous: GraphState,
    current: GraphState,
    schema: NetworkSchema,
) -> tuple[EnrichedGraphChange, ...]:
    """Convert a graph diff into an ID-free, content-anchored reconstruction log.

    Node IDs are removed.  Edge endpoint IDs become endpoint content, because
    the deployed instruction contains no graph identifiers.  Grounding
    scaffold relations are omitted according to the configured schema policy;
    schemas without a grounding policy omit nothing.

    Current nodes override previous nodes when both contain the same ID, which
    preserves the established GRACE reconstruction convention while still
    resolving content for removed nodes from the previous graph.
    """

    nodes = {node.id: node for node in previous.nodes}
    nodes.update({node.id: node for node in current.nodes})
    node_types = {node_id: node.type for node_id, node in nodes.items()}
    enriched: list[EnrichedGraphChange] = []

    for raw_change in changes:
        change = _coerce_change(raw_change)
        if isinstance(change, AddNodeChange):
            enriched.append(EnrichedAddNodeChange(type=change.type, content=change.content))
        elif isinstance(change, ModifyNodeChange):
            enriched.append(
                EnrichedModifyNodeChange(
                    type=change.type,
                    old_type=change.old_type,
                    new_type=change.new_type,
                    old_content=change.old_content,
                    new_content=change.new_content,
                )
            )
        elif isinstance(change, RemoveNodeChange):
            enriched.append(EnrichedRemoveNodeChange(type=change.type, content=change.content))
        else:
            edge = Edge(
                source=change.source,
                target=change.target,
                relation=change.relation,
            )
            if is_grounding_backbone_edge(edge, node_types, schema):
                continue
            source = nodes.get(change.source)
            target = nodes.get(change.target)
            source_content = source.content if source is not None else ""
            target_content = target.content if target is not None else ""
            if isinstance(change, AddEdgeChange):
                enriched.append(
                    EnrichedAddEdgeChange(
                        relation=change.relation,
                        source_content=source_content,
                        target_content=target_content,
                    )
                )
            else:
                enriched.append(
                    EnrichedRemoveEdgeChange(
                        relation=change.relation,
                        source_content=source_content,
                        target_content=target_content,
                    )
                )

    return tuple(enriched)


def diff_log(previous: GraphState, current: GraphState) -> list[dict[str, JsonValue]]:
    """Dictionary compatibility form of :func:`diff_graphs`."""

    return [change.model_dump(mode="json") for change in diff_graphs(previous, current)]


def enrich_change_log(
    change_log: Iterable[GraphChange | Mapping[str, Any]],
    previous: GraphState,
    current: GraphState,
    schema: NetworkSchema,
) -> list[dict[str, JsonValue]]:
    """Dictionary compatibility form of :func:`enrich_changes`."""

    return [
        change.model_dump(mode="json")
        for change in enrich_changes(change_log, previous, current, schema)
    ]


__all__ = [
    "AddEdgeChange",
    "AddNodeChange",
    "EnrichedAddEdgeChange",
    "EnrichedAddNodeChange",
    "EnrichedGraphChange",
    "EnrichedModifyNodeChange",
    "EnrichedRemoveEdgeChange",
    "EnrichedRemoveNodeChange",
    "GraphChange",
    "ModifyNodeChange",
    "RemoveEdgeChange",
    "RemoveNodeChange",
    "diff_graphs",
    "diff_log",
    "enrich_change_log",
    "enrich_changes",
]
