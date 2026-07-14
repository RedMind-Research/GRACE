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

"""Explicit, destructive graph repair for cap/fallback handling.

This module is intentionally *not* part of the ordinary P2G validation loop.
Callers may invoke :func:`repair_graph` explicitly after a repair cap is
reached, when deterministic removal is preferable to returning an invalid
graph.  Every discarded object is retained as typed audit telemetry.

The implementation is schema-driven: relation membership, type signatures,
self-loop policy, and acyclic relations all come from ``NetworkSchema``.  It
therefore contains no knowledge of the paper's default ontology.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grace.graph.models import Edge, GraphState, Node
from grace.schemas.models import NetworkSchema


_NODE_ID_REFERENCE_RE = re.compile(r"\bN(?:_new)?_?\d+\b")
_MIN_CONTENT_LENGTH_AFTER_STRIP = 5


class RepairDropKind(str, Enum):
    """Stable machine-readable reasons for destructive repair."""

    DUPLICATE_NODE = "duplicate_node"
    EMPTY_AFTER_ID_STRIP = "empty_after_id_strip"
    DANGLING_EDGE = "dangling_edge"
    UNKNOWN_RELATION = "unknown_relation"
    SELF_LOOP = "self_loop"
    TYPE_SIGNATURE_VIOLATION = "type_signature_violation"
    DUPLICATE_EDGE = "duplicate_edge"
    RELATION_CYCLE_BREAK = "relation_cycle_break"


class ForcedDrop(BaseModel):
    """One immutable, lossless audit record for a discarded graph object.

    ``original_index`` always addresses the corresponding tuple in the input
    graph, even when an edge is removed later during cycle breaking.
    ``kept_index`` identifies the retained first occurrence for duplicate
    objects.  ``cleaned_content`` records the post-strip value when a node is
    too short to keep.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RepairDropKind
    reason: str = Field(min_length=1)
    original_index: int = Field(ge=0)
    node: Node | None = None
    edge: Edge | None = None
    kept_index: int | None = Field(default=None, ge=0)
    cleaned_content: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> ForcedDrop:
        if (self.node is None) == (self.edge is None):
            raise ValueError("a forced drop must contain exactly one node or edge")
        if self.kind in {
            RepairDropKind.DUPLICATE_NODE,
            RepairDropKind.EMPTY_AFTER_ID_STRIP,
        }:
            if self.node is None:
                raise ValueError("node repair drops require a node payload")
        elif self.edge is None:
            raise ValueError("edge repair drops require an edge payload")
        if (
            self.kind
            in {
                RepairDropKind.DUPLICATE_NODE,
                RepairDropKind.DUPLICATE_EDGE,
            }
            and self.kept_index is None
        ):
            raise ValueError("duplicate repair drops require kept_index")
        if self.kind == RepairDropKind.EMPTY_AFTER_ID_STRIP and self.cleaned_content is None:
            raise ValueError("content-strip drops require cleaned_content")
        return self


class GraphRepairResult(BaseModel):
    """The repaired immutable graph and all deterministic forced drops."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph: GraphState
    forced_drops: tuple[ForcedDrop, ...] = ()


def _strip_node_id_references(content: str) -> str:
    """Remove legacy/provider graph IDs and normalize resulting whitespace."""

    without_ids = _NODE_ID_REFERENCE_RE.sub("", content)
    return re.sub(r"\s+", " ", without_ids).strip()


def _edge_key(edge: Edge) -> tuple[str, str, str]:
    return edge.source, edge.target, edge.relation


def _find_stable_back_edge(
    *,
    node_ids: set[str],
    indexed_edges: list[tuple[int, Edge]],
    relation: str,
) -> tuple[str, str, str] | None:
    """Return one deterministic DFS back-edge for ``relation``, if present.

    Roots and outgoing targets are traversed lexicographically, so the choice
    is independent of set/hash iteration and Python process randomization.
    The iterative DFS avoids recursion limits on large externally supplied
    graphs.
    """

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for _, edge in indexed_edges:
        if edge.relation == relation:
            adjacency[edge.source].append(edge.target)
    for targets in adjacency.values():
        targets.sort()

    white, gray, black = 0, 1, 2
    colors = {node_id: white for node_id in node_ids}

    for root in sorted(node_ids):
        if colors[root] != white:
            continue
        colors[root] = gray
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            source, target_index = stack[-1]
            targets = adjacency[source]
            if target_index >= len(targets):
                colors[source] = black
                stack.pop()
                continue

            target = targets[target_index]
            stack[-1] = (source, target_index + 1)
            if colors[target] == gray:
                return source, target, relation
            if colors[target] == white:
                colors[target] = gray
                stack.append((target, 0))

    return None


def repair_graph(graph: GraphState, schema: NetworkSchema) -> GraphRepairResult:
    """Apply deterministic, destructive fallback repair under ``schema``.

    The input graph is never mutated.  Node-ID tokens in content are stripped;
    a node is discarded when fewer than five characters remain.  Duplicate
    IDs and structural duplicate edges retain their first input occurrence.
    Invalid edges are removed, then each configured acyclic relation is made
    acyclic by repeatedly dropping its deterministic DFS back-edge.

    Unknown object types are deliberately not guessed or rewritten.  They can
    remain as validation errors, while any incident edge that cannot satisfy a
    declared signature is removed.  This keeps repair deterministic without
    inventing ontology semantics.
    """

    if not isinstance(graph, GraphState):
        raise TypeError("graph must be a GraphState")
    if not isinstance(schema, NetworkSchema):
        raise TypeError("schema must be a NetworkSchema")

    forced_drops: list[ForcedDrop] = []

    # Preserve legacy semantics: the first occurrence claims an ID even when
    # that node is subsequently too short after ID-token stripping.
    seen_node_indices: dict[str, int] = {}
    kept_nodes: list[Node] = []
    for node_index, node in enumerate(graph.nodes):
        kept_index = seen_node_indices.get(node.id)
        if kept_index is not None:
            forced_drops.append(
                ForcedDrop(
                    kind=RepairDropKind.DUPLICATE_NODE,
                    reason=f"duplicate node id {node.id!r}",
                    original_index=node_index,
                    kept_index=kept_index,
                    node=node,
                )
            )
            continue
        seen_node_indices[node.id] = node_index

        if _NODE_ID_REFERENCE_RE.search(node.content):
            cleaned_content = _strip_node_id_references(node.content)
            if len(cleaned_content) < _MIN_CONTENT_LENGTH_AFTER_STRIP:
                forced_drops.append(
                    ForcedDrop(
                        kind=RepairDropKind.EMPTY_AFTER_ID_STRIP,
                        reason=(
                            "node content is shorter than five characters after graph-ID removal"
                        ),
                        original_index=node_index,
                        node=node,
                        cleaned_content=cleaned_content,
                    )
                )
                continue
            node = node.model_copy(update={"content": cleaned_content})
        kept_nodes.append(node)

    valid_node_ids = {node.id for node in kept_nodes}
    node_types = {node.id: node.type for node in kept_nodes}
    relation_ids = set(schema.relation_type_ids)

    seen_edge_indices: dict[tuple[str, str, str], int] = {}
    kept_indexed_edges: list[tuple[int, Edge]] = []
    for edge_index, edge in enumerate(graph.edges):
        key = _edge_key(edge)
        if edge.source not in valid_node_ids or edge.target not in valid_node_ids:
            forced_drops.append(
                ForcedDrop(
                    kind=RepairDropKind.DANGLING_EDGE,
                    reason="edge endpoint does not identify a retained node",
                    original_index=edge_index,
                    edge=edge,
                )
            )
            continue
        if edge.relation not in relation_ids:
            forced_drops.append(
                ForcedDrop(
                    kind=RepairDropKind.UNKNOWN_RELATION,
                    reason=f"relation {edge.relation!r} is not declared by the schema",
                    original_index=edge_index,
                    edge=edge,
                )
            )
            continue
        if edge.source == edge.target and not schema.relation_allows_self_loop(edge.relation):
            forced_drops.append(
                ForcedDrop(
                    kind=RepairDropKind.SELF_LOOP,
                    reason=f"relation {edge.relation!r} does not allow self-loops",
                    original_index=edge_index,
                    edge=edge,
                )
            )
            continue
        if not schema.edge_type_ok(
            node_types[edge.source],
            node_types[edge.target],
            edge.relation,
        ):
            forced_drops.append(
                ForcedDrop(
                    kind=RepairDropKind.TYPE_SIGNATURE_VIOLATION,
                    reason=(
                        f"({node_types[edge.source]!r}, {node_types[edge.target]!r}) "
                        f"is not allowed for relation {edge.relation!r}"
                    ),
                    original_index=edge_index,
                    edge=edge,
                )
            )
            continue
        duplicate_index = seen_edge_indices.get(key)
        if duplicate_index is not None:
            forced_drops.append(
                ForcedDrop(
                    kind=RepairDropKind.DUPLICATE_EDGE,
                    reason="duplicate structural edge",
                    original_index=edge_index,
                    kept_index=duplicate_index,
                    edge=edge,
                )
            )
            continue
        seen_edge_indices[key] = edge_index
        kept_indexed_edges.append((edge_index, edge))

    # Repair all acyclic relations, not just the paper's default names.
    # Re-scan after each removal because one relation may contain many cycles.
    for relation in sorted(schema.acyclic_relations):
        while True:
            back_edge = _find_stable_back_edge(
                node_ids=valid_node_ids,
                indexed_edges=kept_indexed_edges,
                relation=relation,
            )
            if back_edge is None:
                break
            for position, (original_index, edge) in enumerate(kept_indexed_edges):
                if _edge_key(edge) != back_edge:
                    continue
                kept_indexed_edges.pop(position)
                forced_drops.append(
                    ForcedDrop(
                        kind=RepairDropKind.RELATION_CYCLE_BREAK,
                        reason=f"edge closes a cycle in relation {relation!r}",
                        original_index=original_index,
                        edge=edge,
                    )
                )
                break

    metadata = dict(graph.metadata)
    metadata["total_nodes"] = len(kept_nodes)
    metadata["total_edges"] = len(kept_indexed_edges)
    repaired_graph = GraphState(
        nodes=tuple(kept_nodes),
        edges=tuple(edge for _, edge in kept_indexed_edges),
        metadata=metadata,
    )
    return GraphRepairResult(
        graph=repaired_graph,
        forced_drops=tuple(forced_drops),
    )


def repair_schema(graph: GraphState, schema: NetworkSchema) -> GraphRepairResult:
    """Compatibility alias for callers migrating from ``repair_schema``."""

    return repair_graph(graph, schema)


__all__ = [
    "ForcedDrop",
    "GraphRepairResult",
    "RepairDropKind",
    "repair_graph",
    "repair_schema",
]
