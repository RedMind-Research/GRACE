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

"""Deterministic, schema-driven grounding-backbone construction.

The graph layer must not assume that an anchor is called ``identity``, that a
governed object is called ``norm``, or that either relation has a default-
schema-specific name. Those choices belong exclusively to the optional
:class:`~grace.schemas.models.GroundingPolicy` carried by a network schema.

For a configured policy, the lexicographically first root node is the stable
top root.  It supports every configured grounded node, while every other root
is connected upward to the top root when doing so is not redundant and does
not violate the schema.  The input graph is never mutated.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from grace.graph.models import Edge, GraphState
from grace.graph.validation import relation_has_cycle
from grace.schemas.models import NetworkSchema


GroundingPhase = Literal["policy", "support", "hierarchy"]


class GroundingSkip(BaseModel):
    """One candidate grounding action that was deliberately not applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: GroundingPhase
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    source: str | None = None
    target: str | None = None
    relation: str | None = None


class GroundingResult(BaseModel):
    """Graph plus deterministic telemetry from grounding-policy application."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph: GraphState
    top_root_id: str | None = None
    added_edges: tuple[Edge, ...] = ()
    skipped: tuple[GroundingSkip, ...] = ()

    @property
    def added_count(self) -> int:
        return len(self.added_edges)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def _reachable(edges: Iterable[Edge], source: str, target: str, relation: str) -> bool:
    """Whether ``target`` is reachable from ``source`` through one relation."""

    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        if edge.relation == relation:
            adjacency.setdefault(edge.source, set()).add(edge.target)

    seen: set[str] = set()
    frontier = list(sorted(adjacency.get(source, ()), reverse=True))
    while frontier:
        current = frontier.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(
            neighbor
            for neighbor in sorted(adjacency.get(current, ()), reverse=True)
            if neighbor not in seen
        )
    return False


def apply_grounding_policy(
    graph: GraphState,
    schema: NetworkSchema,
) -> GroundingResult:
    """Apply ``schema.grounding_policy`` without mutating ``graph``.

    No configured policy is a true graph no-op.  With a policy, root and
    grounded node IDs are sorted before use, so the selected top root and all
    proposed edges are independent of input list ordering.  Every edge is
    checked for endpoint ambiguity, relation existence, self-loop permission,
    type-signature admissibility, duplication, and configured acyclicity.

    Hierarchy edges receive an additional transitive-reachability guard: a
    non-top root is not connected directly to the top when it already reaches
    the top through the hierarchy relation.  This keeps repeated grounding
    idempotent and avoids unnecessary shortcuts in an existing hierarchy.
    """

    policy = schema.grounding_policy
    if policy is None:
        return GroundingResult(
            graph=graph,
            skipped=(
                GroundingSkip(
                    phase="policy",
                    code="no_grounding_policy",
                    reason=f"schema {schema.id!r} does not configure grounding",
                ),
            ),
        )

    # Sets make selection independent of node declaration order.  Tracking all
    # types per ID lets us reject additions against an already malformed graph
    # instead of choosing one duplicate record by list position.
    types_by_id: dict[str, set[str]] = {}
    for node in graph.nodes:
        types_by_id.setdefault(node.id, set()).add(node.type)

    root_ids = sorted({node.id for node in graph.nodes if node.type == policy.root_object_type})
    if not root_ids:
        metadata = dict(graph.metadata)
        metadata["total_nodes"] = len(graph.nodes)
        metadata["total_edges"] = len(graph.edges)
        return GroundingResult(
            graph=GraphState(
                nodes=graph.nodes,
                edges=graph.edges,
                metadata=metadata,
            ),
            skipped=(
                GroundingSkip(
                    phase="policy",
                    code="no_root_nodes",
                    reason=(
                        f"graph has no nodes of grounding root type {policy.root_object_type!r}"
                    ),
                ),
            ),
        )

    top = root_ids[0]
    grounded_ids = sorted(
        {node.id for node in graph.nodes if node.type == policy.grounded_object_type}
    )
    edges = list(graph.edges)
    edge_keys = {(edge.source, edge.target, edge.relation) for edge in graph.edges}
    added: list[Edge] = []
    skipped: list[GroundingSkip] = []

    def skip(
        *,
        phase: GroundingPhase,
        code: str,
        reason: str,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        skipped.append(
            GroundingSkip(
                phase=phase,
                code=code,
                reason=reason,
                source=source,
                target=target,
                relation=relation,
            )
        )

    def try_add(
        *,
        phase: GroundingPhase,
        source: str,
        target: str,
        relation: str,
        skip_if_reachable: bool = False,
    ) -> None:
        source_types = types_by_id.get(source, set())
        target_types = types_by_id.get(target, set())
        if len(source_types) != 1 or len(target_types) != 1:
            skip(
                phase=phase,
                code="ambiguous_endpoint_type",
                reason="an endpoint ID is missing or is declared with multiple types",
                source=source,
                target=target,
                relation=relation,
            )
            return

        relation_spec = schema.relation_type_map.get(relation)
        if relation_spec is None:
            # Normally prevented when NetworkSchema validates its policy.  The
            # runtime guard keeps this function safe at deserialization edges.
            skip(
                phase=phase,
                code="invalid_relation",
                reason=f"relation {relation!r} is not declared by schema {schema.id!r}",
                source=source,
                target=target,
                relation=relation,
            )
            return

        if source == target and not relation_spec.allow_self_loops:
            skip(
                phase=phase,
                code="self_loop",
                reason=f"relation {relation!r} does not allow self-loops",
                source=source,
                target=target,
                relation=relation,
            )
            return

        source_type = next(iter(source_types))
        target_type = next(iter(target_types))
        if not schema.edge_type_ok(source_type, target_type, relation):
            skip(
                phase=phase,
                code="type_signature_violation",
                reason=(
                    f"({source_type!r}, {target_type!r}) is not admissible for "
                    f"relation {relation!r}"
                ),
                source=source,
                target=target,
                relation=relation,
            )
            return

        key = (source, target, relation)
        if key in edge_keys:
            skip(
                phase=phase,
                code="duplicate_edge",
                reason=f"edge {key!r} already exists",
                source=source,
                target=target,
                relation=relation,
            )
            return

        if skip_if_reachable and _reachable(edges, source, target, relation):
            skip(
                phase=phase,
                code="already_reachable",
                reason=(
                    f"{target!r} is already reachable from {source!r} through relation {relation!r}"
                ),
                source=source,
                target=target,
                relation=relation,
            )
            return

        candidate = Edge(source=source, target=target, relation=relation)
        if relation_spec.acyclic and relation_has_cycle(
            GraphState(nodes=graph.nodes, edges=tuple((*edges, candidate))),
            relation,
        ):
            skip(
                phase=phase,
                code="acyclic_relation_cycle",
                reason=f"edge {key!r} would leave relation {relation!r} cyclic",
                source=source,
                target=target,
                relation=relation,
            )
            return

        edges.append(candidate)
        edge_keys.add(key)
        added.append(candidate)

    for grounded_id in grounded_ids:
        try_add(
            phase="support",
            source=top,
            target=grounded_id,
            relation=policy.support_relation,
        )

    for root_id in root_ids[1:]:
        try_add(
            phase="hierarchy",
            source=root_id,
            target=top,
            relation=policy.hierarchy_relation,
            skip_if_reachable=True,
        )

    metadata = dict(graph.metadata)
    metadata["total_nodes"] = len(graph.nodes)
    metadata["total_edges"] = len(edges)
    grounded_graph = GraphState(
        nodes=graph.nodes,
        edges=tuple(edges),
        metadata=metadata,
    )
    return GroundingResult(
        graph=grounded_graph,
        top_root_id=top,
        added_edges=tuple(added),
        skipped=tuple(skipped),
    )


__all__ = [
    "GroundingResult",
    "GroundingSkip",
    "apply_grounding_policy",
]
