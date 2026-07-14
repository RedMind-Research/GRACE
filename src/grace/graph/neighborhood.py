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

"""Schema-aware local neighborhoods for GRACE structural analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from grace.graph.models import Edge, GraphState
from grace.schemas.models import GroundingPolicy, NetworkSchema


def _policy(schema: NetworkSchema) -> GroundingPolicy | None:
    """Single access point for optional schema-specific grounding behavior."""

    return getattr(schema, "grounding_policy", None)


def _policy_value(policy: GroundingPolicy | Mapping[str, Any], field: str, default: Any) -> Any:
    if isinstance(policy, Mapping):
        return policy.get(field, default)
    return getattr(policy, field, default)


def is_grounding_backbone_edge(
    edge: Edge,
    node_types: Mapping[str, str],
    schema: NetworkSchema,
) -> bool:
    """Whether ``edge`` is a deterministic grounding-scaffold relation.

    This exact predicate is useful when converting a graph diff back into
    instruction language.  It recognizes both the primary anchor-to-grounded
    relation and the optional secondary-anchor-to-top-anchor relation, solely
    from the configured grounding policy.
    """

    policy = _policy(schema)
    if policy is None:
        return False

    root_type = str(_policy_value(policy, "root_object_type", ""))
    grounded_type = str(_policy_value(policy, "grounded_object_type", ""))
    support_relation = str(_policy_value(policy, "support_relation", ""))
    hierarchy_relation = str(_policy_value(policy, "hierarchy_relation", ""))
    source_type = node_types.get(edge.source)
    target_type = node_types.get(edge.target)

    primary = (
        bool(root_type)
        and edge.relation == support_relation
        and source_type == root_type
        and target_type == grounded_type
    )
    secondary = (
        bool(root_type)
        and edge.relation == hierarchy_relation
        and source_type == root_type
        and target_type == root_type
    )
    return primary or secondary


def is_excluded_neighborhood_edge(
    edge: Edge,
    node_types: Mapping[str, str],
    schema: NetworkSchema,
    *,
    exclude_grounding_backbone: bool | None = None,
) -> bool:
    """Whether schema policy excludes an edge from local-neighborhood traversal.

    The default ontology's policy excludes every edge incident to its anchor
    object type.  Reading that behavior from ``GroundingPolicy`` avoids making
    ``identity`` (or any relation name) a universal assumption for custom
    domains.  An explicit override is available for diagnostics and tests.
    """

    policy = _policy(schema)
    if policy is None:
        return False
    excluded_types = schema.excluded_neighborhood_object_types
    configured = bool(excluded_types)
    enabled = configured if exclude_grounding_backbone is None else bool(exclude_grounding_backbone)
    if not enabled:
        return False
    return bool(excluded_types) and (
        node_types.get(edge.source) in excluded_types
        or node_types.get(edge.target) in excluded_types
    )


def induced_neighborhood(
    graph: GraphState,
    anchors: Iterable[str],
    radius: int,
    schema: NetworkSchema,
    *,
    exclude_grounding_backbone: bool | None = None,
) -> GraphState:
    """Return the undirected ``radius``-hop induced subgraph around ``anchors``.

    Traversal treats admissible directed relations as undirected relatedness,
    then returns all retained edges whose endpoints both lie in the visited
    node set.  Edges excluded by schema grounding policy are omitted from both
    traversal and the induced result, preventing a universal grounding hub
    from turning a local structural-analysis scope into the full graph.

    Unknown anchor IDs are ignored.  Node and edge output order is canonical so
    equivalent input orderings produce byte-equivalent serialized subgraphs.
    """

    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TypeError("radius must be an integer")
    if radius < 0:
        raise ValueError("radius must be greater than or equal to zero")

    nodes_by_id = {node.id: node for node in graph.nodes}
    node_types = {node_id: node.type for node_id, node in nodes_by_id.items()}
    retained_edges = tuple(
        edge
        for edge in graph.edges
        if edge.source in nodes_by_id
        and edge.target in nodes_by_id
        and not is_excluded_neighborhood_edge(
            edge,
            node_types,
            schema,
            exclude_grounding_backbone=exclude_grounding_backbone,
        )
    )

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes_by_id}
    for edge in retained_edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    visited = {str(anchor) for anchor in anchors if str(anchor) in nodes_by_id}
    frontier = set(visited)
    for _ in range(radius):
        next_frontier: set[str] = set()
        for node_id in sorted(frontier):
            next_frontier.update(adjacency[node_id] - visited)
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier

    nodes = sorted(
        (nodes_by_id[node_id] for node_id in visited),
        key=lambda node: (node.id, node.canonical_json()),
    )
    edges = sorted(
        (edge for edge in retained_edges if edge.source in visited and edge.target in visited),
        key=lambda edge: (
            edge.source,
            edge.target,
            edge.relation,
            edge.canonical_json(),
        ),
    )
    return GraphState(nodes=tuple(nodes), edges=tuple(edges))


def local_subgraph(
    graph: GraphState,
    anchors: Iterable[str],
    k: int,
    schema: NetworkSchema,
    *,
    exclude_grounding_backbone: bool | None = None,
) -> GraphState:
    """Compatibility alias using the experimental implementation's name."""

    return induced_neighborhood(
        graph,
        anchors,
        k,
        schema,
        exclude_grounding_backbone=exclude_grounding_backbone,
    )


__all__ = [
    "induced_neighborhood",
    "is_excluded_neighborhood_edge",
    "is_grounding_backbone_edge",
    "local_subgraph",
]
