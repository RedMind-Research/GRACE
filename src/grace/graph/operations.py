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

"""Deterministic, schema-aware assembly for the GRACE editing algebra.

The assembler is deliberately independent of the paper's ontology.  Every
object type, relation signature, self-loop rule, and acyclicity constraint is
read from the supplied :class:`~grace.schemas.models.NetworkSchema`.

Operations are applied in order.  An invalid operation is rejected without
rolling back previously accepted operations, and every rejection retains its
input position and a stable machine-readable reason code for audit artifacts.
``AddNode.attach_edges`` follows the legacy GRACE binding rule: an endpoint
which is not already a node ID denotes the node just created by that operation.
"""

from __future__ import annotations

import heapq
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from grace.graph.models import Edge, GraphState, Node
from grace.schemas.models import NetworkSchema


# -- Public operation models -------------------------------------------------


class _OperationModel(BaseModel):
    """Base model which preserves optional LLM provenance fields for audit."""

    model_config = ConfigDict(extra="allow", frozen=True)


class EdgeAttachment(BaseModel):
    """An edge proposed together with a newly added node.

    Either endpoint may name a non-existing placeholder (commonly ``NEW``);
    the assembler binds such an endpoint to the fresh node ID.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    source: str
    target: str
    relation: str


class AddNode(_OperationModel):
    op: Literal["AddNode"] = "AddNode"
    type: str
    content: str
    attach_edges: tuple[EdgeAttachment, ...] = ()


class ModifyNode(_OperationModel):
    op: Literal["ModifyNode"] = "ModifyNode"
    node_id: str
    new_content: str


class RemoveNode(_OperationModel):
    op: Literal["RemoveNode"] = "RemoveNode"
    node_id: str


class AddEdge(_OperationModel):
    op: Literal["AddEdge"] = "AddEdge"
    source: str
    target: str
    relation: str


class RemoveEdge(_OperationModel):
    op: Literal["RemoveEdge"] = "RemoveEdge"
    source: str
    target: str
    relation: str


class Merge(_OperationModel):
    """Merge ``v_id`` into representative ``u_id`` (reserved for SA/SR)."""

    op: Literal["Merge"] = "Merge"
    u_id: str
    v_id: str
    new_content: str


Operation: TypeAlias = Annotated[
    Union[AddNode, ModifyNode, RemoveNode, AddEdge, RemoveEdge, Merge],
    Field(discriminator="op"),
]

_OPERATION_ADAPTER: TypeAdapter[Operation] = TypeAdapter(Operation)


# -- Public result/audit models ---------------------------------------------


class AppliedOperation(BaseModel):
    """Audit record for one accepted top-level operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_index: int = Field(ge=0)
    operation: dict[str, JsonValue]
    effect: dict[str, JsonValue] = Field(default_factory=dict)


class RejectedOperation(BaseModel):
    """Audit record for a rejected operation or nested attachment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_index: int = Field(ge=0)
    path: str
    operation: JsonValue
    code: str
    reason: str


class AssemblyResult(BaseModel):
    """The assembled graph plus deterministic application telemetry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph: GraphState
    touched_node_ids: tuple[str, ...] = ()
    applied_operations: tuple[AppliedOperation, ...] = ()
    rejected_operations: tuple[RejectedOperation, ...] = ()

    @property
    def applied_count(self) -> int:
        return len(self.applied_operations)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_operations)


# -- Deterministic helpers ---------------------------------------------------


_SEQUENTIAL_ID_RE = re.compile(r"^N(\d+)$")


def _fresh_id_factory(existing_ids: set[str]):
    """Yield stable fresh IDs in the legacy ``N###`` namespace.

    The starting point and zero-padding depend only on the set of existing
    IDs, never their input order.  IDs issued earlier in the same assembly are
    retained even if a later operation removes their nodes, avoiding reuse in
    one audit transaction.
    """

    matches = [match for node_id in existing_ids if (match := _SEQUENTIAL_ID_RE.fullmatch(node_id))]
    counter = max((int(match.group(1)) for match in matches), default=0)
    width = max((len(match.group(1)) for match in matches), default=3)
    width = max(3, width)
    issued = set(existing_ids)

    def fresh_id() -> str:
        nonlocal counter
        while True:
            counter += 1
            candidate = f"N{counter:0{width}d}"
            if candidate not in issued:
                issued.add(candidate)
                return candidate

    return fresh_id


def _json_safe(value: Any) -> JsonValue:
    """Best-effort conversion of malformed inputs into auditable JSON data."""

    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        try:
            json.dumps(value, allow_nan=False)
        except ValueError:
            return repr(value)
        return value
    return repr(value)


def _operation_data(operation: _OperationModel) -> dict[str, JsonValue]:
    payload = _json_safe(operation.model_dump(mode="json"))
    assert isinstance(payload, dict)  # model_dump always returns a mapping
    return payload


def _relation_has_cycle(
    node_ids: set[str],
    edges: Iterable[Edge],
    relation: str,
) -> bool:
    """Return whether one configured relation contains a directed cycle."""

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if edge.relation != relation:
            continue
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1

    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        node_id = heapq.heappop(ready)
        visited += 1
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    return visited != len(node_ids)


# -- Assembly ---------------------------------------------------------------


def assemble(
    graph: GraphState | Mapping[str, Any],
    operations: Iterable[Operation | Mapping[str, Any]],
    schema: NetworkSchema,
) -> AssemblyResult:
    """Apply GRACE editing operations while preserving ``schema`` invariants.

    The input graph is never mutated.  Operations are interpreted sequentially
    against accepted earlier edits.  Rejection is local to one operation (or
    one ``AddNode`` attachment), with the exception that ``Merge`` is tested
    atomically because vertex contraction can create a cycle.
    """

    typed_graph = graph if isinstance(graph, GraphState) else GraphState.model_validate(graph)
    nodes = list(typed_graph.nodes)
    edges = list(typed_graph.edges)
    node_index = {node.id: node for node in nodes}
    node_type = {node.id: node.type for node in nodes}
    edge_set = {(edge.source, edge.target, edge.relation) for edge in edges}

    object_types = set(schema.object_type_ids)
    relation_types = set(schema.relation_type_ids)
    touched: set[str] = set()
    applied: list[AppliedOperation] = []
    rejected: list[RejectedOperation] = []
    fresh_id = _fresh_id_factory(set(node_index))

    def reject(
        *,
        operation_index: int,
        path: str,
        operation: Any,
        code: str,
        reason: str,
    ) -> None:
        rejected.append(
            RejectedOperation(
                operation_index=operation_index,
                path=path,
                operation=_json_safe(operation),
                code=code,
                reason=reason,
            )
        )

    def try_add_edge(
        *,
        source: str,
        target: str,
        relation: str,
        operation_index: int,
        path: str,
        audit_operation: Any,
    ) -> bool:
        if source not in node_type or target not in node_type:
            reject(
                operation_index=operation_index,
                path=path,
                operation=audit_operation,
                code="dangling_endpoint",
                reason=f"edge endpoint does not exist ({source!r} -> {target!r})",
            )
            return False
        if relation not in relation_types:
            reject(
                operation_index=operation_index,
                path=path,
                operation=audit_operation,
                code="invalid_relation",
                reason=f"relation {relation!r} is not defined by schema {schema.id!r}",
            )
            return False
        if source == target and not schema.relation_allows_self_loop(relation):
            reject(
                operation_index=operation_index,
                path=path,
                operation=audit_operation,
                code="self_loop",
                reason=f"relation {relation!r} does not allow self-loops",
            )
            return False
        if not schema.edge_type_ok(node_type[source], node_type[target], relation):
            reject(
                operation_index=operation_index,
                path=path,
                operation=audit_operation,
                code="type_signature_violation",
                reason=(
                    f"({node_type[source]!r}, {node_type[target]!r}) is not admissible "
                    f"for relation {relation!r}"
                ),
            )
            return False

        key = (source, target, relation)
        if key in edge_set:
            reject(
                operation_index=operation_index,
                path=path,
                operation=audit_operation,
                code="duplicate_edge",
                reason=f"edge {key!r} already exists",
            )
            return False

        candidate = Edge(source=source, target=target, relation=relation)
        if schema.relation_is_acyclic(relation) and _relation_has_cycle(
            set(node_type), [*edges, candidate], relation
        ):
            reject(
                operation_index=operation_index,
                path=path,
                operation=audit_operation,
                code="acyclic_relation_cycle",
                reason=f"edge {key!r} would create a cycle in relation {relation!r}",
            )
            return False

        edges.append(candidate)
        edge_set.add(key)
        touched.update((source, target))
        return True

    for operation_index, raw_operation in enumerate(operations):
        path = f"operations[{operation_index}]"
        try:
            operation = (
                raw_operation
                if isinstance(raw_operation, _OperationModel)
                else _OPERATION_ADAPTER.validate_python(raw_operation)
            )
        except ValidationError as exc:
            reject(
                operation_index=operation_index,
                path=path,
                operation=raw_operation,
                code="invalid_operation",
                reason=str(exc),
            )
            continue

        operation_data = _operation_data(operation)

        if isinstance(operation, AddNode):
            if operation.type not in object_types:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="invalid_object_type",
                    reason=(
                        f"object type {operation.type!r} is not defined by schema {schema.id!r}"
                    ),
                )
                continue
            content = operation.content.strip()
            if not content:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="empty_content",
                    reason="AddNode content is empty",
                )
                continue

            node_id = fresh_id()
            node = Node(id=node_id, type=operation.type, content=content)
            nodes.append(node)
            node_index[node_id] = node
            node_type[node_id] = node.type
            touched.add(node_id)

            attached_count = 0
            rejected_before = len(rejected)
            for attachment_index, attachment in enumerate(operation.attach_edges):
                source = attachment.source if attachment.source in node_type else node_id
                target = attachment.target if attachment.target in node_type else node_id
                if try_add_edge(
                    source=source,
                    target=target,
                    relation=attachment.relation,
                    operation_index=operation_index,
                    path=f"{path}.attach_edges[{attachment_index}]",
                    audit_operation=attachment.model_dump(mode="json"),
                ):
                    attached_count += 1
            applied.append(
                AppliedOperation(
                    operation_index=operation_index,
                    operation=operation_data,
                    effect={
                        "generated_node_id": node_id,
                        "attached_edge_count": attached_count,
                        "rejected_attachment_count": len(rejected) - rejected_before,
                    },
                )
            )

        elif isinstance(operation, ModifyNode):
            target_node = node_index.get(operation.node_id)
            if target_node is None:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="unknown_node",
                    reason=f"node {operation.node_id!r} does not exist",
                )
                continue
            content = operation.new_content.strip()
            if not content:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="empty_content",
                    reason="ModifyNode new_content is empty",
                )
                continue
            replacement = target_node.model_copy(update={"content": content})
            nodes[nodes.index(target_node)] = replacement
            node_index[operation.node_id] = replacement
            touched.add(operation.node_id)
            applied.append(
                AppliedOperation(
                    operation_index=operation_index,
                    operation=operation_data,
                    effect={"node_id": operation.node_id},
                )
            )

        elif isinstance(operation, RemoveNode):
            target_node = node_index.get(operation.node_id)
            if target_node is None:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="unknown_node",
                    reason=f"node {operation.node_id!r} does not exist",
                )
                continue
            nodes.remove(target_node)
            removed_edge_count = sum(
                edge.source == operation.node_id or edge.target == operation.node_id
                for edge in edges
            )
            edges = [
                edge
                for edge in edges
                if edge.source != operation.node_id and edge.target != operation.node_id
            ]
            edge_set = {(edge.source, edge.target, edge.relation) for edge in edges}
            del node_index[operation.node_id]
            del node_type[operation.node_id]
            touched.add(operation.node_id)
            applied.append(
                AppliedOperation(
                    operation_index=operation_index,
                    operation=operation_data,
                    effect={
                        "node_id": operation.node_id,
                        "removed_edge_count": removed_edge_count,
                    },
                )
            )

        elif isinstance(operation, AddEdge):
            if try_add_edge(
                source=operation.source,
                target=operation.target,
                relation=operation.relation,
                operation_index=operation_index,
                path=path,
                audit_operation=operation_data,
            ):
                applied.append(
                    AppliedOperation(
                        operation_index=operation_index,
                        operation=operation_data,
                        effect={
                            "source": operation.source,
                            "target": operation.target,
                            "relation": operation.relation,
                        },
                    )
                )

        elif isinstance(operation, RemoveEdge):
            if operation.source not in node_type or operation.target not in node_type:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="dangling_endpoint",
                    reason=(
                        "edge endpoint does not exist "
                        f"({operation.source!r} -> {operation.target!r})"
                    ),
                )
                continue
            if operation.relation not in relation_types:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="invalid_relation",
                    reason=(
                        f"relation {operation.relation!r} is not defined by schema {schema.id!r}"
                    ),
                )
                continue
            key = (operation.source, operation.target, operation.relation)
            if key not in edge_set:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="edge_not_found",
                    reason=f"edge {key!r} does not exist",
                )
                continue
            edges = [edge for edge in edges if (edge.source, edge.target, edge.relation) != key]
            edge_set.discard(key)
            touched.update((operation.source, operation.target))
            applied.append(
                AppliedOperation(
                    operation_index=operation_index,
                    operation=operation_data,
                    effect={
                        "source": operation.source,
                        "target": operation.target,
                        "relation": operation.relation,
                    },
                )
            )

        elif isinstance(operation, Merge):
            representative = node_index.get(operation.u_id)
            absorbed = node_index.get(operation.v_id)
            if representative is None or absorbed is None:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="unknown_node",
                    reason=(
                        f"merge nodes must both exist ({operation.u_id!r}, {operation.v_id!r})"
                    ),
                )
                continue
            if operation.u_id == operation.v_id:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="same_merge_node",
                    reason="Merge requires two distinct node IDs",
                )
                continue
            if representative.type != absorbed.type:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="merge_type_mismatch",
                    reason=(
                        f"merge nodes have different types ({representative.type!r} != "
                        f"{absorbed.type!r})"
                    ),
                )
                continue
            content = operation.new_content.strip()
            if not content:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="empty_content",
                    reason="Merge new_content is empty",
                )
                continue

            merged_node = representative.model_copy(update={"content": content})
            tentative_nodes = [
                merged_node if node.id == operation.u_id else node
                for node in nodes
                if node.id != operation.v_id
            ]
            tentative_types = {node.id: node.type for node in tentative_nodes}
            tentative_edges: list[Edge] = []
            tentative_keys: set[tuple[str, str, str]] = set()
            collapsed_self_loops = 0
            collapsed_duplicates = 0
            invalid_reroute: tuple[str, str] | None = None

            for edge in edges:
                source = operation.u_id if edge.source == operation.v_id else edge.source
                target = operation.u_id if edge.target == operation.v_id else edge.target
                rerouted = source != edge.source or target != edge.target
                key = (source, target, edge.relation)
                if source == target:
                    collapsed_self_loops += 1
                    continue
                if key in tentative_keys:
                    collapsed_duplicates += 1
                    continue
                if rerouted and (
                    edge.relation not in relation_types
                    or source not in tentative_types
                    or target not in tentative_types
                    or not schema.edge_type_ok(
                        tentative_types[source], tentative_types[target], edge.relation
                    )
                ):
                    invalid_reroute = (
                        "merge_reroute_violation",
                        f"rerouted edge {key!r} violates the configured schema",
                    )
                    break
                tentative_keys.add(key)
                tentative_edges.append(
                    edge.model_copy(update={"source": source, "target": target})
                    if rerouted
                    else edge
                )

            if invalid_reroute is not None:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code=invalid_reroute[0],
                    reason=invalid_reroute[1],
                )
                continue

            cycle_relation = next(
                (
                    relation
                    for relation in sorted(schema.acyclic_relations)
                    if _relation_has_cycle(set(tentative_types), tentative_edges, relation)
                ),
                None,
            )
            if cycle_relation is not None:
                reject(
                    operation_index=operation_index,
                    path=path,
                    operation=operation_data,
                    code="acyclic_relation_cycle",
                    reason=(
                        f"merging the nodes would create a cycle in relation {cycle_relation!r}"
                    ),
                )
                continue

            nodes = tentative_nodes
            edges = tentative_edges
            edge_set = tentative_keys
            node_index[operation.u_id] = merged_node
            del node_index[operation.v_id]
            del node_type[operation.v_id]
            touched.update((operation.u_id, operation.v_id))
            applied.append(
                AppliedOperation(
                    operation_index=operation_index,
                    operation=operation_data,
                    effect={
                        "representative_node_id": operation.u_id,
                        "absorbed_node_id": operation.v_id,
                        "collapsed_self_loop_count": collapsed_self_loops,
                        "collapsed_duplicate_count": collapsed_duplicates,
                    },
                )
            )

    metadata = dict(typed_graph.metadata)
    metadata["total_nodes"] = len(nodes)
    metadata["total_edges"] = len(edges)
    assembled_graph = GraphState(nodes=tuple(nodes), edges=tuple(edges), metadata=metadata)
    return AssemblyResult(
        graph=assembled_graph,
        touched_node_ids=tuple(sorted(touched)),
        applied_operations=tuple(applied),
        rejected_operations=tuple(rejected),
    )


__all__ = [
    "AddEdge",
    "AddNode",
    "AppliedOperation",
    "AssemblyResult",
    "EdgeAttachment",
    "Merge",
    "ModifyNode",
    "Operation",
    "RejectedOperation",
    "RemoveEdge",
    "RemoveNode",
    "assemble",
]
