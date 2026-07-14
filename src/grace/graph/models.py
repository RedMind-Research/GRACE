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

"""Typed graph and externally persisted GRACE state models.

The models in this module intentionally know nothing about a concrete network
schema.  A schema implementation supplies its identifier and content hash when
constructing :class:`GraceState`; graph/schema conformance is checked by the
deterministic validation layer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from grace.errors import StateIntegrityError


STATE_FORMAT_VERSION = "1"


class _FrozenDict(dict[str, Any]):
    """A JSON mapping that cannot invalidate a frozen state's stored hashes."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("metadata is immutable")

    __delitem__ = _immutable  # type: ignore[assignment]
    __ior__ = _immutable  # type: ignore[assignment]
    __setitem__ = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable  # type: ignore[assignment]

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        memo[id(self)] = self
        return self


class _FrozenList(list[Any]):
    """A JSON list that retains normal serialization but rejects mutation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("metadata is immutable")

    __delitem__ = _immutable  # type: ignore[assignment]
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]
    __setitem__ = _immutable  # type: ignore[assignment]
    append = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    extend = _immutable  # type: ignore[assignment]
    insert = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    remove = _immutable  # type: ignore[assignment]
    reverse = _immutable  # type: ignore[assignment]
    sort = _immutable  # type: ignore[assignment]

    def __copy__(self) -> _FrozenList:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenList:
        memo[id(self)] = self
        return self


def _freeze_json(value: Any) -> Any:
    """Recursively freeze JSON containers and reject non-finite numbers."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metadata cannot contain NaN or infinite numbers")
    if isinstance(value, Mapping):
        return _FrozenDict({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def _frozen_metadata(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    frozen = _freeze_json(value)
    assert isinstance(frozen, dict)
    return frozen


def _canonical_json(value: JsonValue) -> str:
    """Return one stable, UTF-8-safe JSON representation for ``value``."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_json_value(value: Any) -> JsonValue:
    """Convert Pydantic models and containers to JSON-compatible data."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_json_value(item) for item in value]
    return value


class _CanonicalModel(BaseModel):
    """Shared deterministic serialization helpers for public state models."""

    # Validate default factories as well as caller-supplied values.  Without
    # this, an omitted ``metadata`` field bypasses the container-freezing field
    # validators on Node, Edge, and GraphState and leaves a mutable plain dict
    # inside an otherwise frozen, hash-bearing model.
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    def canonical_data(self) -> JsonValue:
        """Return the JSON-compatible data covered by ``canonical_json``."""

        return _as_json_value(self)

    def canonical_json(self) -> str:
        """Serialize without insignificant whitespace and with sorted keys."""

        return _canonical_json(self.canonical_data())

    def content_hash(self) -> str:
        """Return the SHA-256 digest of the canonical JSON representation."""

        return _sha256_text(self.canonical_json())

    def canonical_hash(self) -> str:
        """Alias for :meth:`content_hash` for callers naming the operation."""

        return self.content_hash()


class Node(_CanonicalModel):
    """A typed policy-graph object.

    ``type`` remains a string here because the admissible values belong to the
    configured network schema, not to this schema-independent state layer.
    """

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _freeze_metadata = field_validator("metadata", mode="after")(_frozen_metadata)


class Edge(_CanonicalModel):
    """A directed, typed relation between two graph nodes."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _freeze_metadata = field_validator("metadata", mode="after")(_frozen_metadata)


class GraphState(_CanonicalModel):
    """The schema-independent, typed policy graph at one GRACE step."""

    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _freeze_metadata = field_validator("metadata", mode="after")(_frozen_metadata)

    def canonical_data(self) -> JsonValue:
        """Return a canonical graph payload independent of list ordering.

        Nodes and edges are graph sets semantically.  Sorting their complete
        canonical records makes hashes stable even when a producer emits an
        equivalent graph in a different order.  Full-record tie breakers also
        keep malformed duplicate-ID inputs deterministic until schema
        validation reports them.
        """

        nodes = [node.canonical_data() for node in self.nodes]
        edges = [edge.canonical_data() for edge in self.edges]
        nodes.sort(key=_canonical_json)
        edges.sort(key=_canonical_json)
        return {
            "edges": edges,
            "metadata": _as_json_value(self.metadata),
            "nodes": nodes,
        }


class GraceState(_CanonicalModel):
    """An instruction and graph bound together as one integrity-checked state."""

    format_version: str = Field(default=STATE_FORMAT_VERSION, min_length=1)
    graph: GraphState
    instruction: str
    step: int = Field(ge=0)
    schema_id: str = Field(min_length=1)
    schema_hash: str = Field(min_length=1)
    instruction_hash: str = Field(min_length=1)
    graph_hash: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    parent_state_id: str | None = None

    _STATE_ID_FIELDS: ClassVar[tuple[str, ...]] = (
        "format_version",
        "graph_hash",
        "instruction_hash",
        "parent_state_id",
        "schema_hash",
        "schema_id",
        "step",
    )

    @model_validator(mode="after")
    def validate_loaded_state(self) -> GraceState:
        """Make integrity validation mandatory for every normal load path."""

        self.validate_integrity()
        return self

    @classmethod
    def from_parts(
        cls,
        *,
        graph: GraphState | Mapping[str, Any],
        instruction: str,
        schema_id: str,
        schema_hash: str,
        step: int = 0,
        parent_state_id: str | None = None,
        format_version: str = STATE_FORMAT_VERSION,
    ) -> GraceState:
        """Build a state and derive all content- and lineage-bound hashes.

        The instruction hash covers the instruction's exact UTF-8 text.  The
        graph hash covers :meth:`GraphState.canonical_json`.  ``state_id`` then
        binds those hashes to the schema, step, parent, and data-format version.
        """

        typed_graph = graph if isinstance(graph, GraphState) else GraphState.model_validate(graph)
        instruction_hash = _sha256_text(instruction)
        graph_hash = typed_graph.content_hash()
        state_id = cls._derive_state_id(
            format_version=format_version,
            graph_hash=graph_hash,
            instruction_hash=instruction_hash,
            parent_state_id=parent_state_id,
            schema_hash=schema_hash,
            schema_id=schema_id,
            step=step,
        )
        return cls(
            format_version=format_version,
            graph=typed_graph,
            instruction=instruction,
            step=step,
            schema_id=schema_id,
            schema_hash=schema_hash,
            instruction_hash=instruction_hash,
            graph_hash=graph_hash,
            state_id=state_id,
            parent_state_id=parent_state_id,
        )

    @classmethod
    def _derive_state_id(cls, **parts: JsonValue) -> str:
        payload = {field: parts[field] for field in cls._STATE_ID_FIELDS}
        return _sha256_text(_canonical_json(payload))

    def validate_integrity(
        self,
        *,
        schema_id: str | None = None,
        schema_hash: str | None = None,
        parent_state: GraceState | None = None,
    ) -> bool:
        """Validate stored hashes, expected schema identity, and basic lineage.

        A concrete schema object is deliberately not accepted here.  Callers
        that have loaded a schema snapshot can pass its expected ID and hash.
        When ``parent_state`` is supplied, this also verifies a one-step,
        same-schema lineage link.  Integrity failures raise
        :class:`~grace.errors.StateIntegrityError`; successful validation
        returns ``True`` for convenient guards and assertions.
        """

        errors: list[str] = []

        if not self.instruction.strip():
            errors.append("instruction is empty")
        if not self.schema_id.strip():
            errors.append("schema_id is empty")
        if not self.schema_hash.strip():
            errors.append("schema_hash is empty")

        actual_instruction_hash = _sha256_text(self.instruction)
        actual_graph_hash = self.graph.content_hash()
        if self.instruction_hash != actual_instruction_hash:
            errors.append("instruction_hash does not match instruction")
        if self.graph_hash != actual_graph_hash:
            errors.append("graph_hash does not match graph")

        expected_state_id = self._derive_state_id(
            format_version=self.format_version,
            graph_hash=actual_graph_hash,
            instruction_hash=actual_instruction_hash,
            parent_state_id=self.parent_state_id,
            schema_hash=self.schema_hash,
            schema_id=self.schema_id,
            step=self.step,
        )
        if self.state_id != expected_state_id:
            errors.append("state_id does not match state content and lineage")

        if schema_id is not None and self.schema_id != schema_id:
            errors.append(f"schema_id mismatch: expected {schema_id!r}, found {self.schema_id!r}")
        if schema_hash is not None and self.schema_hash != schema_hash:
            errors.append("schema_hash does not match the expected schema")

        if self.step == 0 and self.parent_state_id is not None:
            errors.append("step 0 must not have a parent_state_id")
        if self.step > 0 and not self.parent_state_id:
            errors.append("a state after step 0 must have a parent_state_id")
        if self.parent_state_id == self.state_id:
            errors.append("a state cannot be its own parent")

        if parent_state is not None:
            if self.parent_state_id != parent_state.state_id:
                errors.append("parent_state_id does not match the supplied parent state")
            if self.step != parent_state.step + 1:
                errors.append("step is not exactly one greater than the parent step")
            if self.schema_id != parent_state.schema_id:
                errors.append("schema_id differs from the parent state")
            if self.schema_hash != parent_state.schema_hash:
                errors.append("schema_hash differs from the parent state")

        if errors:
            raise StateIntegrityError("; ".join(errors))
        return True


__all__ = [
    "STATE_FORMAT_VERSION",
    "Edge",
    "GraceState",
    "GraphState",
    "Node",
]
