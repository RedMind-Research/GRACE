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

"""Typed, deterministic network-schema models for GRACE."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator


TypePair = tuple[str, str]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class ObjectType(BaseModel):
    """One ontology-level object type in :math:`A`."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RelationType(BaseModel):
    """One directed relation type and its admissible type signature."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    allowed_pairs: tuple[TypePair, ...] = Field(min_length=1)
    acyclic: bool = False
    allow_self_loops: bool = False

    @model_validator(mode="after")
    def validate_pairs(self) -> RelationType:
        if any(not source or not target for source, target in self.allowed_pairs):
            raise ValueError("allowed_pairs cannot contain an empty type id")
        if len(set(self.allowed_pairs)) != len(self.allowed_pairs):
            raise ValueError(f"relation {self.id!r} contains duplicate allowed_pairs")
        if self.acyclic and self.allow_self_loops:
            raise ValueError(f"relation {self.id!r} cannot be both acyclic and self-looping")
        return self


class GroundingPolicy(BaseModel):
    """Optional schema-specific policy for the default grounding backbone.

    GRACE's graph engine is domain/schema neutral.  Schemas that want the
    paper's identity-root convention opt in through this policy instead
    of relying on hard-coded ``identity``/``norm`` strings in generic code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    root_object_type: str = Field(min_length=1)
    grounded_object_type: str = Field(min_length=1)
    support_relation: str = Field(min_length=1)
    hierarchy_relation: str = Field(min_length=1)
    exclude_incident_object_types_from_neighborhood: tuple[str, ...] = ()


class NetworkSchema(BaseModel):
    """High-level ontology :math:`T_G=(A,R,{sigma_r})` used by GRACE."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    version: str = Field(default="1", min_length=1)
    description: str = Field(min_length=1)
    object_types: tuple[ObjectType, ...] = Field(min_length=1)
    relation_types: tuple[RelationType, ...] = Field(min_length=1)
    grounding_policy: GroundingPolicy | None = None

    @model_validator(mode="after")
    def validate_references(self) -> NetworkSchema:
        object_ids = [item.id for item in self.object_types]
        relation_ids = [item.id for item in self.relation_types]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("object type ids must be unique")
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("relation type ids must be unique")

        known_objects = set(object_ids)
        for relation in self.relation_types:
            for source, target in relation.allowed_pairs:
                if source not in known_objects or target not in known_objects:
                    raise ValueError(
                        f"relation {relation.id!r} references unknown type pair "
                        f"({source!r}, {target!r})"
                    )

        policy = self.grounding_policy
        if policy is not None:
            for field_name in (
                "root_object_type",
                "grounded_object_type",
            ):
                value = getattr(policy, field_name)
                if value not in known_objects:
                    raise ValueError(
                        f"grounding_policy.{field_name} references unknown object type {value!r}"
                    )
            excluded = set(policy.exclude_incident_object_types_from_neighborhood)
            unknown_excluded = excluded - known_objects
            if unknown_excluded:
                raise ValueError(
                    "grounding policy excludes unknown object types: "
                    + ", ".join(sorted(unknown_excluded))
                )

            relation_map = {item.id: item for item in self.relation_types}
            if policy.support_relation not in relation_map:
                raise ValueError("grounding_policy.support_relation is unknown")
            if policy.hierarchy_relation not in relation_map:
                raise ValueError("grounding_policy.hierarchy_relation is unknown")

            support_pair = (policy.root_object_type, policy.grounded_object_type)
            if support_pair not in set(relation_map[policy.support_relation].allowed_pairs):
                raise ValueError(
                    "grounding support relation does not admit the configured root/target pair"
                )
            hierarchy_pair = (policy.root_object_type, policy.root_object_type)
            if hierarchy_pair not in set(relation_map[policy.hierarchy_relation].allowed_pairs):
                raise ValueError("grounding hierarchy relation does not admit root-to-root edges")

        return self

    @property
    def object_type_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.object_types)

    @property
    def relation_type_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.relation_types)

    @property
    def object_type_map(self) -> dict[str, ObjectType]:
        return {item.id: item for item in self.object_types}

    @property
    def relation_type_map(self) -> dict[str, RelationType]:
        return {item.id: item for item in self.relation_types}

    @property
    def acyclic_relations(self) -> frozenset[str]:
        return frozenset(item.id for item in self.relation_types if item.acyclic)

    @property
    def excluded_neighborhood_object_types(self) -> frozenset[str]:
        if self.grounding_policy is None:
            return frozenset()
        return frozenset(self.grounding_policy.exclude_incident_object_types_from_neighborhood)

    def edge_type_ok(self, source_type: str, target_type: str, relation: str) -> bool:
        relation_spec = self.relation_type_map.get(relation)
        return bool(
            relation_spec and (source_type, target_type) in set(relation_spec.allowed_pairs)
        )

    def relation_is_acyclic(self, relation: str) -> bool:
        relation_spec = self.relation_type_map.get(relation)
        return bool(relation_spec and relation_spec.acyclic)

    def relation_allows_self_loop(self, relation: str) -> bool:
        relation_spec = self.relation_type_map.get(relation)
        return bool(relation_spec and relation_spec.allow_self_loops)

    def canonical_data(self) -> dict[str, object]:
        """Return the exact operational schema snapshot covered by its hash.

        Declaration order is intentionally retained because it is also used by
        prompt rendering.  Two snapshots that can produce different prompts
        therefore cannot share a schema hash.
        """

        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_data())

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def schema_hash(self) -> str:
        return self.content_hash()


__all__ = [
    "GroundingPolicy",
    "NetworkSchema",
    "ObjectType",
    "RelationType",
    "TypePair",
]
