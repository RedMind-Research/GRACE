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

"""Render a configured ontology into reusable prompt sections."""

from grace.schemas.models import NetworkSchema


def render_object_types(schema: NetworkSchema) -> str:
    lines = ["<object_types A>"]
    for object_type in schema.object_types:
        lines.append(f"- {object_type.id}: {object_type.description}")
    lines.append("</object_types>")
    return "\n".join(lines)


def render_relation_types(schema: NetworkSchema) -> str:
    lines = [
        "<relation_types R and type signatures>",
        "Notation: each pair (a, b) is a directed edge from type a to type b.",
    ]
    for relation in schema.relation_types:
        pairs = ", ".join(f"({source}, {target})" for source, target in relation.allowed_pairs)
        constraints: list[str] = []
        if relation.acyclic:
            constraints.append("acyclic")
        if relation.allow_self_loops:
            constraints.append("self-loops allowed")
        suffix = f" Constraints: {', '.join(constraints)}." if constraints else ""
        lines.append(f"- {relation.id}: {relation.description} Admissible pairs: {pairs}.{suffix}")
    lines.append("</relation_types>")
    return "\n".join(lines)


def render_schema(schema: NetworkSchema) -> str:
    return f"{render_object_types(schema)}\n\n{render_relation_types(schema)}"


__all__ = ["render_object_types", "render_relation_types", "render_schema"]
