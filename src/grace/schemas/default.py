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

"""The GRACE network schema defined in the paper."""

from grace.schemas.models import GroundingPolicy, NetworkSchema, ObjectType, RelationType


class DefaultGraceSchema(NetworkSchema):
    """Exact default ontology and type signatures used by the GRACE method."""

    id: str = "grace-default"
    version: str = "1"
    description: str = "The identity, norm, and knowledge schema defined by GRACE."
    object_types: tuple[ObjectType, ...] = (
        ObjectType(
            id="identity",
            description=(
                "Identity constitutes the agent as the subject of action: its content "
                "fixes the agent's standing and the mandate under which it acts, "
                "establishing the source to which its conduct is attributed."
            ),
        ),
        ObjectType(
            id="norm",
            description=(
                "A norm governs the agent's conduct: its content is a standard the "
                "agent's action must conform to, so that any action either satisfies "
                "or breaches it."
            ),
        ),
        ObjectType(
            id="knowledge",
            description=(
                "Knowledge represents the world the agent acts upon: its content "
                "describes a self-contained piece of the task domain that the agent "
                "reasons from."
            ),
        ),
    )
    relation_types: tuple[RelationType, ...] = (
        RelationType(
            id="supports",
            description=(
                "A relation of grounding. The target content rests on the source "
                "content as its basis, and the edge runs from the grounding content "
                "to the content it grounds."
            ),
            allowed_pairs=(
                ("identity", "norm"),
                ("knowledge", "norm"),
                ("knowledge", "knowledge"),
                ("norm", "norm"),
            ),
        ),
        RelationType(
            id="refines",
            description=(
                "A relation of specialization between contents of the same kind. The "
                "source is the target under a narrower scope, and the edge runs from "
                "the more specific content to the more general content."
            ),
            allowed_pairs=(
                ("identity", "identity"),
                ("norm", "norm"),
                ("knowledge", "knowledge"),
            ),
            acyclic=True,
        ),
        RelationType(
            id="sequence",
            description=(
                "A relation of procedural order over norms. The source content is "
                "carried out before the target content in execution."
            ),
            allowed_pairs=(("norm", "norm"),),
            acyclic=True,
        ),
    )
    grounding_policy: GroundingPolicy | None = GroundingPolicy(
        root_object_type="identity",
        grounded_object_type="norm",
        support_relation="supports",
        hierarchy_relation="refines",
        exclude_incident_object_types_from_neighborhood=("identity",),
    )


__all__ = ["DefaultGraceSchema"]
