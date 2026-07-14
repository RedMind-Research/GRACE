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

import pytest
from pydantic import ValidationError

from grace.errors import ConfigurationError
from grace.schemas.default import DefaultGraceSchema
from grace.schemas.loader import load_schema, load_schema_data
from grace.schemas.models import NetworkSchema, ObjectType, RelationType
from grace.schemas.prompt_renderer import render_schema


def test_default_schema_matches_paper_signatures() -> None:
    schema = DefaultGraceSchema()

    assert schema.object_type_ids == ("identity", "norm", "knowledge")
    assert schema.relation_type_ids == ("supports", "refines", "sequence")
    assert set(schema.relation_type_map["supports"].allowed_pairs) == {
        ("identity", "norm"),
        ("knowledge", "norm"),
        ("knowledge", "knowledge"),
        ("norm", "norm"),
    }
    assert set(schema.relation_type_map["refines"].allowed_pairs) == {
        ("identity", "identity"),
        ("norm", "norm"),
        ("knowledge", "knowledge"),
    }
    assert schema.relation_type_map["sequence"].allowed_pairs == (("norm", "norm"),)
    assert schema.acyclic_relations == frozenset({"refines", "sequence"})


def test_schema_hash_captures_prompt_relevant_declaration_order() -> None:
    original = DefaultGraceSchema()
    reordered = NetworkSchema(
        id=original.id,
        version=original.version,
        description=original.description,
        object_types=tuple(reversed(original.object_types)),
        relation_types=tuple(reversed(original.relation_types)),
        grounding_policy=original.grounding_policy,
    )

    assert render_schema(reordered) != render_schema(original)
    assert reordered.canonical_json() != original.canonical_json()
    assert reordered.schema_hash != original.schema_hash


def test_custom_ontology_loads_without_default_schema_assumptions(tmp_path) -> None:
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(
        """
id: legal-review
version: '1'
description: A minimal legal review ontology.
object_types:
  - id: role
    description: The reviewing party and its mandate.
  - id: rule
    description: A requirement applied during review.
relation_types:
  - id: governs
    description: The source role is responsible for the target rule.
    allowed_pairs:
      - [role, rule]
""".strip(),
        encoding="utf-8",
    )

    schema = load_schema(schema_file)

    assert schema.id == "legal-review"
    assert schema.edge_type_ok("role", "rule", "governs")
    assert not schema.edge_type_ok("rule", "role", "governs")
    assert schema.grounding_policy is None
    assert schema.excluded_neighborhood_object_types == frozenset()


def test_schema_rejects_pairs_that_reference_unknown_types() -> None:
    with pytest.raises(ConfigurationError, match="unknown type pair"):
        load_schema_data(
            {
                "id": "invalid",
                "description": "Invalid fixture.",
                "object_types": [{"id": "known", "description": "Known type."}],
                "relation_types": [
                    {
                        "id": "bad",
                        "description": "Invalid relation.",
                        "allowed_pairs": [["known", "missing"]],
                    }
                ],
            }
        )


def test_relation_rejects_duplicate_pairs() -> None:
    with pytest.raises(ValidationError, match="duplicate allowed_pairs"):
        RelationType(
            id="same",
            description="Duplicate fixture.",
            allowed_pairs=(("a", "b"), ("a", "b")),
        )


def test_relation_cannot_be_acyclic_and_allow_self_loops() -> None:
    with pytest.raises(ValidationError, match="both acyclic and self-looping"):
        RelationType(
            id="contradiction",
            description="Contradictory fixture.",
            allowed_pairs=(("a", "a"),),
            acyclic=True,
            allow_self_loops=True,
        )


def test_prompt_renderer_uses_configured_type_definitions() -> None:
    schema = NetworkSchema(
        id="simple",
        description="Simple fixture.",
        object_types=(ObjectType(id="policy", description="A policy unit."),),
        relation_types=(
            RelationType(
                id="precedes",
                description="The source happens first.",
                allowed_pairs=(("policy", "policy"),),
                acyclic=True,
            ),
        ),
    )

    rendered = render_schema(schema)
    assert "- policy: A policy unit." in rendered
    assert "- precedes: The source happens first." in rendered
    assert "Admissible pairs: (policy, policy)." in rendered
    assert "Constraints: acyclic." in rendered
