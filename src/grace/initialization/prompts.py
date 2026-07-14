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

"""Schema-rendered Prompt-to-Graph and fidelity prompt builders.

The ontology section is rendered from the supplied ``NetworkSchema`` while
preserving the method's P2G roles: faithful decomposition, full-graph fidelity
detection, and targeted repair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from grace.schemas.models import NetworkSchema
from grace.schemas.prompt_renderer import render_schema


@dataclass(frozen=True)
class PromptPair:
    """One system/user prompt pair sent to a provider."""

    system: str
    user: str


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _type_choices(schema: NetworkSchema) -> str:
    return "|".join(schema.object_type_ids)


def _relation_choices(schema: NetworkSchema) -> str:
    return "|".join(schema.relation_type_ids)


def prompt_to_graph(schema: NetworkSchema, instruction: str) -> PromptPair:
    """Build the faithful initial-decomposition request."""

    system = f"""<task>
Build a typed graph G=(V,E) that completely transcribes the supplied system-level
instruction. Decompose the instruction into atomic units. Assign each unit the object
type whose configured definition it satisfies. Add a directed relation only when the
contents satisfy the configured relation definition and type signature. Preserve all
instruction content: omit nothing and invent nothing.
</task>

<network_schema>
{render_schema(schema)}
</network_schema>

<atomicity>
Each node is one independently checkable instruction unit. Split multi-step procedures,
exceptions, conditional branches, and enumerations into atomic units; do not fragment one
atomic unit. Use relations to retain grounding, specialization, and order represented by
the source instruction.
</atomicity>

<output_format>
Return valid JSON only:
{{"nodes":[{{"id":"N001","type":"{_type_choices(schema)}","content":"..."}}],
 "edges":[{{"source":"N001","target":"N002","relation":"{_relation_choices(schema)}"}}]}}
Do not include commentary or any other top-level field.
</output_format>"""
    user = f"""<system_instruction>
{instruction}
</system_instruction>

Decompose this instruction into the configured typed graph. Output JSON only."""
    return PromptPair(system=system, user=user)


def fidelity_analysis(
    schema: NetworkSchema,
    instruction: str,
    graph: dict[str, Any],
) -> PromptPair:
    """Build the full-graph fidelity-detection request."""

    system = f"""<task>
Compare an original system-level instruction with a typed graph claimed to transcribe it.
Report concrete mismatches only: missing instruction units, content that adds/removes/
distorts meaning, incorrect object typing, incorrect represented relations, or clearly
implied configured relations that are absent. Do not flag equivalent wording or optional
relations not clearly implied by the source.
</task>

<network_schema>
{render_schema(schema)}
</network_schema>

<output_format>
Return valid JSON only with exactly these lists:
{{"missing_node":["source unit"],
 "unfaithful":[{{"node_id":"N001","issue":"..."}}],
 "mistyped":[{{"node_id":"N001","should_be":"{_type_choices(schema)}","reason":"..."}}],
 "wrong_relation":[{{"source":"N001","target":"N002","should_be":"{_relation_choices(schema)}","reason":"..."}}],
 "missing_relation":[{{"source":"N001","target":"N002","relation":"{_relation_choices(schema)}","reason":"..."}}]}}
Use empty lists when no issue exists.
</output_format>"""
    user = f"""<system_instruction>
{instruction}
</system_instruction>

<graph>
{_json(graph)}
</graph>

Judge fidelity and completeness. Output JSON only."""
    return PromptPair(system=system, user=user)


def fidelity_repair(
    schema: NetworkSchema,
    instruction: str,
    graph: dict[str, Any],
    issues: dict[str, Any],
    rejected_operations: list[dict[str, Any]],
) -> PromptPair:
    """Build the targeted fidelity/schema repair request."""

    system = f"""<task>
Repair the current typed graph so it faithfully and completely transcribes the original
instruction. Address only the listed schema/fidelity issues with minimal targeted edits;
leave correct graph content unchanged. Never repeat an operation already rejected by the
deterministic assembler.
</task>

<network_schema>
{render_schema(schema)}
</network_schema>

<editing_algebra>
Available operations are AddNode, ModifyNode, RemoveNode, AddEdge, and RemoveEdge.
ModifyNode changes content only. Correct a mistyped node by replacing it. Every operation
must satisfy the configured schema; if no faithful admissible fix exists, leave that issue
unresolved rather than inventing content.
</editing_algebra>

<output_format>
Return valid JSON only as {{"operations":[...]}}. Each operation uses the public GRACE
operation fields and may include an audit-only "fixes" field.
</output_format>"""
    user = f"""<system_instruction>
{instruction}
</system_instruction>
<graph>{_json(graph)}</graph>
<issues>{_json(issues)}</issues>
<rejected_operations>{_json(rejected_operations)}</rejected_operations>

Propose minimal admissible repairs. Output JSON only."""
    return PromptPair(system=system, user=user)


__all__ = [
    "PromptPair",
    "fidelity_analysis",
    "fidelity_repair",
    "prompt_to_graph",
]
