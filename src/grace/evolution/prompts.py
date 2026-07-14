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

"""Schema-rendered prompt builders for one GRACE Evolution step."""

from __future__ import annotations

import json
from typing import Any

from grace.initialization.prompts import PromptPair
from grace.schemas.models import NetworkSchema
from grace.schemas.prompt_renderer import render_schema


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def operation_planning(
    schema: NetworkSchema,
    graph: dict[str, Any],
    diagnosis_report: str,
) -> PromptPair:
    system = f"""<task>
Translate a diagnosis report into minimal graph-native policy edits. First locate graph
content that already addresses each finding. Propose an edit only for a genuine omission,
misstatement, or removal. Ground every operation in a diagnosis finding. This stage edits
nodes; consolidation through Merge is reserved for structural repair.
</task>
<network_schema>{render_schema(schema)}</network_schema>
<scope>
Allowed operations: AddNode, ModifyNode, RemoveNode. AddNode may attach configured edges.
Do not propose Merge, AddEdge, or RemoveEdge as a top-level operation here.
</scope>
<operation_contracts>
AddNode: {{"op":"AddNode","type":"...","content":"...","attach_edges":[{{"source":"NEW","target":"existing_id","relation":"..."}}]}}
ModifyNode: {{"op":"ModifyNode","node_id":"...","new_content":"..."}}
RemoveNode: {{"op":"RemoveNode","node_id":"..."}}
</operation_contracts>
<output_format>
Return JSON only as {{"operations":[...]}}. Every item must use the exact discriminator
field `op` (never `operation`), for example {{"op":"ModifyNode",...}}.
</output_format>"""
    user = f"""<graph>{_json(graph)}</graph>
<diagnosis_report>{diagnosis_report}</diagnosis_report>
Propose minimal node-level edits. Output JSON only."""
    return PromptPair(system=system, user=user)


def relation_maintenance(
    schema: NetworkSchema,
    graph: dict[str, Any],
    changed_node_ids: list[str],
) -> PromptPair:
    system = f"""<task>
Review relations incident to the changed nodes after node editing. Add a relation only when
the endpoint contents satisfy its configured meaning/signature; remove only a relation that
no longer holds. Preserve unrelated relations.
</task>
<network_schema>{render_schema(schema)}</network_schema>
<scope>Allowed operations: AddEdge and RemoveEdge only.</scope>
<operation_contracts>
AddEdge: {{"op":"AddEdge","source":"...","target":"...","relation":"..."}}
RemoveEdge: {{"op":"RemoveEdge","source":"...","target":"...","relation":"..."}}
</operation_contracts>
<output_format>
Return JSON only as {{"operations":[...]}}. Every item must use the exact discriminator
field `op` (never `operation`).
</output_format>"""
    user = f"""<graph>{_json(graph)}</graph>
<changed_node_ids>{_json(changed_node_ids)}</changed_node_ids>
Propose relation maintenance. Output JSON only."""
    return PromptPair(system=system, user=user)


def structural_analysis(
    schema: NetworkSchema,
    local_graph: dict[str, Any],
) -> PromptPair:
    system = f"""<task>
Detect every contradiction and redundancy between distinct nodes in the supplied local
typed graph. A contradiction has overlapping scope and no coherent joint reading. A
redundancy requires the same object type, overlapping scope, semantic subsumption, and no
configured structural reason to retain both. Detection only; propose no repair.
</task>
<network_schema>{render_schema(schema)}</network_schema>
<output_format>
Return JSON only:
{{"contradictions":[{{"u_id":"...","v_id":"...","reason":"..."}}],
 "redundancies":[{{"u_id":"...","v_id":"...","reason":"..."}}]}}
</output_format>"""
    user = f"""<local_graph>{_json(local_graph)}</local_graph>
Detect local contradictions and redundancies. Output JSON only."""
    return PromptPair(system=system, user=user)


def structural_repair(
    schema: NetworkSchema,
    local_graph: dict[str, Any],
    issues: dict[str, Any],
) -> PromptPair:
    system = f"""<task>
Resolve each listed local contradiction or redundancy with minimal schema-admissible graph
change. Use ModifyNode/remove operations to restore coherence and Merge or RemoveNode to
consolidate true redundancy. Each operation must reference a listed issue.
</task>
<network_schema>{render_schema(schema)}</network_schema>
<scope>
Allowed operations: ModifyNode, Merge, AddEdge, RemoveEdge, RemoveNode. Merge requires the
same object type and keeps u_id as representative.
</scope>
<operation_contracts>
ModifyNode: {{"op":"ModifyNode","node_id":"...","new_content":"..."}}
Merge: {{"op":"Merge","u_id":"...","v_id":"...","new_content":"..."}}
AddEdge/RemoveEdge: {{"op":"AddEdge","source":"...","target":"...","relation":"..."}}
RemoveNode: {{"op":"RemoveNode","node_id":"..."}}
</operation_contracts>
<output_format>
Return JSON only as {{"operations":[...]}}. Every item must use the exact discriminator
field `op` (never `operation`).
</output_format>"""
    user = f"""<local_graph>{_json(local_graph)}</local_graph>
<issues>{_json(issues)}</issues>
Propose minimal repairs. Output JSON only."""
    return PromptPair(system=system, user=user)


def patch_reconstruction(
    previous_instruction: str,
    node_changes: list[dict[str, Any]],
) -> PromptPair:
    system = """<task>
Emit localized operations that update a system instruction for the supplied validated
node-level changes. A deterministic applier copies every untouched character. Do not
reproduce or rewrite unaffected text.
</task>
<operations>
ReplaceSpan: {"op":"ReplaceSpan","anchor":"...","new_text":"..."}
InsertAfter: {"op":"InsertAfter","anchor":"...","new_text":"..."}
DeleteSpan: {"op":"DeleteSpan","anchor":"..."}
</operations>
<anchor_rules>
Every anchor is copied verbatim from the current instruction and appears exactly once.
Use the shortest unique real span. For an addition, anchor at the closest related existing
text. Never invent a heading or use paraphrased graph content as an anchor.
</anchor_rules>
<output_format>
Return valid JSON only as {"operations":[...]}. Every item must use the exact
discriminator field "op" (never "operation").
</output_format>"""
    user = f"""<system_instruction>{previous_instruction}</system_instruction>
<node_changes>{_json(node_changes)}</node_changes>
Produce minimal patch operations. Output JSON only."""
    return PromptPair(system=system, user=user)


__all__ = [
    "operation_planning",
    "patch_reconstruction",
    "relation_maintenance",
    "structural_analysis",
    "structural_repair",
]
