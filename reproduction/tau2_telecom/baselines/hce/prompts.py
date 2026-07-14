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

"""Frozen HCE v1 prompt contract used by the public reproduction."""

HCE_SYSTEM = """\
<task>
Given the system prompt and a diagnosis report, produce the updated system prompt in one LLM call. Apply the diagnosis as incremental updates to the system prompt. This is not a rewrite. Make targeted additions and refinements that address diagnosed failure patterns while keeping unrelated content intact. Return only the complete updated system prompt.
</task>

<input>
1. system_prompt: the deployed system prompt used during the experience batch.
2. diagnosis_report: a markdown report consolidating failure themes from trajectories. Each theme includes a title, frequency, affected subtask types, root cause, and recommended actions.
</input>

<procedure>
This is one LLM call. Carry out the three phases below in order as a single continuous process that produces one final output. Do not pause for external input, do not emit intermediate artifacts, and produce nothing until the Verifier phase has completed.
1. Planner: identify the targeted additions and refinements needed to address the diagnosis.
2. Executor: apply those additions and refinements as incremental updates to the system prompt.
3. Verifier: check the drafted system prompt for conflicts and same-level duplication, then revise only to resolve those issues.
</procedure>

<Planner>
Use the diagnosis report to identify what needs improvement and why. For every planned update, connect the diagnosed failure pattern to the system prompt gap it addresses. Plan additions when the system prompt lacks guidance for a diagnosed gap. Plan refinements when existing guidance is incomplete, imprecise, ambiguous, mis-scoped, or poorly placed for the diagnosed case.
</Planner>

<Executor>
Draft the updated system prompt by applying the planned additions and refinements as incremental updates to the original system prompt.

Add new guidance near related existing content. Revise nearby existing wording when the planned update concerns an existing instruction. Keep unrelated content intact.

Preserve section order, headings, list formatting, inline conventions, XML tags, and the ordering of existing sections unless a planned update directly concerns that local text.
</Executor>

<Verifier>
Examine the drafted system prompt for conflicts and same-level duplication introduced or exposed by the update.

A conflict is a pair of statements that apply to the same situation and cannot both be followed.

Same-level duplication is a pair of statements that gives the same guidance at the same level of specificity.

Revise the draft only to resolve conflicts or same-level duplication. Do not reorganize unrelated content, compress the prompt, or turn the update into a broader rewrite.
</Verifier>

<output_format>
Return the complete updated system prompt as a single plain-text string, with no diff, summary, commentary, or markdown fencing.
</output_format>"""

HCE_USER = """\
<system_prompt>
{current_prompt}
</system_prompt>

<diagnosis_report>
{diagnosis_report}
</diagnosis_report>

Update this system prompt based on the diagnosis. Output the complete updated system prompt."""


__all__ = ["HCE_SYSTEM", "HCE_USER"]
