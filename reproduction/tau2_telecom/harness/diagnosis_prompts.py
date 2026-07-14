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

"""Frozen, domain-general two-stage diagnosis prompts for reproduction v1."""

from __future__ import annotations

from .identity import canonical_sha256


PER_FAILURE_REFLECTION_SYSTEM = """\
<task>
You are given a failed agent trajectory and the system prompt the agent operated under. Produce a comprehensive per-failure diagnosis. Determine how the agent handled each required subtask, identify every system-prompt gap that contributed to the failure, and report each gap as one finding with a concrete instruction the system prompt should add or change. Work through the full trajectory and required subtasks using the analysis approach below, follow the rules, and return the output in the format below.
</task>

<input>
1. system_prompt: the system prompt the agent operated under.
2. trajectory: the complete trajectory, including the agent's actions, tool calls, observations, and user turns.
3. subtasks: the required subtasks for this task.
</input>

<analysis_approach>
- Trace the full trajectory in order, including user turns, agent actions, tool calls, observations, user-facing instructions, tool errors, and transfers.
- For each required subtask, decide whether the agent attempted it, completed it, left it unresolved, or left it unaddressed before failure.
- Mark a subtask as attempted when the trajectory contains an explicit action, tool call, user-facing instruction, or diagnostic step targeting that subtask's required outcome.
- For each unresolved or unaddressed subtask, identify what system-prompt guidance, procedure, knowledge, decision criterion, or capability description is missing or inadequate.
- Also identify any issue that affects multiple subtasks or the overall failure trajectory.
- Report all distinct gaps that contributed to the failure.
</analysis_approach>

<global_rules>
- Ground every finding in behavior observed in the trajectory.
- Cover every distinct system-prompt gap the trajectory reveals, through to the last failure-relevant gap.
- Prefer one finding per distinct missing or inadequate instruction.
- Include multiple affected subtasks in the same finding when one prompt gap explains several failures.
- Give each finding a concrete proposed instruction that could be added to or used to revise the system prompt.
- When the trajectory suggests that the system prompt lacks guidance for obtaining needed state, carrying out an allowed operation, choosing the next step, or verifying an outcome, describe the missing guidance or capability concretely in the proposed instruction.
</global_rules>

<output_format>
Return valid JSON only:
{
  "task_id": "<task identifier>",
  "subtasks_attempted": ["<required subtask the agent attempted through an action, tool call, user-facing instruction, or diagnostic step>"],
  "subtasks_missed": ["<required subtask with no meaningful attempt before failure>"],
  "findings": [
    {"finding_id": "<e.g. F1>", "affected_subtasks": ["<subtask affected by this gap>"], "description": "<2-3 sentences: what went wrong and what guidance, procedure, knowledge, decision criterion, or capability description is missing or inadequate>", "proposed_instruction": "<concrete instruction text to add or modify in the system prompt>"}
  ]
}
</output_format>"""


PER_FAILURE_REFLECTION_USER = """\
<system_prompt>
{system_prompt}
</system_prompt>

<task_id>
{task_id}
</task_id>

<required_subtasks>
{subtasks_list}
</required_subtasks>

<trajectory>
{trajectory}
</trajectory>

Analyze all gaps in this failed trajectory. For each required subtask that was unresolved or unaddressed, identify what guidance is missing or inadequate in the system prompt. Output valid JSON only."""


DIAGNOSIS_SYNTHESIS_SYSTEM = """\
<task>
You are given the per-failure analyses for one batch of failed trajectories. Produce a domain-general batch diagnosis for downstream prompt evolution. Consolidate the findings into themes, where each theme represents one underlying system-prompt gap and one coherent fix target. Follow the synthesis approach below, follow the rules, and return the output in the format below.
</task>

<input>
reflections: a JSON list of per-failure analyses, each containing findings that identify system-prompt gaps.
</input>

<synthesis_approach>
- Start from the input findings, and preserve every finding in the final diagnosis.
- Group findings into the same theme only when they point to the same missing or inadequate system-prompt instruction.
- Split findings into separate themes when they require different prompt instructions, even if they affect the same subtask or tool.
- Merge findings when their proposed instructions are equivalent, overlapping, or one clearly subsumes the other.
- Keep the most relevant proposed instruction from the supporting findings, preserving its concrete names and wording where possible.
- Keep each theme as specific as the gap it describes, naming the behavior, tool, action, or condition the system prompt should govern.
</synthesis_approach>

<theme_fields>
- frequency: the count of distinct failed trajectories that show this gap, counting each trajectory once. It records how widespread the gap is, and every theme remains equally required to be addressed.
- affected_subtask_types: the subtask types the gap touches, drawn from the findings.
- root_cause: the shared system-prompt gap, grounded in the findings that support the theme.
- recommended_actions: concrete, self-contained instructions that the next evolution step can add to or use to revise the system prompt.
</theme_fields>

<global_rules>
- Build every theme from findings that describe behavior observed in the trajectories.
- Place every finding in exactly one theme, and keep all findings.
- Represent each distinct system-prompt gap once.
- A theme drawn from a single trajectory is valid and equally required to be addressed.
- Let recommended_actions hold several instructions only when one coordinated fix requires them.
- Make each recommended action specific enough to be inserted into a system prompt without additional interpretation.
- When a supporting Phase 1 finding contains concrete tool names, action names, condition names, policy section names, or other named capabilities, preserve those names exactly as written in the theme root_cause or recommended_actions.
</global_rules>

<output_format>
Return valid JSON only:
{
  "themes": [
    {"theme_id": "<theme id>", "title": "<concise theme name, max 10 words>", "frequency": <distinct trajectory count>, "affected_subtask_types": ["<subtask type affected>"], "root_cause": "<2-3 sentences synthesizing the shared prompt gap and its evidence>", "recommended_actions": ["<specific instruction describing the guidance the system prompt should hold>"]}
  ],
  "summary": "<overall diagnosis in 2-3 sentences>"
}
</output_format>"""


DIAGNOSIS_SYNTHESIS_USER = """\
<reflections>
{num_reflections} per-failure analyses to consolidate:
{reflections_json}
</reflections>

Consolidate these Phase 1 findings into one structured batch diagnosis. Preserve every finding, cluster by underlying system-prompt gap, and output valid JSON only."""


DIAGNOSIS_PROMPT_SUITE_HASH = canonical_sha256(
    {
        "namespace": "grace.diagnosis-prompts.v1",
        "reflection_system": PER_FAILURE_REFLECTION_SYSTEM,
        "reflection_user": PER_FAILURE_REFLECTION_USER,
        "synthesis_system": DIAGNOSIS_SYNTHESIS_SYSTEM,
        "synthesis_user": DIAGNOSIS_SYNTHESIS_USER,
    }
)


__all__ = [
    "DIAGNOSIS_PROMPT_SUITE_HASH",
    "DIAGNOSIS_SYNTHESIS_SYSTEM",
    "DIAGNOSIS_SYNTHESIS_USER",
    "PER_FAILURE_REFLECTION_SYSTEM",
    "PER_FAILURE_REFLECTION_USER",
]
