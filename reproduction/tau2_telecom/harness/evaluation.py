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

"""Three-episode final-policy evaluation smoke without publication metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .identity import validate_sha256
from .models import ExpectedMatrix, TaskSelectionManifest
from .tau2_interface import (
    EpisodeArtifactAuthority,
    EpisodeContractError,
    EpisodeExecutor,
    EpisodeHarnessConfig,
    EpisodeMatrixExecution,
    execute_matrix_sequential,
)


NonEmpty = Annotated[str, Field(min_length=1)]
EVAL_SMOKE_SLOTS = ("eval-s020", "eval-s043", "eval-s060")
EVAL_SMOKE_TASK_IDS = (
    "[mms_issue]break_apn_mms_setting[PERSONA:Hard]",
    "[mobile_data_issue]airplane_mode_on|data_saver_mode_on[PERSONA:Easy]",
    "[service_issue]airplane_mode_on|break_apn_settings|unseat_sim_card[PERSONA:None]",
)


class EvalEpisodeSummary(BaseModel):
    """One accepted behavioral outcome; reward zero is valid completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: NonEmpty
    attempt_id: NonEmpty
    task_uid: NonEmpty
    task_definition_hash: NonEmpty
    benchmark_task_id: NonEmpty
    split_slot: NonEmpty
    trial: Literal[0] = 0
    protocol_seed: int
    status: Literal["completed"] = "completed"
    reward: float = Field(ge=0.0, le=1.0)
    termination_reason: NonEmpty

    @model_validator(mode="after")
    def validate_identifiers(self) -> EvalEpisodeSummary:
        validate_sha256(self.episode_id, field="episode_id")
        validate_sha256(self.attempt_id, field="attempt_id")
        validate_sha256(self.task_uid, field="task_uid")
        validate_sha256(self.task_definition_hash, field="task_definition_hash")
        return self


class EvalSmokeSummary(BaseModel):
    """Execution-only schema that structurally cannot contain pass metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["grace.eval-smoke-summary.v1"] = "grace.eval-smoke-summary.v1"
    scope: Literal["smoke/non-comparable"] = "smoke/non-comparable"
    status: Literal["complete", "incomplete"]
    matrix_hash: NonEmpty
    expected_count: Literal[3] = 3
    accepted_count: int = Field(ge=0, le=3)
    missing_episode_ids: tuple[str, ...] = ()
    episodes: tuple[EvalEpisodeSummary, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> EvalSmokeSummary:
        validate_sha256(self.matrix_hash, field="matrix_hash")
        for episode_id in self.missing_episode_ids:
            validate_sha256(episode_id, field="missing_episode_id")
        if len(self.missing_episode_ids) != len(set(self.missing_episode_ids)):
            raise ValueError("smoke missing episode IDs must be unique")
        observed_episode_ids = {item.episode_id for item in self.episodes}
        if observed_episode_ids.intersection(self.missing_episode_ids):
            raise ValueError("smoke episode cannot be both accepted and missing")
        if self.accepted_count != len(self.episodes):
            raise ValueError("accepted_count must match smoke episode records")
        complete = self.accepted_count == 3 and not self.missing_episode_ids
        if (self.status == "complete") != complete:
            raise ValueError("smoke status must match exact three-episode completion")
        if len(self.missing_episode_ids) != 3 - self.accepted_count:
            raise ValueError("smoke missing count must complement accepted episodes")
        slots = tuple(item.split_slot for item in self.episodes)
        expected_subsequence = tuple(slot for slot in EVAL_SMOKE_SLOTS if slot in slots)
        if slots != expected_subsequence or len(slots) != len(set(slots)):
            raise ValueError("smoke episode records must retain frozen canonical order")
        expected_tasks = dict(zip(EVAL_SMOKE_SLOTS, EVAL_SMOKE_TASK_IDS, strict=True))
        if any(item.benchmark_task_id != expected_tasks[item.split_slot] for item in self.episodes):
            raise ValueError("smoke episode task does not match its frozen slot")
        return self


class EvaluationSmokeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    execution: EpisodeMatrixExecution
    summary: EvalSmokeSummary

    @model_validator(mode="after")
    def validate_status(self) -> EvaluationSmokeResult:
        if self.execution.status != self.summary.status:
            raise ValueError("evaluation execution and smoke summary status differ")
        return self


def _validate_eval_smoke_scope(
    manifest: TaskSelectionManifest,
    expected: ExpectedMatrix,
    config: EpisodeHarnessConfig,
) -> None:
    if manifest.benchmark_revision != config.benchmark_revision:
        raise EpisodeContractError("manifest and execution tau2 revisions differ")
    actual_selection = tuple((task.split_slot, task.benchmark_task_id) for task in manifest.tasks)
    frozen_selection = tuple(zip(EVAL_SMOKE_SLOTS, EVAL_SMOKE_TASK_IDS, strict=True))
    if actual_selection != frozen_selection:
        raise EpisodeContractError("v1 eval smoke is fixed to EVAL20, EVAL43, and EVAL60")
    expected_shape = tuple(
        (episode.split_slot, episode.benchmark_task_id, episode.trial)
        for episode in expected.episodes
    )
    manifest_shape = tuple((task.split_slot, task.benchmark_task_id, 0) for task in manifest.tasks)
    if expected_shape != manifest_shape:
        raise EpisodeContractError(
            "eval smoke matrix must contain exactly the three manifest cells at trial zero"
        )


def run_evaluation_smoke(
    *,
    manifest: TaskSelectionManifest,
    expected: ExpectedMatrix,
    tasks: Sequence[Any],
    policy: str,
    prompt_hash: str,
    config: EpisodeHarnessConfig,
    executor: EpisodeExecutor,
    authority: EpisodeArtifactAuthority,
) -> EvaluationSmokeResult:
    """Execute exactly three final-policy cells and report no aggregate metric."""

    _validate_eval_smoke_scope(manifest, expected, config)
    execution = execute_matrix_sequential(
        expected=expected,
        tasks=tasks,
        policy=policy,
        prompt_hash=prompt_hash,
        config=config,
        executor=executor,
        authority=authority,
    )
    episodes = tuple(
        EvalEpisodeSummary(
            episode_id=outcome.contract.episode_id,
            attempt_id=outcome.contract.attempt_id,
            task_uid=outcome.contract.task_uid,
            task_definition_hash=outcome.contract.task_definition_hash,
            benchmark_task_id=outcome.contract.benchmark_task_id,
            split_slot=outcome.contract.split_slot,
            trial=0,
            protocol_seed=outcome.contract.protocol_seed,
            status="completed",
            reward=outcome.reward,
            termination_reason=outcome.termination_reason,
        )
        for outcome in execution.accepted
        if outcome.reward is not None and outcome.disposition != "infrastructure_error"
    )
    summary = EvalSmokeSummary(
        status=execution.status,
        matrix_hash=execution.matrix_hash,
        accepted_count=len(episodes),
        missing_episode_ids=execution.completeness.missing_episode_ids,
        episodes=episodes,
    )
    return EvaluationSmokeResult(execution=execution, summary=summary)


run_eval_smoke = run_evaluation_smoke


__all__ = [
    "EVAL_SMOKE_SLOTS",
    "EVAL_SMOKE_TASK_IDS",
    "EvalEpisodeSummary",
    "EvalSmokeSummary",
    "EvaluationSmokeResult",
    "run_eval_smoke",
    "run_evaluation_smoke",
]
