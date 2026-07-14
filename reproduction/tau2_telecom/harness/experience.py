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

"""Sequential experience collection over an exact manifest/matrix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .models import ExpectedMatrix, TaskSelectionManifest, TrajectoryRecord
from .tau2_interface import (
    EpisodeArtifactAuthority,
    EpisodeContractError,
    EpisodeExecutor,
    EpisodeHarnessConfig,
    EpisodeMatrixExecution,
    execute_matrix_sequential,
)


class ExperienceRunResult(BaseModel):
    """Accepted public-safe trajectories plus exact execution evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["grace.experience-run.v1"] = "grace.experience-run.v1"
    status: Literal["complete", "incomplete"]
    execution: EpisodeMatrixExecution
    trajectories: tuple[TrajectoryRecord, ...]

    @model_validator(mode="after")
    def validate_bindings(self) -> ExperienceRunResult:
        if self.status != self.execution.status:
            raise ValueError("experience status must match matrix execution")
        expected = tuple(item.trajectory for item in self.execution.accepted)
        if any(item is None for item in expected) or self.trajectories != expected:
            raise ValueError("experience trajectories must be accepted authority artifacts")
        return self


def _validate_experience_scope(
    manifest: TaskSelectionManifest,
    expected: ExpectedMatrix,
    config: EpisodeHarnessConfig,
) -> None:
    if manifest.benchmark_revision != config.benchmark_revision:
        raise EpisodeContractError("manifest and execution tau2 revisions differ")
    if not manifest.tasks or any(task.split != "experience" for task in manifest.tasks):
        raise EpisodeContractError("experience runner requires only experience task slots")
    expected_shape = tuple(
        (episode.split_slot, episode.benchmark_task_id, episode.trial)
        for episode in expected.episodes
    )
    manifest_shape = tuple((task.split_slot, task.benchmark_task_id, 0) for task in manifest.tasks)
    if expected_shape != manifest_shape:
        raise EpisodeContractError(
            "experience expected matrix must contain manifest tasks once, trial zero, in order"
        )


def run_experience_collection(
    *,
    manifest: TaskSelectionManifest,
    expected: ExpectedMatrix,
    tasks: Sequence[Any],
    policy: str,
    prompt_hash: str,
    config: EpisodeHarnessConfig,
    executor: EpisodeExecutor,
    authority: EpisodeArtifactAuthority,
) -> ExperienceRunResult:
    """Collect one accepted behavioral trajectory per exact matrix cell.

    Reward-zero and non-success tau2 terminations are retained as behavioral
    trajectories.  Infrastructure attempts remain in immutable authority but
    never become diagnosis input.
    """

    _validate_experience_scope(manifest, expected, config)
    execution = execute_matrix_sequential(
        expected=expected,
        tasks=tasks,
        policy=policy,
        prompt_hash=prompt_hash,
        config=config,
        executor=executor,
        authority=authority,
    )
    trajectories = tuple(
        outcome.trajectory for outcome in execution.accepted if outcome.trajectory is not None
    )
    return ExperienceRunResult(
        status=execution.status,
        execution=execution,
        trajectories=trajectories,
    )


run_experience = run_experience_collection


__all__ = ["ExperienceRunResult", "run_experience", "run_experience_collection"]
