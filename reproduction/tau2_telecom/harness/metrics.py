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

"""Transparent pass metrics for an exact, complete evaluation matrix."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .completeness import require_exact_matrix
from .models import ExpectedMatrix


NonEmpty = Annotated[str, Field(min_length=1)]


class EvaluationObservation(BaseModel):
    """One accepted evaluation cell used for aggregate metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: NonEmpty
    benchmark_task_id: NonEmpty
    trial: int = Field(ge=0)
    success: bool


class EvaluationMetrics(BaseModel):
    """Task-level and episode-level success metrics for exactly ``k`` trials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_count: int = Field(ge=1)
    episode_count: int = Field(ge=1)
    k: int = Field(ge=1)
    pass_at_1: float = Field(ge=0.0, le=1.0)
    pass_at_k: float = Field(ge=0.0, le=1.0)
    pass_power_k: float = Field(ge=0.0, le=1.0)


class MetricsContractError(ValueError):
    """Raised when observations do not match the declared evaluation design."""


def compute_evaluation_metrics(
    expected: ExpectedMatrix,
    observations: Sequence[EvaluationObservation],
) -> EvaluationMetrics:
    """Compute metrics only after exact identity and task/trial validation.

    ``pass@k`` is the share of tasks with at least one successful trial.
    ``pass^k`` is the share of tasks successful in every trial.
    ``pass@1`` is the mean success rate over all task/trial cells.
    """

    require_exact_matrix(expected, observations)
    cells = {cell.episode_id: cell for cell in expected.episodes}
    by_task: dict[str, dict[int, bool]] = defaultdict(dict)

    for observation in observations:
        cell = cells[observation.episode_id]
        if (
            observation.benchmark_task_id != cell.benchmark_task_id
            or observation.trial != cell.trial
        ):
            raise MetricsContractError(
                "observation task/trial identity does not match its expected episode"
            )
        task_trials = by_task[observation.benchmark_task_id]
        if observation.trial in task_trials:
            raise MetricsContractError("duplicate task/trial observation")
        task_trials[observation.trial] = observation.success

    trial_sets = {
        tuple(sorted(trials)) for trials in (values.keys() for values in by_task.values())
    }
    if len(trial_sets) != 1:
        raise MetricsContractError("every task must contain the same trial set")
    trials = next(iter(trial_sets), ())
    if not trials or trials != tuple(range(len(trials))):
        raise MetricsContractError("trials must be contiguous and zero-indexed")

    successes = tuple(tuple(values[trial] for trial in trials) for values in by_task.values())
    task_count = len(successes)
    episode_count = sum(len(values) for values in successes)
    return EvaluationMetrics(
        task_count=task_count,
        episode_count=episode_count,
        k=len(trials),
        pass_at_1=sum(sum(values) for values in successes) / episode_count,
        pass_at_k=sum(any(values) for values in successes) / task_count,
        pass_power_k=sum(all(values) for values in successes) / task_count,
    )


__all__ = [
    "EvaluationMetrics",
    "EvaluationObservation",
    "MetricsContractError",
    "compute_evaluation_metrics",
]
