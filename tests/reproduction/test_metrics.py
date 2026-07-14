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

from __future__ import annotations

import pytest

from reproduction.tau2_telecom.harness.identity import make_matrix_hash
from reproduction.tau2_telecom.harness.metrics import (
    EvaluationObservation,
    MetricsContractError,
    compute_evaluation_metrics,
)
from reproduction.tau2_telecom.harness.models import EpisodeCell, ExpectedMatrix


def _matrix() -> ExpectedMatrix:
    cells = tuple(
        EpisodeCell(
            episode_id=f"{index + 1:064x}",
            task_uid=f"{task + 20:064x}",
            task_definition_hash=f"{task + 40:064x}",
            benchmark_task_id=f"task-{task}",
            split_slot=f"eval-s{task:03d}",
            trial=trial,
            protocol_seed=100 + trial,
            checkpoint_state_id="state-1",
        )
        for index, (task, trial) in enumerate(
            (task_trial for task in range(2) for task_trial in ((task, 0), (task, 1), (task, 2)))
        )
    )
    return ExpectedMatrix(
        episodes=cells,
        matrix_hash=make_matrix_hash(cell.episode_id for cell in cells),
    )


def _observations(matrix: ExpectedMatrix) -> list[EvaluationObservation]:
    success = (True, False, False, True, True, True)
    return [
        EvaluationObservation(
            episode_id=cell.episode_id,
            benchmark_task_id=cell.benchmark_task_id,
            trial=cell.trial,
            success=value,
        )
        for cell, value in zip(matrix.episodes, success, strict=True)
    ]


def test_metrics_match_declared_pass_semantics() -> None:
    matrix = _matrix()
    result = compute_evaluation_metrics(matrix, _observations(matrix))

    assert result.task_count == 2
    assert result.episode_count == 6
    assert result.k == 3
    assert result.pass_at_1 == pytest.approx(4 / 6)
    assert result.pass_at_k == 1.0
    assert result.pass_power_k == 0.5


def test_metrics_fail_closed_on_missing_or_duplicate_episode() -> None:
    matrix = _matrix()
    observations = _observations(matrix)

    with pytest.raises(ValueError, match="matrix is incomplete"):
        compute_evaluation_metrics(matrix, observations[:-1])
    with pytest.raises(ValueError, match="matrix is incomplete"):
        compute_evaluation_metrics(matrix, [*observations, observations[0]])


def test_metrics_reject_task_or_trial_mismatch() -> None:
    matrix = _matrix()
    observations = _observations(matrix)

    wrong_task = observations.copy()
    wrong_task[0] = wrong_task[0].model_copy(update={"benchmark_task_id": "wrong"})
    with pytest.raises(MetricsContractError, match="task/trial identity"):
        compute_evaluation_metrics(matrix, wrong_task)

    wrong_trial = observations.copy()
    wrong_trial[0] = wrong_trial[0].model_copy(update={"trial": 9})
    with pytest.raises(MetricsContractError, match="task/trial identity"):
        compute_evaluation_metrics(matrix, wrong_trial)
