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

import copy
import json

import pytest

from reproduction.tau2_telecom.harness.splits import (
    EVAL_TASK_COUNT,
    EXPERIENCE_BATCH_COUNT,
    EXPERIENCE_TASKS_PER_BATCH,
    FROZEN_PHASE_SEQUENCE,
    TOTAL_TASK_COUNT,
    Tau2TelecomSplitError,
    Tau2TelecomTaskSplit,
    default_tau2_telecom_split_path,
    load_tau2_telecom_task_split,
)


def _payload() -> dict[str, object]:
    return json.loads(default_tau2_telecom_split_path().read_text(encoding="utf-8"))


def test_public_split_contains_only_selection_metadata() -> None:
    payload = _payload()
    authority = load_tau2_telecom_task_split()

    assert set(payload) == {
        "schema_version",
        "benchmark",
        "benchmark_revision",
        "domain",
        "phase_sequence",
        "eval",
        "experience",
    }
    assert len(authority.eval) == EVAL_TASK_COUNT == 66
    assert len(authority.experience) == EXPERIENCE_BATCH_COUNT == 10
    assert all(
        len(batch.tasks) == EXPERIENCE_TASKS_PER_BATCH == 42 for batch in authority.experience
    )
    assert len(tuple(authority.iter_tasks())) == TOTAL_TASK_COUNT == 486
    assert authority.phase_sequence == FROZEN_PHASE_SEQUENCE


def test_task_ids_are_unique_and_order_is_exact() -> None:
    authority = load_tau2_telecom_task_split()
    assert tuple(task.slot for task in authority.eval) == tuple(range(66))
    assert tuple(batch.batch for batch in authority.experience) == tuple(range(10))
    assert all(
        tuple(task.slot for task in batch.tasks) == tuple(range(42))
        for batch in authority.experience
    )
    ids = tuple(task.task_id for task in authority.iter_tasks())
    assert len(ids) == len(set(ids)) == 486


def test_wrong_size_duplicate_and_phase_fail_closed() -> None:
    payload = _payload()
    missing = copy.deepcopy(payload)
    missing["eval"].pop()  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        Tau2TelecomTaskSplit.model_validate(missing)

    duplicate = copy.deepcopy(payload)
    duplicate["experience"][0]["tasks"][1]["task_id"] = duplicate["experience"][0]["tasks"][0][
        "task_id"
    ]  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate"):
        Tau2TelecomTaskSplit.model_validate(duplicate)

    phase = copy.deepcopy(payload)
    phase["phase_sequence"][0] = "B"  # type: ignore[index]
    with pytest.raises(ValueError, match="AABBAABBAA"):
        Tau2TelecomTaskSplit.model_validate(phase)


def test_eval_slot_lookup_is_bounded() -> None:
    authority = load_tau2_telecom_task_split()
    assert authority.eval_task(20) is authority.eval[20]
    with pytest.raises(Tau2TelecomSplitError, match="outside"):
        authority.eval_task(66)
