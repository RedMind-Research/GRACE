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

"""Typed public task identities for the paper's telecom procedure."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .identity import PINNED_TAU2_REVISION
from .models import TaskSelectionManifest, TaskSlotSpec


Frozen = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1)]
Phase = Literal["A", "B"]
FROZEN_PHASE_SEQUENCE: tuple[Phase, ...] = ("A", "A", "B", "B", "A", "A", "B", "B", "A", "A")
EVAL_TASK_COUNT = 66
EXPERIENCE_BATCH_COUNT = 10
EXPERIENCE_TASKS_PER_BATCH = 42
TOTAL_TASK_COUNT = 486


class Tau2TelecomSplitError(ValueError):
    """Raised when the public split is malformed or incomplete."""


class SplitTask(BaseModel):
    model_config = Frozen

    slot: int = Field(ge=0)
    task_id: NonEmpty


class Tau2TelecomExperienceBatch(BaseModel):
    model_config = Frozen

    batch: int = Field(ge=0, lt=EXPERIENCE_BATCH_COUNT)
    phase: Phase
    tasks: tuple[SplitTask, ...] = Field(
        min_length=EXPERIENCE_TASKS_PER_BATCH,
        max_length=EXPERIENCE_TASKS_PER_BATCH,
    )

    @model_validator(mode="after")
    def validate_order(self) -> Tau2TelecomExperienceBatch:
        if tuple(task.slot for task in self.tasks) != tuple(range(EXPERIENCE_TASKS_PER_BATCH)):
            raise ValueError("experience tasks must be ordered slots 0..41")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("experience batch contains duplicate task IDs")
        return self


class Tau2TelecomTaskSplit(BaseModel):
    """Exact task selection only; no trajectory or private-source provenance."""

    model_config = Frozen

    schema_version: Literal["grace.tau2-telecom-task-splits.v1"]
    benchmark: Literal["tau2"]
    benchmark_revision: NonEmpty
    domain: Literal["telecom"]
    phase_sequence: tuple[Phase, ...]
    eval: tuple[SplitTask, ...] = Field(min_length=EVAL_TASK_COUNT, max_length=EVAL_TASK_COUNT)
    experience: tuple[Tau2TelecomExperienceBatch, ...] = Field(
        min_length=EXPERIENCE_BATCH_COUNT,
        max_length=EXPERIENCE_BATCH_COUNT,
    )

    @model_validator(mode="after")
    def validate_selection(self) -> Tau2TelecomTaskSplit:
        if self.benchmark_revision != PINNED_TAU2_REVISION:
            raise ValueError("benchmark revision does not match the pinned tau2 checkout")
        if self.phase_sequence != FROZEN_PHASE_SEQUENCE:
            raise ValueError("phase sequence must be AABBAABBAA")
        if tuple(task.slot for task in self.eval) != tuple(range(EVAL_TASK_COUNT)):
            raise ValueError("evaluation tasks must be ordered slots 0..65")
        if tuple(batch.batch for batch in self.experience) != tuple(range(EXPERIENCE_BATCH_COUNT)):
            raise ValueError("experience batches must be ordered 0..9")
        if tuple(batch.phase for batch in self.experience) != self.phase_sequence:
            raise ValueError("experience phases do not match phase_sequence")
        task_ids = tuple(task.task_id for task in self.iter_tasks())
        if len(task_ids) != TOTAL_TASK_COUNT or len(set(task_ids)) != TOTAL_TASK_COUNT:
            raise ValueError("task split must contain 486 unique task IDs")
        return self

    def iter_tasks(self) -> Iterator[SplitTask]:
        yield from self.eval
        for batch in self.experience:
            yield from batch.tasks

    def eval_task(self, slot: int) -> SplitTask:
        if not 0 <= slot < EVAL_TASK_COUNT:
            raise Tau2TelecomSplitError("evaluation slot is outside 0..65")
        return self.eval[slot]

    def experience_manifest(self, update_index: int, *, seed: int) -> TaskSelectionManifest:
        """Return the 42-task manifest for paper update ``t`` in 1..10."""

        if not 1 <= update_index <= EXPERIENCE_BATCH_COUNT:
            raise Tau2TelecomSplitError("update index is outside 1..10")
        batch = self.experience[update_index - 1]
        return TaskSelectionManifest(
            benchmark_revision=self.benchmark_revision,
            seed=seed,
            tasks=tuple(
                TaskSlotSpec(
                    split="experience",
                    slot=task.slot,
                    benchmark_task_id=task.task_id,
                    batch=batch.batch,
                    phase=batch.phase,
                )
                for task in batch.tasks
            ),
        )

    def evaluation_manifest(self, *, seed: int) -> TaskSelectionManifest:
        """Return the exact 66-task formal evaluation manifest."""

        return TaskSelectionManifest(
            benchmark_revision=self.benchmark_revision,
            seed=seed,
            tasks=tuple(
                TaskSlotSpec(
                    split="eval",
                    slot=task.slot,
                    benchmark_task_id=task.task_id,
                )
                for task in self.eval
            ),
        )


def default_tau2_telecom_split_path() -> Path:
    return Path(__file__).parents[1] / "data" / "splits.json"


def load_tau2_telecom_task_split(path: str | Path | None = None) -> Tau2TelecomTaskSplit:
    split_path = Path(path) if path is not None else default_tau2_telecom_split_path()
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Tau2TelecomSplitError(f"cannot read task split {split_path}") from exc
    return Tau2TelecomTaskSplit.model_validate(payload)


__all__ = [
    "EVAL_TASK_COUNT",
    "EXPERIENCE_BATCH_COUNT",
    "EXPERIENCE_TASKS_PER_BATCH",
    "FROZEN_PHASE_SEQUENCE",
    "Tau2TelecomExperienceBatch",
    "Tau2TelecomSplitError",
    "Tau2TelecomTaskSplit",
    "TOTAL_TASK_COUNT",
    "default_tau2_telecom_split_path",
    "load_tau2_telecom_task_split",
]
