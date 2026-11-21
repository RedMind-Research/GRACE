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

"""Rebuild and verify the published Tau² Telecom task split offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from reproduction.tau2_telecom.harness.identity import PINNED_TAU2_REVISION
from reproduction.tau2_telecom.harness.splits import (
    FROZEN_PHASE_SEQUENCE,
    SplitTask,
    Tau2TelecomExperienceBatch,
    Tau2TelecomTaskSplit,
    load_tau2_telecom_task_split,
)

from .config import EVALUATION_QUOTAS, EXPERIENCE_QUOTAS, PERSONAS, PINNED_TASKS_SHA256, SPLIT_SEED


def _load_task_ids(path: Path) -> list[str]:
    # Windows checkouts may convert the pinned file to CRLF.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != PINNED_TASKS_SHA256:
        raise ValueError(
            f"Expected the telecom tasks.json from Tau² revision {PINNED_TAU2_REVISION}"
        )
    return [task["id"] for task in json.loads(data)]


def _group_task_ids(task_ids: Sequence[str]) -> dict[tuple[str, str], list[str]]:
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Duplicate task IDs in the telecom task pool")

    pools: dict[tuple[str, str], list[str]] = {
        (intent, persona): [] for intent in EVALUATION_QUOTAS for persona in PERSONAS
    }
    for task_id in task_ids:
        intent = re.match(r"^\[([a-zA-Z_]+)\]", task_id)
        persona = re.search(r"\[PERSONA:([a-zA-Z_]+)\]$", task_id)
        if intent is None or persona is None:
            raise ValueError(f"Missing intent or persona in task ID {task_id!r}")
        group = (intent.group(1), persona.group(1))
        if group not in pools:
            raise ValueError(f"Unsupported intent or persona in task ID {task_id!r}")
        pools[group].append(task_id)

    for pool in pools.values():
        pool.sort()
    return pools


def _sample_tasks(
    pools: dict[tuple[str, str], list[str]], quotas: dict[str, int], rng: random.Random
) -> tuple[SplitTask, ...]:
    selected: list[str] = []
    for intent, count in quotas.items():
        for persona in PERSONAS:
            group = (intent, persona)
            pool = pools[group]
            if len(pool) < count:
                raise ValueError(f"Need {count} tasks for {intent}/{persona} but found {len(pool)}")
            chosen = rng.sample(pool, count)
            selected.extend(chosen)
            chosen_ids = set(chosen)
            pools[group] = [task_id for task_id in pool if task_id not in chosen_ids]
    return tuple(SplitTask(slot=slot, task_id=task_id) for slot, task_id in enumerate(selected))


def build_split(task_ids: Sequence[str]) -> Tau2TelecomTaskSplit:
    """Apply the original held-out-first sampling order to the candidate task IDs."""
    pools = _group_task_ids(task_ids)
    rng = random.Random(SPLIT_SEED)
    evaluation = _sample_tasks(pools, EVALUATION_QUOTAS, rng)
    experience = tuple(
        Tau2TelecomExperienceBatch(
            batch=batch,
            phase=phase,
            tasks=_sample_tasks(pools, EXPERIENCE_QUOTAS[phase], rng),
        )
        for batch, phase in enumerate(FROZEN_PHASE_SEQUENCE)
    )
    return Tau2TelecomTaskSplit(
        schema_version="grace.tau2-telecom-task-splits.v1",
        benchmark="tau2",
        benchmark_revision=PINNED_TAU2_REVISION,
        domain="telecom",
        phase_sequence=FROZEN_PHASE_SEQUENCE,
        eval=evaluation,
        experience=experience,
    )


def _write_split(path: Path, split: Tau2TelecomTaskSplit) -> None:
    data = (split.model_dump_json(indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-json", required=True, type=Path, help="Pinned telecom tasks.json")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for rebuilt-splits.json"
    )
    args = parser.parse_args(argv)

    try:
        rebuilt = build_split(_load_task_ids(args.tasks_json))
        if rebuilt != load_tau2_telecom_task_split():
            raise ValueError("The rebuilt task IDs or order differ from the frozen split")
        output = args.output_dir / "rebuilt-splits.json"
        _write_split(output, rebuilt)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    experience_count = sum(len(batch.tasks) for batch in rebuilt.experience)
    print(f"Verified {len(rebuilt.eval)} evaluation tasks and {experience_count} experience tasks")
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
