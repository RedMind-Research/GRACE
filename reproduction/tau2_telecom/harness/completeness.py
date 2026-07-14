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

"""Exact expected-matrix construction and deterministic completeness gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .identity import make_episode_id, make_matrix_hash, stable_seed_offset
from .models import CompletenessReport, EpisodeCell, ExpectedMatrix, TaskMapEntry


class CompletenessError(ValueError):
    """Raised when observed artifacts do not exactly satisfy a frozen matrix."""

    def __init__(self, report: CompletenessReport):
        self.report = report
        super().__init__(
            "episode matrix is incomplete: "
            f"missing={len(report.missing_episode_ids)}, "
            f"unknown={len(report.unknown_episode_ids)}, "
            f"duplicates={len(report.duplicate_episode_ids)}, "
            f"out_of_order={report.out_of_order}"
        )


def _checkpoint_for(
    checkpoint_state_id: str | Mapping[str, str],
    split_slot: str,
) -> str:
    if isinstance(checkpoint_state_id, str):
        if not checkpoint_state_id:
            raise ValueError("checkpoint_state_id must not be empty")
        return checkpoint_state_id
    try:
        value = checkpoint_state_id[split_slot]
    except KeyError as exc:
        raise ValueError(f"missing checkpoint for {split_slot}") from exc
    if not value:
        raise ValueError(f"checkpoint for {split_slot} must not be empty")
    return value


def build_expected_matrix(
    task_map: Iterable[TaskMapEntry],
    *,
    trials: Iterable[int],
    seed: int,
    checkpoint_state_id: str | Mapping[str, str],
    protocol_hash: str,
    config_hash: str,
    model_roles_hash: str,
    evaluator_hash: str,
) -> ExpectedMatrix:
    """Build cells in task-map order, then caller-declared trial order."""

    entries = tuple(task_map)
    trial_order = tuple(trials)
    if any(trial < 0 for trial in trial_order):
        raise ValueError("trials must be zero or greater")
    if len(trial_order) != len(set(trial_order)):
        raise ValueError("trials contain duplicates")
    if len({entry.split_slot for entry in entries}) != len(entries):
        raise ValueError("task map contains duplicate split locators")

    episodes: list[EpisodeCell] = []
    for entry in entries:
        checkpoint = _checkpoint_for(checkpoint_state_id, entry.split_slot)
        for trial in trial_order:
            protocol_seed = seed + stable_seed_offset(entry.benchmark_task_id) + trial * 1000
            episode_id = make_episode_id(
                task_uid=entry.task_uid,
                task_definition_hash=entry.task_definition_hash,
                split_slot=entry.split_slot,
                trial=trial,
                protocol_seed=protocol_seed,
                checkpoint_state_id=checkpoint,
                benchmark_revision=entry.benchmark_revision,
                protocol_hash=protocol_hash,
                config_hash=config_hash,
                model_roles_hash=model_roles_hash,
                evaluator_hash=evaluator_hash,
            )
            episodes.append(
                EpisodeCell(
                    episode_id=episode_id,
                    task_uid=entry.task_uid,
                    task_definition_hash=entry.task_definition_hash,
                    benchmark_task_id=entry.benchmark_task_id,
                    split_slot=entry.split_slot,
                    trial=trial,
                    protocol_seed=protocol_seed,
                    checkpoint_state_id=checkpoint,
                )
            )
    episode_ids = [episode.episode_id for episode in episodes]
    return ExpectedMatrix(episodes=tuple(episodes), matrix_hash=make_matrix_hash(episode_ids))


def _episode_id(value: Any) -> str:
    if isinstance(value, str):
        candidate: Any = value
    elif isinstance(value, Mapping):
        candidate = value.get("episode_id")
    else:
        candidate = getattr(value, "episode_id", None)
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("every observed item must expose a non-empty episode_id")
    return candidate


def compare_exact_matrix(
    expected: ExpectedMatrix,
    observed: Iterable[Any],
) -> CompletenessReport:
    """Compare observed artifacts without hiding duplicates or unknown cells."""

    expected_ids = tuple(episode.episode_id for episode in expected.episodes)
    observed_ids = tuple(_episode_id(item) for item in observed)
    expected_set = set(expected_ids)
    counts = Counter(observed_ids)

    duplicates = tuple(sorted(episode_id for episode_id, count in counts.items() if count > 1))
    unknown = tuple(sorted(episode_id for episode_id in counts if episode_id not in expected_set))
    observed_known = set(observed_ids) & expected_set
    missing = tuple(episode_id for episode_id in expected_ids if episode_id not in observed_known)
    known_first_occurrence = tuple(
        dict.fromkeys(episode_id for episode_id in observed_ids if episode_id in expected_set)
    )
    expected_known_order = tuple(
        episode_id for episode_id in expected_ids if episode_id in observed_known
    )
    out_of_order = known_first_occurrence != expected_known_order
    complete = (
        not missing and not unknown and not duplicates and len(observed_ids) == len(expected_ids)
    )

    return CompletenessReport(
        matrix_hash=expected.matrix_hash,
        expected_count=len(expected_ids),
        observed_count=len(observed_ids),
        accepted_count=len(observed_known),
        missing_episode_ids=missing,
        unknown_episode_ids=unknown,
        duplicate_episode_ids=duplicates,
        out_of_order=out_of_order,
        complete=complete,
    )


def require_exact_matrix(
    expected: ExpectedMatrix,
    observed: Iterable[Any],
    *,
    require_canonical_order: bool = False,
) -> CompletenessReport:
    """Raise unless the observed IDs equal the exact expected set."""

    report = compare_exact_matrix(expected, observed)
    if not report.complete or (require_canonical_order and report.out_of_order):
        raise CompletenessError(report)
    return report


__all__ = [
    "CompletenessError",
    "build_expected_matrix",
    "compare_exact_matrix",
    "require_exact_matrix",
]
