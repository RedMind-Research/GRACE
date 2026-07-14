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

"""Ordered task resolution without importing tau2 at module import time."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from .identity import canonical_json, canonical_sha256, make_task_uid
from .models import TaskMapEntry, TaskSelectionManifest


TaskT = TypeVar("TaskT")


class TaskLike(Protocol):
    """Minimum interface expected from an injected benchmark task object."""

    id: str


TaskSource = Iterable[TaskT] | Callable[[], Iterable[TaskT]]
DefinitionSerializer = Callable[[TaskT], Any]


class TaskResolutionError(ValueError):
    """Raised when a source cannot satisfy an exact ordered task selection."""


@dataclass(frozen=True)
class ResolvedTaskSelection:
    """Task objects and their immutable mapping entries in manifest order."""

    tasks: tuple[Any, ...]
    task_map: tuple[TaskMapEntry, ...]


def _load_source(source: TaskSource[TaskT]) -> tuple[TaskT, ...]:
    loaded = source() if callable(source) else source
    return tuple(loaded)


def _task_id(task: Any) -> str:
    if isinstance(task, Mapping):
        value = task.get("id")
    else:
        value = getattr(task, "id", None)
    if not isinstance(value, str) or not value:
        raise TaskResolutionError("every source task must expose a non-empty string id")
    return value


def load_tasks_in_manifest_order(
    requested_task_ids: Iterable[str],
    source: TaskSource[TaskT],
) -> tuple[TaskT, ...]:
    """Resolve requested IDs and rebuild their exact requested order.

    Extra source tasks are allowed because a tau2 domain loader returns the full
    master corpus.  Duplicate source IDs, duplicate requested IDs, and missing
    requested IDs are fatal.  Membership filtering followed by source-order
    slicing is intentionally forbidden.
    """

    requested = tuple(requested_task_ids)
    if any(not isinstance(task_id, str) or not task_id for task_id in requested):
        raise TaskResolutionError("requested task IDs must be non-empty strings")
    if len(requested) != len(set(requested)):
        raise TaskResolutionError("requested task IDs contain duplicates")

    by_id: dict[str, TaskT] = {}
    duplicate_source_ids: set[str] = set()
    for task in _load_source(source):
        task_id = _task_id(task)
        if task_id in by_id:
            duplicate_source_ids.add(task_id)
        else:
            by_id[task_id] = task
    if duplicate_source_ids:
        duplicates = ", ".join(sorted(duplicate_source_ids))
        raise TaskResolutionError(f"source contains duplicate task IDs: {duplicates}")

    missing = [task_id for task_id in requested if task_id not in by_id]
    if missing:
        raise TaskResolutionError(f"source is missing requested task IDs: {missing}")
    return tuple(by_id[task_id] for task_id in requested)


def _default_definition_payload(task: Any) -> Any:
    if isinstance(task, BaseModel):
        return task.model_dump(mode="json")
    if isinstance(task, Mapping):
        return dict(task)
    if is_dataclass(task) and not isinstance(task, type):
        return asdict(task)
    model_dump = getattr(task, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    raise TaskResolutionError(
        "task definition is not canonically serializable; inject definition_serializer"
    )


def resolve_task_selection(
    manifest: TaskSelectionManifest,
    source: TaskSource[TaskT],
    *,
    definition_serializer: DefinitionSerializer[TaskT] | None = None,
    publication_status: Literal["public", "sanitized", "private"] = "private",
) -> ResolvedTaskSelection:
    """Resolve one manifest and derive its collision-safe task map."""

    requested_ids = tuple(slot.benchmark_task_id for slot in manifest.tasks)
    tasks = load_tasks_in_manifest_order(requested_ids, source)
    serializer = definition_serializer or _default_definition_payload
    task_map: list[TaskMapEntry] = []
    for slot, task in zip(manifest.tasks, tasks, strict=True):
        definition_hash = canonical_sha256(
            {
                "definition": serializer(task),
                "namespace": "grace.task-definition.v1",
            }
        )
        task_map.append(
            TaskMapEntry(
                benchmark=manifest.benchmark,
                benchmark_revision=manifest.benchmark_revision,
                domain=manifest.domain,
                benchmark_task_id=slot.benchmark_task_id,
                task_uid=make_task_uid(
                    slot.benchmark_task_id,
                    benchmark_revision=manifest.benchmark_revision,
                    domain=manifest.domain,
                    benchmark=manifest.benchmark,
                ),
                split=slot.split,
                batch=slot.batch,
                slot=slot.slot,
                phase=slot.phase,
                split_slot=slot.split_slot,
                task_definition_hash=definition_hash,
                publication_status=publication_status,
            )
        )
    return ResolvedTaskSelection(tasks=tasks, task_map=tuple(task_map))


def load_task_selection_manifest(path: str | Path) -> TaskSelectionManifest:
    """Read and validate a public task-selection manifest."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskResolutionError(f"cannot read task manifest {manifest_path}: {exc}") from exc
    return TaskSelectionManifest.model_validate(payload)


def render_task_map_jsonl(entries: Iterable[TaskMapEntry]) -> str:
    """Render an atomic-write-ready task map without mutating the filesystem."""

    records = tuple(entries)
    task_uids = [entry.task_uid for entry in records]
    split_slots = [entry.split_slot for entry in records]
    if len(task_uids) != len(set(task_uids)):
        raise TaskResolutionError("task map contains duplicate task UIDs")
    if len(split_slots) != len(set(split_slots)):
        raise TaskResolutionError("task map contains duplicate split locators")
    return "".join(canonical_json(entry.model_dump(mode="json")) + "\n" for entry in records)


__all__ = [
    "DefinitionSerializer",
    "ResolvedTaskSelection",
    "TaskLike",
    "TaskResolutionError",
    "load_task_selection_manifest",
    "load_tasks_in_manifest_order",
    "render_task_map_jsonl",
    "resolve_task_selection",
]
