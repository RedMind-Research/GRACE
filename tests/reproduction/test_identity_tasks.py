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

import json
import math
from pathlib import Path

import pytest

from reproduction.tau2_telecom.harness.identity import (
    EVAL20_TASK_ID,
    EVAL20_TASK_UID,
    IdentityError,
    PINNED_TAU2_REVISION,
    canonical_json,
    make_reflection_id,
    make_split_slot,
    make_task_uid,
    verify_release_test_vectors,
)
from reproduction.tau2_telecom.harness.tasks import (
    TaskResolutionError,
    load_task_selection_manifest,
    load_tasks_in_manifest_order,
    render_task_map_jsonl,
    resolve_task_selection,
)


DATA_DIR = Path(__file__).parents[2] / "reproduction/tau2_telecom/data"


def test_eval20_canonical_uid_matches_reviewed_test_vector() -> None:
    verify_release_test_vectors()
    preimage = {
        "benchmark": "tau2",
        "benchmark_revision": PINNED_TAU2_REVISION,
        "domain": "telecom",
        "namespace": "grace.task.v1",
        "task_id": EVAL20_TASK_ID,
    }

    assert canonical_json(preimage) == (
        '{"benchmark":"tau2","benchmark_revision":"c5b2d228d850c59b749b93cf32c4745d3aa53967",'
        '"domain":"telecom","namespace":"grace.task.v1",'
        '"task_id":"[mms_issue]break_apn_mms_setting[PERSONA:Hard]"}'
    )
    assert make_task_uid(EVAL20_TASK_ID, benchmark_revision=PINNED_TAU2_REVISION) == (
        EVAL20_TASK_UID
    )


def test_canonical_json_rejects_ambiguous_values() -> None:
    with pytest.raises(IdentityError, match="object keys"):
        canonical_json({1: "not allowed"})
    with pytest.raises(IdentityError, match="NaN"):
        canonical_json({"bad": math.nan})


def test_long_semantic_ids_never_collide_or_become_paths() -> None:
    shared = "[mms_issue]" + "same_prefix|" * 12
    first = shared + "first[PERSONA:Hard]"
    second = shared + "second[PERSONA:Hard]"
    assert first[:80] == second[:80]

    first_uid = make_task_uid(first, benchmark_revision=PINNED_TAU2_REVISION)
    second_uid = make_task_uid(second, benchmark_revision=PINNED_TAU2_REVISION)
    assert first_uid != second_uid
    assert len(first_uid) == len(second_uid) == 64

    common = {
        "episode_id": "1" * 64,
        "trajectory_hash": "2" * 64,
        "policy_hash": "3" * 64,
        "diagnosis_prompt_hash": "4" * 64,
        "model": "vertex_ai/gemini-2.5-flash",
        "config_hash": "5" * 64,
    }
    assert make_reflection_id(task_uid=first_uid, **common) != make_reflection_id(
        task_uid=second_uid, **common
    )


def test_split_locators_are_zero_based_and_batch_qualified() -> None:
    assert make_split_slot("experience", batch=1, slot=28) == "exp-b01-s028"
    assert make_split_slot("eval", slot=20) == "eval-s020"
    with pytest.raises(IdentityError, match="requires"):
        make_split_slot("experience", slot=28)


def test_ordered_loader_rebuilds_manifest_order_not_source_order() -> None:
    source = [{"id": "task-c"}, {"id": "task-a"}, {"id": "task-b"}]

    loaded = load_tasks_in_manifest_order(["task-b", "task-c"], source)

    assert [task["id"] for task in loaded] == ["task-b", "task-c"]


@pytest.mark.parametrize(
    ("requested", "source", "message"),
    [
        (["task-a", "task-a"], [{"id": "task-a"}], "requested"),
        (["task-a"], [{"id": "task-a"}, {"id": "task-a"}], "source"),
        (["missing"], [{"id": "task-a"}], "missing"),
    ],
)
def test_ordered_loader_fails_on_duplicate_or_missing_ids(requested, source, message) -> None:
    with pytest.raises(TaskResolutionError, match=message):
        load_tasks_in_manifest_order(requested, source)


def test_public_eval_manifest_resolves_without_tau2_import() -> None:
    manifest = load_task_selection_manifest(DATA_DIR / "eval_smoke_v1.json")
    master_source = [
        {"id": task.benchmark_task_id, "definition": {"ordinal": index}}
        for index, task in reversed(tuple(enumerate(manifest.tasks)))
    ]

    resolved = resolve_task_selection(manifest, lambda: master_source, publication_status="public")
    rendered = render_task_map_jsonl(resolved.task_map)
    records = [json.loads(line) for line in rendered.splitlines()]

    assert [task["id"] for task in resolved.tasks] == [
        task.benchmark_task_id for task in manifest.tasks
    ]
    assert [record["split_slot"] for record in records] == [
        "eval-s020",
        "eval-s043",
        "eval-s060",
    ]
    assert records[0]["task_uid"] == EVAL20_TASK_UID
    assert all(len(record["task_definition_hash"]) == 64 for record in records)
