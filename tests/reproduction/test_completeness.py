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

from pathlib import Path

import pytest

from reproduction.tau2_telecom.harness.completeness import (
    CompletenessError,
    build_expected_matrix,
    compare_exact_matrix,
    require_exact_matrix,
)
from reproduction.tau2_telecom.harness.identity import canonical_sha256
from reproduction.tau2_telecom.harness.tasks import (
    load_task_selection_manifest,
    resolve_task_selection,
)


DATA_DIR = Path(__file__).parents[2] / "reproduction/tau2_telecom/data"


def _eval_matrix():
    manifest = load_task_selection_manifest(DATA_DIR / "eval_smoke_v1.json")
    source = tuple({"id": item.benchmark_task_id} for item in reversed(manifest.tasks))
    task_map = resolve_task_selection(manifest, source, publication_status="public").task_map
    hashes = {
        name: canonical_sha256({"name": name})
        for name in ("protocol", "config", "models", "evaluator")
    }
    return build_expected_matrix(
        task_map,
        trials=(0,),
        seed=manifest.seed,
        checkpoint_state_id="state-10",
        protocol_hash=hashes["protocol"],
        config_hash=hashes["config"],
        model_roles_hash=hashes["models"],
        evaluator_hash=hashes["evaluator"],
    )


def test_eval_matrix_has_exact_reviewed_tasks_order_and_seeds() -> None:
    matrix = _eval_matrix()

    assert [episode.split_slot for episode in matrix.episodes] == [
        "eval-s020",
        "eval-s043",
        "eval-s060",
    ]
    assert [episode.protocol_seed for episode in matrix.episodes] == [4309, 4900, 1062]
    assert len({episode.episode_id for episode in matrix.episodes}) == 3


def test_exact_matrix_reports_missing_unknown_duplicates_and_order() -> None:
    matrix = _eval_matrix()
    ids = [episode.episode_id for episode in matrix.episodes]

    complete = compare_exact_matrix(matrix, ids)
    assert complete.complete
    assert not complete.out_of_order

    reordered = compare_exact_matrix(matrix, list(reversed(ids)))
    assert reordered.complete
    assert reordered.out_of_order
    with pytest.raises(CompletenessError):
        require_exact_matrix(matrix, reversed(ids), require_canonical_order=True)

    bad = compare_exact_matrix(matrix, [ids[0], ids[0], "f" * 64])
    assert bad.missing_episode_ids == tuple(ids[1:])
    assert bad.duplicate_episode_ids == (ids[0],)
    assert bad.unknown_episode_ids == ("f" * 64,)
    assert not bad.complete


def test_episode_identity_uses_seed_and_is_repeatable() -> None:
    first = _eval_matrix()
    second = _eval_matrix()

    assert first == second
    assert first.matrix_hash == second.matrix_hash
