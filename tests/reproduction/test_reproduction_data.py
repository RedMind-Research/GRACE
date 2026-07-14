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

from reproduction.tau2_telecom.harness.tasks import load_task_selection_manifest


DATA_DIR = Path(__file__).parents[2] / "reproduction/tau2_telecom/data"


def test_smoke_manifests_freeze_exact_selected_slots() -> None:
    experience = load_task_selection_manifest(DATA_DIR / "experience_smoke_v1.json")
    evaluation = load_task_selection_manifest(DATA_DIR / "eval_smoke_v1.json")

    assert [task.split_slot for task in experience.tasks] == [
        "exp-b01-s028",
        "exp-b01-s036",
        "exp-b01-s040",
    ]
    assert [task.split_slot for task in evaluation.tasks] == [
        "eval-s020",
        "eval-s043",
        "eval-s060",
    ]
    assert experience.benchmark_revision == evaluation.benchmark_revision
