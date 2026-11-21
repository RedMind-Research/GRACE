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

import hashlib
import json
from pathlib import Path

import pytest

from reproduction.tau2_telecom.harness.splits import Tau2TelecomTaskSplit
from reproduction.tau2_telecom_split import build


@pytest.fixture
def task_ids() -> list[str]:
    return [
        f"[{intent}]case_{index:03d}[PERSONA:{persona}]"
        for intent in ("mms_issue", "mobile_data_issue", "service_issue")
        for persona in ("None", "Easy", "Hard")
        for index in range(100)
    ]


@pytest.fixture
def tasks_file(tmp_path: Path, task_ids: list[str], monkeypatch: pytest.MonkeyPatch) -> Path:
    data = json.dumps([{"id": task_id} for task_id in task_ids], indent=2).encode("utf-8")
    path = tmp_path / "tasks.json"
    path.write_bytes(data.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(build, "PINNED_TASKS_SHA256", hashlib.sha256(data).hexdigest())
    return path


def test_split_reproduces_original_draw_order(task_ids: list[str]) -> None:
    result = build.build_split(task_ids[::-1])

    assert result == build.build_split(task_ids)
    # Reference from the historical sampler on these same synthetic task IDs.
    sequence = "\n".join(task.task_id for task in result.iter_tasks()).encode("utf-8")
    assert hashlib.sha256(sequence).hexdigest() == (
        "c72114baa4ad5550cab58639afd2860b49d32c11771a66efa2ac4466bb33863e"
    )


def test_writes_verified_split_to_requested_directory(
    tmp_path: Path, task_ids: list[str], tasks_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = build.build_split(task_ids)
    monkeypatch.setattr(build, "load_tau2_telecom_task_split", lambda: expected)
    output_dir = tmp_path / "split_rebuild"
    args = ["--tasks-json", str(tasks_file), "--output-dir", str(output_dir)]

    assert build.main(args) == 0
    output = output_dir / "rebuilt-splits.json"
    first_write = output.read_bytes()
    assert Tau2TelecomTaskSplit.model_validate_json(first_write) == expected
    assert build.main(args) == 0
    assert output.read_bytes() == first_write
    assert list(output_dir.iterdir()) == [output]


def test_rejects_wrong_task_file_before_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text("[]", encoding="utf-8")
    output_dir = tmp_path / "split_rebuild"

    with pytest.raises(SystemExit) as error:
        build.main(["--tasks-json", str(tasks_file), "--output-dir", str(output_dir)])

    assert error.value.code == 1
    assert "Expected the telecom tasks.json from Tau² revision" in capsys.readouterr().err
    assert not output_dir.exists()


def test_rejects_a_split_that_differs_from_the_frozen_tasks(
    tmp_path: Path, tasks_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "split_rebuild"

    with pytest.raises(SystemExit) as error:
        build.main(["--tasks-json", str(tasks_file), "--output-dir", str(output_dir)])

    assert error.value.code == 1
    assert "differ from the frozen split" in capsys.readouterr().err
    assert not output_dir.exists()
