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

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_offline_quickstart_runs_without_provider_or_network(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "runs"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "offline_quickstart.py"),
            "--artifact-dir",
            str(artifact_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["provider_calls"] == 6
    assert summary["status"] == "completed"
    evolved = Path(summary["evolved_artifact"])
    assert evolved.is_dir()
    assert (evolved / "manifest.json").is_file()
    assert (evolved / "prompt_snapshots.json").is_file()


def test_custom_domain_example_runs_without_provider_or_network(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "custom-runs"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "custom_domain.py"),
            "--artifact-dir",
            str(artifact_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == {
        "artifact": summary["artifact"],
        "provider_calls": 4,
        "schema_id": "incident-response",
        "status": "completed",
        "step": 1,
    }
    evolved = Path(summary["artifact"])
    assert evolved.is_dir()
    assert (evolved / "manifest.json").is_file()
    assert (evolved / "schema.json").is_file() is False
    assert (evolved.parent.parent / "schema.json").is_file()
