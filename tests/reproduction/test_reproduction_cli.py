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
import subprocess
import sys

from pytest import CaptureFixture
import pytest

from reproduction.tau2_telecom.artifacts import (
    Tau2TelecomArtifactLayout,
    Tau2TelecomRunContract,
)
from reproduction.tau2_telecom.cli import EXIT_INVALID, EXIT_OK, main
from reproduction.tau2_telecom.experiment import Tau2TelecomEvaluationPlan
from reproduction.tau2_telecom.harness.identity import canonical_json
from reproduction.tau2_telecom.provenance import (
    Tau2TelecomRuntimeProvenance,
    Tau2TelecomSourceProvenance,
)


ROOT = Path(__file__).parents[2]


def test_repository_module_entrypoint_runs_offline() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reproduction.tau2_telecom",
            "plan",
            "--evaluation",
            "offline:0,6,8,10",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert '"total_episodes":1212' in result.stdout


def test_validate_full_split_is_offline(capsys: CaptureFixture[str]) -> None:
    assert main(["validate-split"]) == EXIT_OK
    output = capsys.readouterr().out
    assert '"evaluation_tasks":66' in output
    assert '"seed":1024' in output


def test_evaluation_parser_fails_before_provider_setup(capsys: CaptureFixture[str]) -> None:
    result = main(["run-all", "--evaluation", "deferred:all"])
    assert result == EXIT_INVALID
    assert "evaluation must use" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("evaluation", "checkpoints", "evaluation_episodes", "total_episodes"),
    [
        ("in-loop:all", list(range(11)), 2178, 2598),
        ("offline:0,6,8,10", [0, 6, 8, 10], 792, 1212),
    ],
)
def test_plan_is_offline_and_emits_canonical_episode_counts(
    evaluation: str,
    checkpoints: list[int],
    evaluation_episodes: int,
    total_episodes: int,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def provider_setup_forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("plan must not initialize a provider")

    monkeypatch.setattr(
        "reproduction.tau2_telecom.cli._provider_and_runtime",
        provider_setup_forbidden,
    )

    assert main(["plan", "--evaluation", evaluation]) == EXIT_OK
    output = capsys.readouterr().out
    expected = {
        "evaluation_checkpoints": checkpoints,
        "evaluation_episodes": evaluation_episodes,
        "evaluation_timing": evaluation.split(":", 1)[0],
        "experience_episodes": 420,
        "total_episodes": total_episodes,
    }
    assert output == canonical_json(expected) + "\n"


def test_status_is_offline_and_does_not_require_provider_environment(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    run_id = "offline-status"
    layout = Tau2TelecomArtifactLayout(tmp_path, run_id)
    layout.bind_run(
        Tau2TelecomRunContract.create(
            run_id=run_id,
            method="grace",
            seed=1024,
            evaluation=Tau2TelecomEvaluationPlan.parse("in-loop:all"),
            experiment_config_hash="a" * 64,
            task_selection_hash="b" * 64,
            initial_instruction="instruction",
            provenance=Tau2TelecomSourceProvenance(
                core_version="0.1.0",
                core_source_hash="d" * 64,
                reproduction_source_hash="c" * 64,
                runtime=Tau2TelecomRuntimeProvenance.create(
                    python_implementation="CPython",
                    python_version="3.13.2",
                    dependencies=(),
                ),
            ),
        )
    )

    assert (
        main(
            [
                "status",
                "--artifacts",
                str(tmp_path),
                "--run-id",
                run_id,
                "--evaluation",
                "in-loop:all",
            ]
        )
        == EXIT_OK
    )
    assert '"complete":false' in capsys.readouterr().out
