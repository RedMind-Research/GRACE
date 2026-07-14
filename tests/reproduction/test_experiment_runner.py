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

from pathlib import Path
from typing import Any

import pytest

from reproduction.tau2_telecom.artifacts import Tau2TelecomCheckpoint
from reproduction.tau2_telecom.experiment import (
    Method,
    Tau2TelecomEvaluationPlan,
    Tau2TelecomExperimentConfig,
)
from reproduction.tau2_telecom.runner import Tau2TelecomExperimentRunner
from grace import __version__ as grace_version


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "reproduction/tau2_telecom/configs/reproduction.yaml"


class ScriptedBackend:
    def __init__(self, fail_once_at: tuple[str, int] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.fail_once_at = fail_once_at

    def _call(self, stage: str, index: int) -> None:
        self.calls.append((stage, index))
        if self.fail_once_at == (stage, index):
            self.fail_once_at = None
            raise RuntimeError("scripted crash")

    def initialize(self, *, method: Method, instruction: str) -> Tau2TelecomCheckpoint:
        self._call("initialize", 0)
        return Tau2TelecomCheckpoint.create(
            method=method,
            checkpoint_index=0,
            instruction=instruction,
            parent_state_id=None,
            method_state={"graph": {"nodes": [], "edges": []}}
            if method in {"grace", "grace-no-sa"}
            else {},
        )

    def experience(self, *, checkpoint: Tau2TelecomCheckpoint, update_index: int) -> Any:
        self._call("experience", update_index)
        return {"trajectory_count": 42, "state": checkpoint.state_id}

    def diagnose(
        self,
        *,
        checkpoint: Tau2TelecomCheckpoint,
        update_index: int,
        experience: Any,
    ) -> Any:
        self._call("diagnosis", update_index)
        assert experience["trajectory_count"] == 42
        return {"report": f"diagnosis-{update_index}"}

    def evolve(
        self,
        *,
        method: Method,
        checkpoint: Tau2TelecomCheckpoint,
        update_index: int,
        diagnosis: Any,
    ) -> Tau2TelecomCheckpoint:
        self._call("evolution", update_index)
        return Tau2TelecomCheckpoint.create(
            method=method,
            checkpoint_index=update_index,
            instruction=f"{checkpoint.instruction}\n{diagnosis['report']}",
            parent_state_id=checkpoint.state_id,
            method_state={"update": update_index},
        )

    def evaluate(self, *, checkpoint: Tau2TelecomCheckpoint) -> Any:
        self._call("evaluation", checkpoint.checkpoint_index)
        return {"task_count": 66, "trial_count": 3}


def make_runner(
    tmp_path: Path,
    backend: ScriptedBackend,
    *,
    method: Method = "grace",
    evaluation: str = "in-loop:0,6,8,10",
    run_id: str | None = None,
) -> Tau2TelecomExperimentRunner:
    return Tau2TelecomExperimentRunner(
        config=Tau2TelecomExperimentConfig.load(CONFIG),
        method=method,
        evaluation=Tau2TelecomEvaluationPlan.parse(evaluation),
        initial_instruction="initial telecom instruction",
        artifact_root=tmp_path,
        run_id=run_id or f"test-{method}",
        backend=backend,
    )


@pytest.mark.parametrize("method", ["grace", "grace-no-sa", "hce"])
def test_complete_ten_update_run_is_sequential_and_exact(tmp_path: Path, method: Method) -> None:
    backend = ScriptedBackend()
    runner = make_runner(tmp_path, backend, method=method)

    checkpoints = runner.run_all()

    assert tuple(item.checkpoint_index for item in checkpoints) == tuple(range(11))
    assert [item for item in backend.calls if item[0] == "experience"] == [
        ("experience", index) for index in range(1, 11)
    ]
    assert [item for item in backend.calls if item[0] == "evaluation"] == [
        ("evaluation", index) for index in (0, 6, 8, 10)
    ]
    assert runner.status()["complete"] is True


def test_offline_evaluation_waits_until_all_updates_finish(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    runner = make_runner(tmp_path, backend, evaluation="offline:0,10")

    runner.run_all()

    last_evolution = max(i for i, call in enumerate(backend.calls) if call[0] == "evolution")
    first_evaluation = min(i for i, call in enumerate(backend.calls) if call[0] == "evaluation")
    assert first_evaluation > last_evolution


def test_crash_resume_reuses_completed_stages_without_reexecution(tmp_path: Path) -> None:
    first = ScriptedBackend(fail_once_at=("evolution", 5))
    runner = make_runner(tmp_path, first)
    with pytest.raises(RuntimeError, match="scripted crash"):
        runner.run_all()

    second = ScriptedBackend()
    resumed = make_runner(tmp_path, second)
    resumed.run_all()

    assert ("initialize", 0) not in second.calls
    assert all(("experience", index) not in second.calls for index in range(1, 6))
    assert all(("diagnosis", index) not in second.calls for index in range(1, 6))
    assert second.calls[0] == ("evolution", 5)


def test_changed_evaluation_contract_cannot_reuse_run(tmp_path: Path) -> None:
    make_runner(tmp_path, ScriptedBackend()).initialize()
    with pytest.raises(ValueError, match="conflicts"):
        make_runner(tmp_path, ScriptedBackend(), evaluation="offline:all")


def test_run_contract_records_exact_software_provenance(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, ScriptedBackend())

    assert runner.contract.provenance.core_version == grace_version
    assert len(runner.contract.provenance.core_source_hash) == 64
    assert len(runner.contract.provenance.reproduction_source_hash) == 64
    assert runner.contract.provenance.runtime.python_version
    assert len(runner.contract.provenance.runtime.runtime_hash) == 64


def test_grace_no_sa_cannot_resume_grace_artifacts(tmp_path: Path) -> None:
    run_id = "method-isolation"
    grace = make_runner(tmp_path, ScriptedBackend(), method="grace", run_id=run_id)
    grace.initialize()

    with pytest.raises(ValueError, match="conflicts"):
        make_runner(
            tmp_path,
            ScriptedBackend(),
            method="grace-no-sa",
            run_id=run_id,
        )


def test_run_id_cannot_escape_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        Tau2TelecomExperimentRunner(
            config=Tau2TelecomExperimentConfig.load(CONFIG),
            method="grace",
            evaluation=Tau2TelecomEvaluationPlan.parse("in-loop:all"),
            initial_instruction="instruction",
            artifact_root=tmp_path,
            run_id="../escape",
            backend=ScriptedBackend(),
        )
