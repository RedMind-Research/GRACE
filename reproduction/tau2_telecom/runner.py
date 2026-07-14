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

"""Resumable, sequential Tau2 telecom experiment orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import JsonValue

from .artifacts import (
    Tau2TelecomArtifactLayout,
    Tau2TelecomCheckpoint,
    Tau2TelecomRunContract,
    Tau2TelecomStageArtifact,
)
from .experiment import (
    EvaluationTiming,
    Method,
    Tau2TelecomEvaluationPlan,
    Tau2TelecomExperimentConfig,
)
from .harness.identity import canonical_sha256
from .harness.splits import load_tau2_telecom_task_split
from .provenance import capture_reproduction_provenance


ProgressSink = Callable[[str], None]


@runtime_checkable
class Tau2TelecomWorkflowBackend(Protocol):
    """Injectable boundary between orchestration and model/benchmark execution."""

    def initialize(self, *, method: Method, instruction: str) -> Tau2TelecomCheckpoint: ...

    def experience(self, *, checkpoint: Tau2TelecomCheckpoint, update_index: int) -> JsonValue: ...

    def diagnose(
        self,
        *,
        checkpoint: Tau2TelecomCheckpoint,
        update_index: int,
        experience: JsonValue,
    ) -> JsonValue: ...

    def evolve(
        self,
        *,
        method: Method,
        checkpoint: Tau2TelecomCheckpoint,
        update_index: int,
        diagnosis: JsonValue,
    ) -> Tau2TelecomCheckpoint: ...

    def evaluate(self, *, checkpoint: Tau2TelecomCheckpoint) -> JsonValue: ...


class Tau2TelecomExperimentRunner:
    """Execute checkpoints ℓ0..ℓ10 with exact, create-only resume artifacts."""

    def __init__(
        self,
        *,
        config: Tau2TelecomExperimentConfig,
        method: Method,
        evaluation: Tau2TelecomEvaluationPlan,
        initial_instruction: str,
        artifact_root: str | Path,
        run_id: str,
        backend: Tau2TelecomWorkflowBackend,
        progress: ProgressSink | None = None,
    ) -> None:
        if not initial_instruction.strip():
            raise ValueError("initial_instruction must not be empty")
        self.config = config
        self.method = method
        self.evaluation = evaluation._bounded(config.execution.evolution_updates)
        self.initial_instruction = initial_instruction
        self.backend = backend
        self.artifacts = Tau2TelecomArtifactLayout(artifact_root, run_id)
        self.progress = progress or (lambda _: None)

        selection = load_tau2_telecom_task_split(config.benchmark.task_split)
        selection_hash = canonical_sha256(selection.model_dump(mode="json"))
        provenance = capture_reproduction_provenance()
        self.contract = Tau2TelecomRunContract.create(
            run_id=run_id,
            method=method,
            seed=config.execution.seed,
            evaluation=self.evaluation,
            experiment_config_hash=config.config_hash,
            task_selection_hash=selection_hash,
            initial_instruction=initial_instruction,
            provenance=provenance,
        )
        self.artifacts.bind_run(self.contract)

    def initialize(self) -> Tau2TelecomCheckpoint:
        cached = self.artifacts.load_checkpoint(0)
        if cached is not None:
            self._require_checkpoint(cached, index=0, parent=None)
            self.progress("initialize: checkpoint 0 resumed")
            return cached
        self.progress("initialize: P2G checkpoint 0")
        checkpoint = self.backend.initialize(
            method=self.method, instruction=self.initial_instruction
        )
        self._require_checkpoint(checkpoint, index=0, parent=None)
        self.artifacts.save_checkpoint(checkpoint)
        return checkpoint

    def _parent(self, update_index: int) -> Tau2TelecomCheckpoint:
        if not 1 <= update_index <= self.config.execution.evolution_updates:
            raise ValueError("update_index must be between 1 and 10")
        parent = self.artifacts.load_checkpoint(update_index - 1)
        if parent is None:
            raise ValueError(f"checkpoint {update_index - 1} must exist before update")
        self._require_checkpoint(parent, index=update_index - 1, parent=None, check_parent=False)
        return parent

    def run_batch(self, update_index: int) -> Tau2TelecomStageArtifact:
        """Run or resume only the 42-task experience stage for update ``t``."""

        parent = self._parent(update_index)
        experience_artifact = self.artifacts.load_stage("experience", update_index)
        if experience_artifact is None:
            self.progress(f"update {update_index}: experience")
            payload = self.backend.experience(checkpoint=parent, update_index=update_index)
            experience_artifact = Tau2TelecomStageArtifact.create(
                stage="experience",
                checkpoint_index=parent.checkpoint_index,
                checkpoint_state_id=parent.state_id,
                update_index=update_index,
                payload=payload,
            )
            self.artifacts.save_stage(experience_artifact)
        else:
            self._require_stage(experience_artifact, parent, update_index)
            self.progress(f"update {update_index}: experience resumed")
        return experience_artifact

    def diagnose(self, update_index: int) -> Tau2TelecomStageArtifact:
        """Run or resume Diagnosis after the exact experience batch is complete."""

        parent = self._parent(update_index)
        experience_artifact = self.artifacts.load_stage("experience", update_index)
        if experience_artifact is None:
            raise ValueError(f"experience update {update_index} must exist before diagnosis")
        self._require_stage(experience_artifact, parent, update_index)
        diagnosis_artifact = self.artifacts.load_stage("diagnosis", update_index)
        if diagnosis_artifact is None:
            self.progress(f"update {update_index}: diagnosis")
            payload = self.backend.diagnose(
                checkpoint=parent,
                update_index=update_index,
                experience=experience_artifact.payload,
            )
            diagnosis_artifact = Tau2TelecomStageArtifact.create(
                stage="diagnosis",
                checkpoint_index=parent.checkpoint_index,
                checkpoint_state_id=parent.state_id,
                update_index=update_index,
                payload=payload,
            )
            self.artifacts.save_stage(diagnosis_artifact)
        else:
            self._require_stage(diagnosis_artifact, parent, update_index)
            self.progress(f"update {update_index}: diagnosis resumed")
        return diagnosis_artifact

    def evolve(self, update_index: int) -> Tau2TelecomCheckpoint:
        """Run or resume Evolution and publish checkpoint ``ℓ_t``."""

        parent = self._parent(update_index)
        completed = self.artifacts.load_checkpoint(update_index)
        if completed is not None:
            self._require_checkpoint(completed, index=update_index, parent=parent.state_id)
            self.progress(f"update {update_index}: checkpoint resumed")
            return completed
        diagnosis_artifact = self.artifacts.load_stage("diagnosis", update_index)
        if diagnosis_artifact is None:
            raise ValueError(f"diagnosis update {update_index} must exist before evolution")
        self._require_stage(diagnosis_artifact, parent, update_index)
        self.progress(f"update {update_index}: evolution")
        checkpoint = self.backend.evolve(
            method=self.method,
            checkpoint=parent,
            update_index=update_index,
            diagnosis=diagnosis_artifact.payload,
        )
        self._require_checkpoint(checkpoint, index=update_index, parent=parent.state_id)
        self.artifacts.save_checkpoint(checkpoint)
        return checkpoint

    def evaluate(self, checkpoint_index: int) -> Tau2TelecomStageArtifact:
        if not self.evaluation.includes(checkpoint_index):
            raise ValueError(f"checkpoint {checkpoint_index} is not selected for evaluation")
        checkpoint = self.artifacts.load_checkpoint(checkpoint_index)
        if checkpoint is None:
            raise ValueError(f"checkpoint {checkpoint_index} does not exist")
        cached = self.artifacts.load_stage("evaluation", checkpoint_index)
        if cached is not None:
            self._require_stage(cached, checkpoint, None)
            self.progress(f"evaluation: checkpoint {checkpoint_index} resumed")
            return cached
        self.progress(f"evaluation: checkpoint {checkpoint_index}")
        artifact = Tau2TelecomStageArtifact.create(
            stage="evaluation",
            checkpoint_index=checkpoint_index,
            checkpoint_state_id=checkpoint.state_id,
            payload=self.backend.evaluate(checkpoint=checkpoint),
        )
        self.artifacts.save_stage(artifact)
        return artifact

    def run_all(self) -> tuple[Tau2TelecomCheckpoint, ...]:
        checkpoints = [self.initialize()]
        if self.evaluation.timing == EvaluationTiming.IN_LOOP and self.evaluation.includes(0):
            self.evaluate(0)
        for update_index in range(1, self.config.execution.evolution_updates + 1):
            self.run_batch(update_index)
            self.diagnose(update_index)
            checkpoint = self.evolve(update_index)
            checkpoints.append(checkpoint)
            if self.evaluation.timing == EvaluationTiming.IN_LOOP and self.evaluation.includes(
                update_index
            ):
                self.evaluate(update_index)
        if self.evaluation.timing == EvaluationTiming.OFFLINE:
            for checkpoint_index in self.evaluation.checkpoint_indices:
                self.evaluate(checkpoint_index)
        self.progress("complete: all requested stages passed")
        return tuple(checkpoints)

    def status(self) -> dict[str, Any]:
        checkpoints = tuple(
            index
            for index in range(self.config.execution.evolution_updates + 1)
            if self.artifacts.load_checkpoint(index) is not None
        )
        evaluations = tuple(
            index
            for index in self.evaluation.checkpoint_indices
            if self.artifacts.load_stage("evaluation", index) is not None
        )
        return {
            "run_id": self.contract.run_id,
            "method": self.method,
            "run_fingerprint": self.contract.run_fingerprint,
            "provenance": self.contract.provenance.model_dump(mode="json"),
            "completed_checkpoints": checkpoints,
            "completed_evaluations": evaluations,
            "complete": checkpoints == tuple(range(self.config.execution.evolution_updates + 1))
            and evaluations == self.evaluation.checkpoint_indices,
        }

    def _require_checkpoint(
        self,
        checkpoint: Tau2TelecomCheckpoint,
        *,
        index: int,
        parent: str | None,
        check_parent: bool = True,
    ) -> None:
        if checkpoint.method != self.method or checkpoint.checkpoint_index != index:
            raise ValueError("backend checkpoint method/index violates the run contract")
        if check_parent and checkpoint.parent_state_id != parent:
            raise ValueError("backend checkpoint parent violates the run lineage")

    @staticmethod
    def _require_stage(
        artifact: Tau2TelecomStageArtifact,
        checkpoint: Tau2TelecomCheckpoint,
        update_index: int | None,
    ) -> None:
        if (
            artifact.checkpoint_index != checkpoint.checkpoint_index
            or artifact.checkpoint_state_id != checkpoint.state_id
            or artifact.update_index != update_index
        ):
            raise ValueError("resume stage artifact does not match checkpoint lineage")


__all__ = ["Tau2TelecomExperimentRunner", "Tau2TelecomWorkflowBackend"]
