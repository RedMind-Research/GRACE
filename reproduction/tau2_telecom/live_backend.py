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

"""Concrete provider-backed implementation of the Tau2 telecom workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import JsonValue

from grace import GraceEngine
from grace.config import GraceConfig
from grace.graph.models import GraceState
from grace.providers.base import LLMProvider

from .artifacts import Tau2TelecomCheckpoint
from .baselines.hce.engine import evolve_hce
from .experiment import Method, Tau2TelecomExperimentConfig
from .harness.attempt_ledger import FileProviderAttemptLedger
from .harness.completeness import build_expected_matrix
from .harness.diagnosis import run_diagnosis
from .harness.episode_store import FileEpisodeArtifactAuthority
from .harness.experience import run_experience_collection
from .harness.identity import canonical_sha256
from .harness.metrics import EvaluationObservation, compute_evaluation_metrics
from .harness.models import ExpectedMatrix, TrajectoryRecord
from .harness.splits import Tau2TelecomTaskSplit
from .harness.tau2_interface import (
    EpisodeHarnessConfig,
    Tau2EpisodeExecutor,
    execute_matrix_sequential,
    load_tau2_task_selection,
)


class Tau2TelecomLiveBackend:
    """Run the formal sequential workflow using pinned tau2 and injected models."""

    def __init__(
        self,
        *,
        config: Tau2TelecomExperimentConfig,
        split: Tau2TelecomTaskSplit,
        provider: LLMProvider,
        artifact_root: str | Path,
        run_id: str,
        attempt_ledger: FileProviderAttemptLedger | None = None,
        agent_runtime_args: dict[str, Any] | None = None,
        user_runtime_args: dict[str, Any] | None = None,
    ) -> None:
        method_models = {
            config.models.task_agent,
            config.models.diagnosis,
            config.models.evolution,
            config.models.p2g,
        }
        if method_models != {provider.model}:
            raise ValueError(
                "the task agent, P2G, Diagnosis, and Evolution must use the configured provider"
            )
        if split.benchmark_revision != config.benchmark.revision:
            raise ValueError("task split and experiment config benchmark revisions differ")
        if split.phase_sequence != config.execution.phase_schedule:
            raise ValueError("task split and experiment config phase schedules differ")
        self.config = config
        self.split = split
        self.provider = provider
        self.run_dir = Path(artifact_root).expanduser().resolve() / run_id
        self.run_id = run_id
        self.ledger = attempt_ledger or FileProviderAttemptLedger(
            self.run_dir / "provider_attempts"
        )
        self.executor = Tau2EpisodeExecutor(
            expected_revision=config.benchmark.revision,
            agent_runtime_args=agent_runtime_args,
            user_runtime_args=user_runtime_args,
            attempt_hook=self.ledger,
            agent_descriptor=getattr(provider, "descriptor", None),
        )
        self.episode_config = EpisodeHarnessConfig.create(
            protocol_hash=canonical_sha256(
                {
                    "benchmark_revision": config.benchmark.revision,
                    "domain": "telecom",
                    "namespace": "grace.tau2-telecom-protocol.v1",
                }
            ),
            benchmark_revision=config.benchmark.revision,
            agent_model=config.models.task_agent,
            user_model=config.models.user_simulator,
            model_turn_timeout_seconds=config.episode_runtime.timeout_seconds,
            model_turn_max_tokens=config.episode_runtime.max_tokens,
            max_steps=config.episode_runtime.max_steps,
            max_errors=config.episode_runtime.max_errors,
        )

    def _grace_engine(self, *, method: Method) -> GraceEngine:
        if method not in {"grace", "grace-no-sa"}:
            raise ValueError(f"method {method!r} does not use the GRACE engine")
        method_config = GraceConfig(
            max_output_tokens=self.config.method_calls.max_tokens,
            p2g_max_rounds=self.config.grace.p2g_max_rounds,
            structural_analysis=(self.config.grace.structural_analysis and method != "grace-no-sa"),
            sa_max_rounds=self.config.grace.sa_max_rounds,
            sa_radius_step=self.config.grace.sa_radius_step,
            schema_repair_max_rounds=self.config.grace.schema_repair_max_rounds,
            artifact_dir=self.run_dir / "grace_method",
            artifact_mode=self.config.grace.artifact_mode,
        )
        return GraceEngine(provider=self.provider, config=method_config, run_id=self.run_id)

    @staticmethod
    def _grace_state(checkpoint: Tau2TelecomCheckpoint) -> GraceState:
        try:
            return GraceState.model_validate(checkpoint.method_state["grace_state"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("GRACE checkpoint does not contain a valid GraceState") from exc

    def initialize(self, *, method: Method, instruction: str) -> Tau2TelecomCheckpoint:
        if method in {"grace", "grace-no-sa"}:
            result = self._grace_engine(method=method).initialize(instruction=instruction)
            method_state: dict[str, JsonValue] = {
                "grace_state": result.state.model_dump(mode="json"),
                "result_status": result.status.value,
            }
            output_instruction = result.state.instruction
        else:
            method_state = {"baseline": "hce"}
            output_instruction = instruction
        return Tau2TelecomCheckpoint.create(
            method=method,
            checkpoint_index=0,
            instruction=output_instruction,
            parent_state_id=None,
            method_state=method_state,
        )

    def _episode_inputs(
        self,
        *,
        checkpoint: Tau2TelecomCheckpoint,
        manifest: Any,
        trials: tuple[int, ...],
    ) -> tuple[Any, ExpectedMatrix, str]:
        resolved = load_tau2_task_selection(manifest)
        expected = build_expected_matrix(
            resolved.task_map,
            trials=trials,
            seed=self.config.execution.seed,
            checkpoint_state_id=checkpoint.state_id,
            protocol_hash=self.episode_config.protocol_hash,
            config_hash=self.episode_config.config_hash,
            model_roles_hash=self.episode_config.model_roles_hash,
            evaluator_hash=self.episode_config.evaluator_hash,
        )
        prompt_hash = self.executor.prompt_hash(
            task=resolved.tasks[0],
            policy=checkpoint.instruction,
            agent_model=self.episode_config.agent_model,
            user_model=self.episode_config.user_model,
            model_turn_timeout_seconds=self.episode_config.model_turn_timeout_seconds,
            model_turn_max_tokens=self.episode_config.model_turn_max_tokens,
        )
        return resolved, expected, prompt_hash

    def experience(self, *, checkpoint: Tau2TelecomCheckpoint, update_index: int) -> JsonValue:
        manifest = self.split.experience_manifest(update_index, seed=self.config.execution.seed)
        resolved, expected, prompt_hash = self._episode_inputs(
            checkpoint=checkpoint, manifest=manifest, trials=(0,)
        )
        authority = FileEpisodeArtifactAuthority(
            self.run_dir / "episode_artifacts", run_id=self.run_id
        )
        result = run_experience_collection(
            manifest=manifest,
            expected=expected,
            tasks=resolved.tasks,
            policy=checkpoint.instruction,
            prompt_hash=prompt_hash,
            config=self.episode_config,
            executor=self.executor,
            authority=authority,
        )
        if result.status != "complete":
            raise RuntimeError(f"experience update {update_index} is incomplete")
        return {
            "expected": expected.model_dump(mode="json"),
            "trajectories": [item.model_dump(mode="json") for item in result.trajectories],
            "execution": result.execution.model_dump(mode="json"),
        }

    def diagnose(
        self,
        *,
        checkpoint: Tau2TelecomCheckpoint,
        update_index: int,
        experience: JsonValue,
    ) -> JsonValue:
        if not isinstance(experience, dict):
            raise ValueError("experience artifact must be an object")
        expected = ExpectedMatrix.model_validate(experience.get("expected"))
        raw_trajectories = experience.get("trajectories")
        if not isinstance(raw_trajectories, list):
            raise ValueError("experience artifact does not contain trajectories")
        trajectories = tuple(TrajectoryRecord.model_validate(item) for item in raw_trajectories)
        result = run_diagnosis(
            provider=self.provider,
            expected=expected,
            trajectories=trajectories,
            policy=checkpoint.instruction,
            artifact_dir=self.run_dir / "updates" / str(update_index) / "diagnosis" / "work",
            config_hash=self.config.config_hash,
        )
        if result.status.value not in {"complete", "no_update"}:
            raise RuntimeError(f"diagnosis update {update_index} is {result.status.value}")
        return {
            "status": result.status.value,
            "report": result.report,
            "manifest": result.manifest.model_dump(mode="json"),
        }

    def evolve(
        self,
        *,
        method: Method,
        checkpoint: Tau2TelecomCheckpoint,
        update_index: int,
        diagnosis: JsonValue,
    ) -> Tau2TelecomCheckpoint:
        if not isinstance(diagnosis, dict):
            raise ValueError("diagnosis artifact must be an object")
        report = diagnosis.get("report")
        if not isinstance(report, str) or not report.strip():
            raise ValueError("diagnosis artifact does not contain a report")
        diagnosis_status = diagnosis.get("status")
        if diagnosis_status not in {"complete", "no_update"}:
            raise ValueError("diagnosis artifact status must be complete or no_update")
        instruction: str
        method_state: dict[str, JsonValue]
        if diagnosis_status == "no_update" and method in {"grace", "grace-no-sa"}:
            result = self._grace_engine(method=method).advance_no_update(
                state=self._grace_state(checkpoint),
                diagnosis_report=report,
            )
            instruction = result.state.instruction
            method_state = {
                "grace_state": result.state.model_dump(mode="json"),
                "result_status": result.status.value,
            }
        elif diagnosis_status == "no_update":
            instruction = checkpoint.instruction
            method_state = checkpoint.method_state
        elif method in {"grace", "grace-no-sa"}:
            result = self._grace_engine(method=method).evolve(
                state=self._grace_state(checkpoint), diagnosis_report=report
            )
            instruction = result.state.instruction
            method_state = {
                "grace_state": result.state.model_dump(mode="json"),
                "result_status": result.status.value,
            }
        else:
            hce_result = evolve_hce(
                current_prompt=checkpoint.instruction,
                diagnosis_report=report,
                provider=self.provider,
            )
            instruction = hce_result.evolved_prompt
            method_state = {"hce_result": hce_result.model_dump(mode="json")}
        return Tau2TelecomCheckpoint.create(
            method=method,
            checkpoint_index=update_index,
            instruction=instruction,
            parent_state_id=checkpoint.state_id,
            method_state=method_state,
        )

    def evaluate(self, *, checkpoint: Tau2TelecomCheckpoint) -> JsonValue:
        manifest = self.split.evaluation_manifest(seed=self.config.execution.seed)
        resolved, expected, prompt_hash = self._episode_inputs(
            checkpoint=checkpoint, manifest=manifest, trials=(0, 1, 2)
        )
        authority = FileEpisodeArtifactAuthority(
            self.run_dir / "episode_artifacts", run_id=self.run_id
        )
        execution = execute_matrix_sequential(
            expected=expected,
            tasks=resolved.tasks,
            policy=checkpoint.instruction,
            prompt_hash=prompt_hash,
            config=self.episode_config,
            executor=self.executor,
            authority=authority,
        )
        if execution.status != "complete":
            raise RuntimeError(f"evaluation checkpoint {checkpoint.checkpoint_index} is incomplete")
        observations = tuple(
            EvaluationObservation(
                episode_id=outcome.contract.episode_id,
                benchmark_task_id=outcome.contract.benchmark_task_id,
                trial=outcome.contract.trial,
                success=outcome.success,
            )
            for outcome in execution.accepted
        )
        metrics = compute_evaluation_metrics(expected, observations)
        return {
            "expected": expected.model_dump(mode="json"),
            "execution": execution.model_dump(mode="json"),
            "metrics": metrics.model_dump(mode="json"),
        }


__all__ = ["Tau2TelecomLiveBackend"]
