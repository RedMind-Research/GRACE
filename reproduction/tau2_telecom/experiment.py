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

"""Frozen public contracts for the Tau2 telecom experiment workflow."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .harness.identity import canonical_sha256


Frozen = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1)]
Method = Literal["grace", "grace-no-sa", "hce"]


class EvaluationTiming(str, Enum):
    IN_LOOP = "in-loop"
    OFFLINE = "offline"


class Tau2TelecomEvaluationPlan(BaseModel):
    """When evaluation executes and which formal checkpoints it covers."""

    model_config = Frozen

    timing: EvaluationTiming
    checkpoint_indices: tuple[int, ...]

    @classmethod
    def parse(cls, value: str, *, final_checkpoint: int = 10) -> Tau2TelecomEvaluationPlan:
        try:
            timing_text, selection = value.split(":", 1)
            timing = EvaluationTiming(timing_text)
        except (ValueError, AttributeError):
            raise ValueError("evaluation must use '<in-loop|offline>:<all|indices>'") from None
        if selection == "all":
            indices = tuple(range(final_checkpoint + 1))
        else:
            if not selection or any(not part.isdecimal() for part in selection.split(",")):
                raise ValueError("evaluation checkpoints must be 'all' or comma-separated integers")
            indices = tuple(int(part) for part in selection.split(","))
        return cls(timing=timing, checkpoint_indices=indices)._bounded(final_checkpoint)

    def _bounded(self, final_checkpoint: int) -> Tau2TelecomEvaluationPlan:
        if not self.checkpoint_indices:
            raise ValueError("evaluation must select at least one checkpoint")
        if self.checkpoint_indices != tuple(sorted(set(self.checkpoint_indices))):
            raise ValueError("evaluation checkpoint indices must be unique and increasing")
        if self.checkpoint_indices[0] < 0 or self.checkpoint_indices[-1] > final_checkpoint:
            raise ValueError(f"evaluation checkpoints must be between 0 and {final_checkpoint}")
        return self

    def includes(self, checkpoint_index: int) -> bool:
        return checkpoint_index in self.checkpoint_indices

    def __str__(self) -> str:
        return f"{self.timing.value}:{','.join(str(i) for i in self.checkpoint_indices)}"


class BenchmarkConfig(BaseModel):
    model_config = Frozen
    name: Literal["tau2"] = "tau2"
    domain: Literal["telecom"] = "telecom"
    revision: NonEmpty
    task_split: Path


class ExecutionConfig(BaseModel):
    model_config = Frozen
    seed: int = 1024
    workers: Literal[1] = 1
    evolution_updates: Literal[10] = 10
    phase_schedule: tuple[Literal["A", "B"], ...]
    experience_tasks_per_update: Literal[42] = 42
    evaluation_tasks: Literal[66] = 66
    evaluation_trials: Literal[3] = 3

    @model_validator(mode="after")
    def validate_schedule(self) -> ExecutionConfig:
        if self.phase_schedule != tuple("AABBAABBAA"):
            raise ValueError("the released phase schedule must be AABBAABBAA")
        return self


class ModelConfig(BaseModel):
    model_config = Frozen
    task_agent: NonEmpty
    diagnosis: NonEmpty
    evolution: NonEmpty
    p2g: NonEmpty
    user_simulator: Literal["gpt-4.1-2025-04-14"]


class MethodConfig(BaseModel):
    model_config = Frozen
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    max_tokens: Literal[65536] = 65536


class EpisodeRuntimeConfig(BaseModel):
    model_config = Frozen
    max_tokens: int = Field(default=8192, ge=1)
    timeout_seconds: int = Field(default=300, ge=1)
    max_steps: int = Field(default=200, ge=1)
    max_errors: int = Field(default=10, ge=1)


class GraceMethodConfig(BaseModel):
    model_config = Frozen
    p2g_max_rounds: Literal[10] = 10
    structural_analysis: Literal[True] = True
    sa_max_rounds: Literal[3] = 3
    sa_radius_step: Literal[3] = 3
    schema_repair_max_rounds: Literal[3] = 3
    artifact_mode: Literal["checkpoint", "audit"] = "checkpoint"


class Tau2TelecomExperimentConfig(BaseModel):
    """Single source of truth for the released paper-reproduction recipe."""

    model_config = Frozen

    schema_version: Literal["grace.tau2-telecom-experiment.v1"]
    benchmark: BenchmarkConfig
    execution: ExecutionConfig
    models: ModelConfig
    method_calls: MethodConfig
    episode_runtime: EpisodeRuntimeConfig
    grace: GraceMethodConfig

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        # Filesystem location is deployment metadata. The selected manifest's
        # canonical content is bound separately by Tau2TelecomRunContract.
        payload["benchmark"].pop("task_split", None)
        return canonical_sha256({"namespace": self.schema_version, "config": payload})

    @classmethod
    def load(cls, path: str | Path) -> Tau2TelecomExperimentConfig:
        config_path = Path(path).expanduser().resolve()
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read experiment config {config_path}: {exc}") from exc
        config = cls.model_validate(payload)
        split = config.benchmark.task_split
        if not split.is_absolute():
            split = (config_path.parent / split).resolve()
        return config.model_copy(
            update={"benchmark": config.benchmark.model_copy(update={"task_split": split})}
        )


class Tau2TelecomEpisodePlan(BaseModel):
    """Deterministic benchmark-episode counts for one requested run."""

    model_config = Frozen

    experience_episodes: int = Field(ge=0)
    evaluation_timing: EvaluationTiming
    evaluation_checkpoints: tuple[int, ...] = Field(min_length=1)
    evaluation_episodes: int = Field(ge=0)
    total_episodes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Tau2TelecomEpisodePlan:
        if self.evaluation_checkpoints != tuple(sorted(set(self.evaluation_checkpoints))):
            raise ValueError("evaluation checkpoint indices must be unique and increasing")
        if self.evaluation_checkpoints[0] < 0:
            raise ValueError("evaluation checkpoint indices must be non-negative")
        if self.total_episodes != self.experience_episodes + self.evaluation_episodes:
            raise ValueError("total_episodes must equal experience plus evaluation episodes")
        return self

    @classmethod
    def create(
        cls,
        *,
        config: Tau2TelecomExperimentConfig,
        evaluation: Tau2TelecomEvaluationPlan,
    ) -> Tau2TelecomEpisodePlan:
        bounded = evaluation._bounded(config.execution.evolution_updates)
        experience_episodes = (
            config.execution.evolution_updates * config.execution.experience_tasks_per_update
        )
        evaluation_episodes = (
            len(bounded.checkpoint_indices)
            * config.execution.evaluation_tasks
            * config.execution.evaluation_trials
        )
        return cls(
            experience_episodes=experience_episodes,
            evaluation_timing=bounded.timing,
            evaluation_checkpoints=bounded.checkpoint_indices,
            evaluation_episodes=evaluation_episodes,
            total_episodes=experience_episodes + evaluation_episodes,
        )


__all__ = [
    "EvaluationTiming",
    "Method",
    "Tau2TelecomEpisodePlan",
    "Tau2TelecomEvaluationPlan",
    "Tau2TelecomExperimentConfig",
]
