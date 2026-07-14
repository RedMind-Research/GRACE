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

"""Lazy tau2 adapter and sequential episode execution contracts.

No tau2 module is imported until a caller explicitly loads tasks, obtains the
telecom policy, or executes a real episode.  Tests and dry runs inject an
``EpisodeExecutor`` and therefore require neither tau2 nor network access.

The paper identifies the ``gpt-4.1`` model family. The public recipe pins the
official ``gpt-4.1-2025-04-14`` snapshot as a new reproducibility hardening
choice.
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from grace.errors import ProviderConfigurationError
from grace.providers.base import (
    ProviderAttemptContext,
    ProviderAttemptHook,
    ProviderAttemptRecord,
    UsageRecord,
)
from grace.providers.config import ProviderDescriptor

from .completeness import compare_exact_matrix
from .identity import (
    PINNED_TAU2_REVISION,
    canonical_json,
    canonical_sha256,
    make_episode_id,
    make_task_uid,
    text_sha256,
    validate_sha256,
)
from .models import (
    CompletenessReport,
    EpisodeCell,
    ExpectedMatrix,
    TaskSelectionManifest,
    TrajectoryMessage,
    TrajectoryRecord,
    TrajectoryToolCall,
)
from .tasks import (
    ResolvedTaskSelection,
    load_tasks_in_manifest_order,
    resolve_task_selection,
)


FrozenModelConfig = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1)]
CANONICAL_TAU2_USER_MODEL: Final[Literal["gpt-4.1-2025-04-14"]] = "gpt-4.1-2025-04-14"
LEGACY_TAU2_USER_MODEL_ALIAS: Final = "gpt-4.1"
CANONICAL_TAU2_MODEL_TURN_MAX_INPUT_BOUND: Final[Literal[1_000_000]] = 1_000_000
EpisodeDisposition = Literal["behavioral_success", "behavioral_failure", "infrastructure_error"]


class Tau2HarnessError(RuntimeError):
    """Base exception for reproduction episode execution."""


class Tau2UnavailableError(Tau2HarnessError):
    """Raised when the optional pinned tau2 dependency cannot be imported."""


class Tau2ProviderHookError(Tau2HarnessError):
    """Raised when the per-turn persistence/budget boundary cannot be proven."""


class Tau2RevisionError(Tau2HarnessError):
    """Raised when the installed tau2 revision cannot satisfy the manifest."""


class EpisodeContractError(Tau2HarnessError):
    """Raised when task, matrix, attempt, or outcome identities disagree."""


class EpisodeAuthorityError(Tau2HarnessError):
    """Raised when immutable authority data is contradictory or incomplete."""


class EpisodeHarnessConfig(BaseModel):
    """Serializable canonical v1 episode profile, excluding credentials."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-harness-config.v1"] = "grace.episode-harness-config.v1"
    benchmark_revision: NonEmpty = PINNED_TAU2_REVISION
    domain: Literal["telecom"] = "telecom"
    agent_model: NonEmpty = "vertex_ai/gemini-2.5-flash"
    user_model: Literal["gpt-4.1-2025-04-14"] = CANONICAL_TAU2_USER_MODEL
    workers: Literal[1] = 1
    episode_max_attempts: Literal[2] = 2
    tau2_inner_num_retries: Literal[0] = 0
    model_turn_timeout_seconds: int = Field(default=300, ge=1)
    model_turn_max_tokens: int = Field(default=8192, ge=1)
    model_turn_max_input_bound: Literal[1_000_000] = CANONICAL_TAU2_MODEL_TURN_MAX_INPUT_BOUND
    require_per_turn_attempt_hook: Literal[True] = True
    max_steps: int = Field(default=200, ge=1)
    max_errors: int = Field(default=10, ge=1)
    retry_seed_stride: Literal[7] = 7
    attempt_agent_temperatures: tuple[float, float] = (0.0, 0.5)
    user_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    enforce_communication_protocol: bool = False
    evaluation_type: Literal["all"] = "all"
    protocol_hash: NonEmpty
    model_roles_hash: NonEmpty
    evaluator_hash: NonEmpty
    config_hash: NonEmpty

    @staticmethod
    def _model_roles_payload(agent_model: str, user_model: str) -> dict[str, Any]:
        return {
            "agent_model": agent_model,
            "namespace": "grace.tau2-model-roles.v1",
            "tau2_inner_num_retries": 0,
            "user_model": user_model,
        }

    @staticmethod
    def _evaluator_payload(benchmark_revision: str, domain: str) -> dict[str, Any]:
        return {
            "benchmark_revision": benchmark_revision,
            "domain": domain,
            "evaluation_type": "all",
            "judge_llm": None,
            "namespace": "grace.tau2-evaluator.v1",
        }

    @staticmethod
    def _config_payload(
        *,
        model_turn_timeout_seconds: int,
        model_turn_max_tokens: int,
        model_turn_max_input_bound: Literal[1_000_000],
        require_per_turn_attempt_hook: Literal[True],
        max_steps: int,
        max_errors: int,
        retry_seed_stride: int,
        attempt_agent_temperatures: tuple[float, float],
        user_temperature: float,
        enforce_communication_protocol: bool,
    ) -> dict[str, Any]:
        return {
            "attempt_agent_temperatures": list(attempt_agent_temperatures),
            "enforce_communication_protocol": enforce_communication_protocol,
            "episode_max_attempts": 2,
            "max_errors": max_errors,
            "max_steps": max_steps,
            "model_turn_timeout_seconds": model_turn_timeout_seconds,
            "model_turn_max_input_bound": model_turn_max_input_bound,
            "model_turn_max_tokens": model_turn_max_tokens,
            "require_per_turn_attempt_hook": require_per_turn_attempt_hook,
            "namespace": "grace.episode-harness-runtime.v1",
            "retry_seed_stride": retry_seed_stride,
            "tau2_inner_num_retries": 0,
            "user_temperature": user_temperature,
            "workers": 1,
        }

    @classmethod
    def create(
        cls,
        *,
        protocol_hash: str,
        benchmark_revision: str = PINNED_TAU2_REVISION,
        domain: Literal["telecom"] = "telecom",
        agent_model: str = "vertex_ai/gemini-2.5-flash",
        user_model: Literal["gpt-4.1-2025-04-14"] = CANONICAL_TAU2_USER_MODEL,
        model_turn_timeout_seconds: int = 300,
        model_turn_max_tokens: int = 8192,
        model_turn_max_input_bound: Literal[1_000_000] = CANONICAL_TAU2_MODEL_TURN_MAX_INPUT_BOUND,
        require_per_turn_attempt_hook: Literal[True] = True,
        max_steps: int = 200,
        max_errors: int = 10,
        retry_seed_stride: Literal[7] = 7,
        attempt_agent_temperatures: tuple[float, float] = (0.0, 0.5),
        user_temperature: float = 0.0,
        enforce_communication_protocol: bool = False,
    ) -> EpisodeHarnessConfig:
        model_roles_hash = canonical_sha256(cls._model_roles_payload(agent_model, user_model))
        evaluator_hash = canonical_sha256(cls._evaluator_payload(benchmark_revision, domain))
        config_hash = canonical_sha256(
            cls._config_payload(
                model_turn_timeout_seconds=model_turn_timeout_seconds,
                model_turn_max_tokens=model_turn_max_tokens,
                model_turn_max_input_bound=model_turn_max_input_bound,
                require_per_turn_attempt_hook=require_per_turn_attempt_hook,
                max_steps=max_steps,
                max_errors=max_errors,
                retry_seed_stride=retry_seed_stride,
                attempt_agent_temperatures=attempt_agent_temperatures,
                user_temperature=user_temperature,
                enforce_communication_protocol=enforce_communication_protocol,
            )
        )
        return cls(
            benchmark_revision=benchmark_revision,
            domain=domain,
            agent_model=agent_model,
            user_model=user_model,
            model_turn_timeout_seconds=model_turn_timeout_seconds,
            model_turn_max_tokens=model_turn_max_tokens,
            model_turn_max_input_bound=model_turn_max_input_bound,
            require_per_turn_attempt_hook=require_per_turn_attempt_hook,
            max_steps=max_steps,
            max_errors=max_errors,
            retry_seed_stride=retry_seed_stride,
            attempt_agent_temperatures=attempt_agent_temperatures,
            user_temperature=user_temperature,
            enforce_communication_protocol=enforce_communication_protocol,
            protocol_hash=protocol_hash,
            model_roles_hash=model_roles_hash,
            evaluator_hash=evaluator_hash,
            config_hash=config_hash,
        )

    @model_validator(mode="after")
    def validate_profile_hashes(self) -> EpisodeHarnessConfig:
        validate_sha256(self.protocol_hash, field="protocol_hash")
        expected_model_roles = canonical_sha256(
            self._model_roles_payload(self.agent_model, self.user_model)
        )
        expected_evaluator = canonical_sha256(
            self._evaluator_payload(self.benchmark_revision, self.domain)
        )
        expected_config = canonical_sha256(
            self._config_payload(
                model_turn_timeout_seconds=self.model_turn_timeout_seconds,
                model_turn_max_tokens=self.model_turn_max_tokens,
                model_turn_max_input_bound=self.model_turn_max_input_bound,
                require_per_turn_attempt_hook=self.require_per_turn_attempt_hook,
                max_steps=self.max_steps,
                max_errors=self.max_errors,
                retry_seed_stride=self.retry_seed_stride,
                attempt_agent_temperatures=self.attempt_agent_temperatures,
                user_temperature=self.user_temperature,
                enforce_communication_protocol=self.enforce_communication_protocol,
            )
        )
        if self.model_roles_hash != expected_model_roles:
            raise ValueError("model_roles_hash does not match the configured roles")
        if self.evaluator_hash != expected_evaluator:
            raise ValueError("evaluator_hash does not match deterministic tau2 ALL evaluation")
        if self.config_hash != expected_config:
            raise ValueError("config_hash does not match the canonical episode profile")
        if self.attempt_agent_temperatures != (0.0, 0.5):
            raise ValueError("episode attempt temperatures must be exactly (0.0, 0.5)")
        if self.user_temperature != 0.0:
            raise ValueError("the tau2 user simulator temperature must be zero")
        return self


class EpisodeAttemptContract(BaseModel):
    """Identity-bound input for one physical episode attempt."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-attempt.v1"] = "grace.episode-attempt.v1"
    attempt_id: NonEmpty
    contract_hash: NonEmpty
    episode_id: NonEmpty
    task_uid: NonEmpty
    task_definition_hash: NonEmpty
    benchmark_task_id: NonEmpty
    split_slot: NonEmpty
    trial: int = Field(ge=0)
    protocol_seed: int
    attempt_index: int = Field(ge=0, le=1)
    attempt_seed: int
    agent_temperature: float = Field(ge=0.0)
    user_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    checkpoint_state_id: NonEmpty
    benchmark_revision: NonEmpty
    policy_hash: NonEmpty
    prompt_hash: NonEmpty
    protocol_hash: NonEmpty
    model_roles_hash: NonEmpty
    evaluator_hash: NonEmpty
    config_hash: NonEmpty
    agent_model: NonEmpty
    user_model: NonEmpty
    tau2_inner_num_retries: Literal[0] = 0
    model_turn_timeout_seconds: int = Field(ge=1)
    model_turn_max_tokens: int = Field(ge=1)
    model_turn_max_input_bound: Literal[1_000_000]
    require_per_turn_attempt_hook: Literal[True]
    max_steps: int = Field(ge=1)
    max_errors: int = Field(ge=1)
    enforce_communication_protocol: bool = False

    @staticmethod
    def _payload(values: Mapping[str, Any]) -> dict[str, Any]:
        excluded = {"attempt_id", "contract_hash", "schema_version"}
        return {
            "namespace": "grace.episode-attempt-contract.v1",
            **{key: value for key, value in values.items() if key not in excluded},
        }

    @classmethod
    def create(
        cls,
        *,
        episode: EpisodeCell,
        attempt_index: int,
        config: EpisodeHarnessConfig,
        policy_hash: str,
        prompt_hash: str,
    ) -> EpisodeAttemptContract:
        if attempt_index not in (0, 1):
            raise EpisodeContractError("episode attempt index must be zero or one")
        values: dict[str, Any] = {
            "episode_id": episode.episode_id,
            "task_uid": episode.task_uid,
            "task_definition_hash": episode.task_definition_hash,
            "benchmark_task_id": episode.benchmark_task_id,
            "split_slot": episode.split_slot,
            "trial": episode.trial,
            "protocol_seed": episode.protocol_seed,
            "attempt_index": attempt_index,
            "attempt_seed": episode.protocol_seed + attempt_index * config.retry_seed_stride,
            "agent_temperature": config.attempt_agent_temperatures[attempt_index],
            "user_temperature": config.user_temperature,
            "checkpoint_state_id": episode.checkpoint_state_id,
            "benchmark_revision": config.benchmark_revision,
            "policy_hash": policy_hash,
            "prompt_hash": prompt_hash,
            "protocol_hash": config.protocol_hash,
            "model_roles_hash": config.model_roles_hash,
            "evaluator_hash": config.evaluator_hash,
            "config_hash": config.config_hash,
            "agent_model": config.agent_model,
            "user_model": config.user_model,
            "tau2_inner_num_retries": config.tau2_inner_num_retries,
            "model_turn_timeout_seconds": config.model_turn_timeout_seconds,
            "model_turn_max_tokens": config.model_turn_max_tokens,
            "model_turn_max_input_bound": config.model_turn_max_input_bound,
            "require_per_turn_attempt_hook": config.require_per_turn_attempt_hook,
            "max_steps": config.max_steps,
            "max_errors": config.max_errors,
            "enforce_communication_protocol": config.enforce_communication_protocol,
        }
        contract_hash = canonical_sha256(cls._payload(values))
        attempt_id = canonical_sha256(
            {
                "attempt_index": attempt_index,
                "attempt_seed": values["attempt_seed"],
                "contract_hash": contract_hash,
                "episode_id": episode.episode_id,
                "namespace": "grace.episode-attempt-id.v1",
            }
        )
        return cls(attempt_id=attempt_id, contract_hash=contract_hash, **values)

    @model_validator(mode="after")
    def validate_contract(self) -> EpisodeAttemptContract:
        for name, value in {
            "attempt_id": self.attempt_id,
            "contract_hash": self.contract_hash,
            "episode_id": self.episode_id,
            "task_uid": self.task_uid,
            "task_definition_hash": self.task_definition_hash,
            "policy_hash": self.policy_hash,
            "prompt_hash": self.prompt_hash,
            "protocol_hash": self.protocol_hash,
            "model_roles_hash": self.model_roles_hash,
            "evaluator_hash": self.evaluator_hash,
            "config_hash": self.config_hash,
        }.items():
            validate_sha256(value, field=name)
        payload = self.model_dump(mode="json")
        if self.contract_hash != canonical_sha256(self._payload(payload)):
            raise ValueError("contract_hash does not match attempt inputs")
        expected_attempt_id = canonical_sha256(
            {
                "attempt_index": self.attempt_index,
                "attempt_seed": self.attempt_seed,
                "contract_hash": self.contract_hash,
                "episode_id": self.episode_id,
                "namespace": "grace.episode-attempt-id.v1",
            }
        )
        if self.attempt_id != expected_attempt_id:
            raise ValueError("attempt_id does not match the canonical attempt preimage")
        expected_temperature = (0.0, 0.5)[self.attempt_index]
        if self.agent_temperature != expected_temperature:
            raise ValueError("agent temperature does not match the canonical episode attempt")
        if self.user_temperature != 0.0:
            raise ValueError("the tau2 user simulator temperature must be zero")
        return self


class EpisodeAccounting(BaseModel):
    """Episode-level accounting derived from tau2 messages, never fabricated."""

    model_config = FrozenModelConfig

    agent_input_tokens: int | None = Field(default=None, ge=0)
    agent_output_tokens: int | None = Field(default=None, ge=0)
    user_input_tokens: int | None = Field(default=None, ge=0)
    user_output_tokens: int | None = Field(default=None, ge=0)
    agent_cost_usd: float | None = Field(default=None, ge=0.0)
    user_cost_usd: float | None = Field(default=None, ge=0.0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    agent_model_turns: int = Field(default=0, ge=0)
    user_model_turns: int = Field(default=0, ge=0)
    model_turns: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_turn_total(self) -> EpisodeAccounting:
        if self.model_turns != self.agent_model_turns + self.user_model_turns:
            raise ValueError("model_turns must equal agent plus user model turns")
        return self


class Tau2ProviderTurnRecord(BaseModel):
    """Credential-free immutable binding for one physical tau2 model turn.

    The provider ledger remains the spending authority.  This record is the
    episode-side half of the join: it binds the exact role-local turn index and
    canonical request hash to the logical call and physical provider attempt
    that were durably settled before the episode outcome was published.
    """

    model_config = FrozenModelConfig

    schema_version: Literal["grace.tau2-provider-turn-record.v1"] = (
        "grace.tau2-provider-turn-record.v1"
    )
    record_id: NonEmpty
    episode_attempt_id: NonEmpty
    role: Literal["agent", "user"]
    turn_index: int = Field(ge=0)
    canonical_request_hash: NonEmpty
    logical_call_id: NonEmpty
    provider_attempt_id: NonEmpty
    route: NonEmpty
    model: NonEmpty
    outcome: Literal["succeeded", "failed"]

    @staticmethod
    def _payload(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "namespace": "grace.tau2-provider-turn-record.v1",
            **{
                key: value
                for key, value in values.items()
                if key not in {"record_id", "schema_version"}
            },
        }

    @classmethod
    def create(
        cls,
        *,
        episode_attempt_id: str,
        role: Literal["agent", "user"],
        turn_index: int,
        canonical_request_hash: str,
        logical_call_id: str,
        provider_attempt_id: str,
        route: str,
        model: str,
        outcome: Literal["succeeded", "failed"],
    ) -> Tau2ProviderTurnRecord:
        values = {
            "episode_attempt_id": episode_attempt_id,
            "role": role,
            "turn_index": turn_index,
            "canonical_request_hash": canonical_request_hash,
            "logical_call_id": logical_call_id,
            "provider_attempt_id": provider_attempt_id,
            "route": route,
            "model": model,
            "outcome": outcome,
        }
        return cls.model_validate({"record_id": canonical_sha256(cls._payload(values)), **values})

    @model_validator(mode="after")
    def validate_record(self) -> Tau2ProviderTurnRecord:
        for field, value in {
            "record_id": self.record_id,
            "episode_attempt_id": self.episode_attempt_id,
            "canonical_request_hash": self.canonical_request_hash,
            "logical_call_id": self.logical_call_id,
            "provider_attempt_id": self.provider_attempt_id,
        }.items():
            validate_sha256(value, field=field)
        expected_logical_call_id = make_tau2_turn_logical_call_id(
            episode_attempt_id=self.episode_attempt_id,
            role=self.role,
            turn_index=self.turn_index,
            model=self.model,
            canonical_request_hash=self.canonical_request_hash,
        )
        if self.logical_call_id != expected_logical_call_id:
            raise ValueError("tau2 turn logical_call_id does not match its exact request binding")
        if self.record_id != canonical_sha256(self._payload(self.model_dump(mode="json"))):
            raise ValueError("tau2 provider turn record_id does not match record content")
        return self


class EpisodeAttemptOutcome(BaseModel):
    """Immutable result of one physical attempt; only behavioral results are accepted."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-attempt-outcome.v2"] = "grace.episode-attempt-outcome.v2"
    outcome_id: NonEmpty
    contract: EpisodeAttemptContract
    disposition: EpisodeDisposition
    success: bool
    reward: float | None = Field(default=None, ge=0.0, le=1.0)
    termination_reason: NonEmpty
    infrastructure_error_type: str | None = None
    trajectory: TrajectoryRecord | None = None
    reward_details: JsonValue | None = None
    accounting: EpisodeAccounting = Field(default_factory=EpisodeAccounting)
    provider_turns: tuple[Tau2ProviderTurnRecord, ...] = ()

    @staticmethod
    def _payload(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "namespace": "grace.episode-attempt-outcome.v2",
            **{
                key: value
                for key, value in values.items()
                if key not in {"outcome_id", "schema_version"}
            },
        }

    @classmethod
    def behavioral(
        cls,
        *,
        contract: EpisodeAttemptContract,
        reward: float,
        termination_reason: str,
        messages: Sequence[TrajectoryMessage],
        reward_details: JsonValue | None = None,
        accounting: EpisodeAccounting | None = None,
        provider_turns: Sequence[Tau2ProviderTurnRecord] = (),
    ) -> EpisodeAttemptOutcome:
        success = abs(reward - 1.0) < 1e-6
        trajectory = TrajectoryRecord(
            episode_id=contract.episode_id,
            task_uid=contract.task_uid,
            task_definition_hash=contract.task_definition_hash,
            benchmark_task_id=contract.benchmark_task_id,
            split_slot=contract.split_slot,
            trial=contract.trial,
            protocol_seed=contract.protocol_seed,
            success=success,
            termination_reason=termination_reason,
            messages=tuple(messages),
        )
        values: dict[str, Any] = {
            "contract": contract.model_dump(mode="json"),
            "disposition": "behavioral_success" if success else "behavioral_failure",
            "success": success,
            "reward": reward,
            "termination_reason": termination_reason,
            "infrastructure_error_type": None,
            "trajectory": trajectory.model_dump(mode="json"),
            "reward_details": reward_details,
            "accounting": (accounting or EpisodeAccounting()).model_dump(mode="json"),
            "provider_turns": tuple(record.model_dump(mode="json") for record in provider_turns),
        }
        return cls(outcome_id=canonical_sha256(cls._payload(values)), **values)

    @classmethod
    def infrastructure(
        cls,
        *,
        contract: EpisodeAttemptContract,
        error_type: str,
        accounting: EpisodeAccounting | None = None,
        provider_turns: Sequence[Tau2ProviderTurnRecord] = (),
    ) -> EpisodeAttemptOutcome:
        values: dict[str, Any] = {
            "contract": contract.model_dump(mode="json"),
            "disposition": "infrastructure_error",
            "success": False,
            "reward": None,
            "termination_reason": "infrastructure_error",
            "infrastructure_error_type": error_type,
            "trajectory": None,
            "reward_details": None,
            "accounting": (accounting or EpisodeAccounting()).model_dump(mode="json"),
            "provider_turns": tuple(record.model_dump(mode="json") for record in provider_turns),
        }
        return cls(outcome_id=canonical_sha256(cls._payload(values)), **values)

    @model_validator(mode="after")
    def validate_outcome(self) -> EpisodeAttemptOutcome:
        validate_sha256(self.outcome_id, field="outcome_id")
        payload = self.model_dump(mode="json")
        if self.outcome_id != canonical_sha256(self._payload(payload)):
            raise ValueError("outcome_id does not match outcome content")
        if self.disposition == "infrastructure_error":
            if self.reward is not None or self.trajectory is not None:
                raise ValueError("infrastructure attempts cannot carry reward/trajectory")
            if not self.infrastructure_error_type:
                raise ValueError("infrastructure attempts require a safe error type")
        else:
            if self.reward is None or self.trajectory is None:
                raise ValueError("behavioral attempts require reward and trajectory")
            if self.infrastructure_error_type is not None:
                raise ValueError("behavioral attempts cannot carry infrastructure error type")
            if self.trajectory.episode_id != self.contract.episode_id:
                raise ValueError("trajectory episode does not match attempt contract")
            if self.trajectory.task_uid != self.contract.task_uid:
                raise ValueError("trajectory task does not match attempt contract")
            if not self.trajectory.substantive:
                raise ValueError("behavioral attempt trajectory must be substantive")
            expected_success = abs(self.reward - 1.0) < 1e-6
            if self.success != expected_success or self.trajectory.success != expected_success:
                raise ValueError("success must be derived from deterministic reward==1")
        if any(
            record.episode_attempt_id != self.contract.attempt_id for record in self.provider_turns
        ):
            raise ValueError("provider turn record belongs to another episode attempt")
        provider_attempt_ids = [record.provider_attempt_id for record in self.provider_turns]
        logical_call_ids = [record.logical_call_id for record in self.provider_turns]
        role_indices = [(record.role, record.turn_index) for record in self.provider_turns]
        if (
            len(provider_attempt_ids) != len(set(provider_attempt_ids))
            or len(logical_call_ids) != len(set(logical_call_ids))
            or len(role_indices) != len(set(role_indices))
        ):
            raise ValueError("provider turn records must have unique physical and logical IDs")
        return self


class EpisodeMatrixExecution(BaseModel):
    """Sequential execution result; completion still depends on immutable authority."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-matrix-execution.v1"] = (
        "grace.episode-matrix-execution.v1"
    )
    status: Literal["complete", "incomplete"]
    matrix_hash: NonEmpty
    accepted: tuple[EpisodeAttemptOutcome, ...]
    attempts: tuple[EpisodeAttemptOutcome, ...]
    completeness: CompletenessReport
    executor_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_execution(self) -> EpisodeMatrixExecution:
        if self.matrix_hash != self.completeness.matrix_hash:
            raise ValueError("execution and completeness matrix hashes differ")
        canonically_complete = self.completeness.complete and not self.completeness.out_of_order
        if (self.status == "complete") != canonically_complete:
            raise ValueError("execution status does not match exact ordered completeness")
        if self.completeness.observed_count != len(self.accepted):
            raise ValueError("completeness observed count does not match accepted outcomes")
        if self.completeness.accepted_count != len(self.accepted):
            raise ValueError("completeness accepted count does not match accepted outcomes")
        attempt_ids = [item.contract.attempt_id for item in self.attempts]
        accepted_ids = [item.contract.attempt_id for item in self.accepted]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("execution contains duplicate attempt IDs")
        if len(accepted_ids) != len(set(accepted_ids)):
            raise ValueError("execution contains duplicate accepted attempt IDs")
        for outcome in self.accepted:
            if outcome.disposition == "infrastructure_error":
                raise ValueError("execution cannot accept infrastructure attempts")
            if [item for item in self.attempts if item.outcome_id == outcome.outcome_id] != [
                outcome
            ]:
                raise ValueError("accepted outcome must match one immutable attempt")
        return self


@runtime_checkable
class EpisodeExecutor(Protocol):
    """Injected one-attempt executor used by fake and real tau2 paths."""

    def execute(
        self,
        *,
        task: Any,
        contract: EpisodeAttemptContract,
        policy: str,
    ) -> EpisodeAttemptOutcome: ...


@runtime_checkable
class EpisodeArtifactAuthority(Protocol):
    """Minimal immutable authority interface; derived summaries are never queried."""

    def load_accepted(self, episode_id: str) -> EpisodeAttemptOutcome | None: ...

    def load_attempts(self, episode_id: str) -> Sequence[EpisodeAttemptOutcome]: ...

    def reserve_attempt(self, contract: EpisodeAttemptContract) -> None:
        """Persist before dispatch; pending/duplicate reservations must raise."""

        ...

    def record_attempt(self, outcome: EpisodeAttemptOutcome) -> None:
        """Finalize one exactly matching reserved attempt, once."""

        ...

    def accept_attempt(self, episode_id: str, attempt_id: str) -> EpisodeAttemptOutcome:
        """Atomically accept one already-recorded behavioral outcome."""

        ...


def _task_id(task: Any) -> str:
    if isinstance(task, Mapping):
        value = task.get("id")
    else:
        value = getattr(task, "id", None)
    if not isinstance(value, str) or not value:
        raise EpisodeContractError("each task must expose a non-empty string id")
    return value


def _validate_episode(
    episode: EpisodeCell,
    *,
    config: EpisodeHarnessConfig,
) -> None:
    expected_task_uid = make_task_uid(
        episode.benchmark_task_id,
        benchmark_revision=config.benchmark_revision,
        domain=config.domain,
    )
    if episode.task_uid != expected_task_uid:
        raise EpisodeContractError("matrix task UID does not match pinned tau2 revision")
    expected_episode_id = make_episode_id(
        task_uid=episode.task_uid,
        task_definition_hash=episode.task_definition_hash,
        split_slot=episode.split_slot,
        trial=episode.trial,
        protocol_seed=episode.protocol_seed,
        checkpoint_state_id=episode.checkpoint_state_id,
        benchmark_revision=config.benchmark_revision,
        protocol_hash=config.protocol_hash,
        config_hash=config.config_hash,
        model_roles_hash=config.model_roles_hash,
        evaluator_hash=config.evaluator_hash,
    )
    if episode.episode_id != expected_episode_id:
        raise EpisodeContractError("matrix episode ID does not match execution profile")


def _validate_attempt_for_episode(
    outcome: EpisodeAttemptOutcome,
    episode: EpisodeCell,
    config: EpisodeHarnessConfig,
    policy_hash: str,
    prompt_hash: str,
) -> None:
    expected = EpisodeAttemptContract.create(
        episode=episode,
        attempt_index=outcome.contract.attempt_index,
        config=config,
        policy_hash=policy_hash,
        prompt_hash=prompt_hash,
    )
    if outcome.contract != expected:
        raise EpisodeAuthorityError("attempt contract does not match expected episode inputs")


def _safe_exception_type(error: Exception) -> str:
    message = str(error).lower()
    if "openai" in message and ("quota" in message or "exceeded" in message):
        return "openai_quota"
    if any(token in message for token in ("429", "resource_exhausted", "rate limit")):
        return "rate_limit"
    if "assistantmessage must have either content or tool calls" in message:
        return "empty_response"
    if "timeout" in message or isinstance(error, TimeoutError):
        return "timeout"
    if "auth" in message or "credential" in message or "permission" in message:
        return "authentication"
    return type(error).__name__


def _safe_provider_attempt_category(error: Exception) -> str:
    """Map arbitrary tau2/provider exceptions into the bounded ledger taxonomy."""

    error_type = _safe_exception_type(error)
    if error_type == "authentication":
        return "authentication"
    if error_type == "empty_response":
        return "empty_response"
    if error_type == "timeout":
        return "overall_timeout"
    if error_type in {"openai_quota", "rate_limit"}:
        return "transient_transport"
    return "transport"


def _validated_history(
    *,
    authority: EpisodeArtifactAuthority,
    episode: EpisodeCell,
    config: EpisodeHarnessConfig,
    policy_hash: str,
    prompt_hash: str,
) -> tuple[EpisodeAttemptOutcome, ...]:
    history = tuple(authority.load_attempts(episode.episode_id))
    indices = [outcome.contract.attempt_index for outcome in history]
    if indices != list(range(len(indices))) or len(indices) != len(set(indices)):
        raise EpisodeAuthorityError("episode attempt history must be unique and contiguous")
    for outcome in history:
        _validate_attempt_for_episode(outcome, episode, config, policy_hash, prompt_hash)
    behavioral_indices = [
        index
        for index, outcome in enumerate(history)
        if outcome.disposition != "infrastructure_error"
    ]
    if len(behavioral_indices) > 1 or (
        behavioral_indices and behavioral_indices[0] != len(history) - 1
    ):
        raise EpisodeAuthorityError("attempt history cannot continue after a behavioral result")
    return history


def execute_matrix_sequential(
    *,
    expected: ExpectedMatrix,
    tasks: Sequence[Any],
    policy: str,
    prompt_hash: str,
    config: EpisodeHarnessConfig,
    executor: EpisodeExecutor,
    authority: EpisodeArtifactAuthority,
) -> EpisodeMatrixExecution:
    """Run exact cells in matrix order with two attempts and no parallel branch."""

    if not policy.strip():
        raise EpisodeContractError("policy must not be empty")
    policy_hash = text_sha256(policy)
    validate_sha256(prompt_hash, field="prompt_hash")
    for episode in expected.episodes:
        _validate_episode(episode, config=config)

    ordered_task_ids = tuple(dict.fromkeys(ep.benchmark_task_id for ep in expected.episodes))
    ordered_tasks = load_tasks_in_manifest_order(ordered_task_ids, tasks)
    task_by_id = {_task_id(task): task for task in ordered_tasks}

    accepted: list[EpisodeAttemptOutcome] = []
    all_attempts: list[EpisodeAttemptOutcome] = []
    executor_calls = 0
    for episode in expected.episodes:
        task = task_by_id[episode.benchmark_task_id]
        history = _validated_history(
            authority=authority,
            episode=episode,
            config=config,
            policy_hash=policy_hash,
            prompt_hash=prompt_hash,
        )
        cached = authority.load_accepted(episode.episode_id)
        if cached is not None:
            _validate_attempt_for_episode(cached, episode, config, policy_hash, prompt_hash)
            if cached.disposition == "infrastructure_error":
                raise EpisodeAuthorityError("authority accepted an infrastructure attempt")
            matching = [
                item for item in history if item.contract.attempt_id == cached.contract.attempt_id
            ]
            if matching != [cached]:
                raise EpisodeAuthorityError("accepted outcome is not its immutable attempt record")
            accepted.append(cached)
            all_attempts.extend(history)
            continue

        all_attempts.extend(history)
        episode_attempts = list(history)

        completed_behavioral = next(
            (outcome for outcome in history if outcome.disposition != "infrastructure_error"),
            None,
        )
        if completed_behavioral is not None:
            accepted_outcome = authority.accept_attempt(
                episode.episode_id, completed_behavioral.contract.attempt_id
            )
            if accepted_outcome != completed_behavioral:
                raise EpisodeAuthorityError("accepted outcome differs from recorded attempt")
            if authority.load_accepted(episode.episode_id) != completed_behavioral:
                raise EpisodeAuthorityError("authority did not persist the accepted attempt")
            accepted.append(accepted_outcome)
            continue

        for attempt_index in range(len(history), config.episode_max_attempts):
            contract = EpisodeAttemptContract.create(
                episode=episode,
                attempt_index=attempt_index,
                config=config,
                policy_hash=policy_hash,
                prompt_hash=prompt_hash,
            )
            # This is intentionally outside the executor exception boundary:
            # dispatch is forbidden unless the exact attempt contract is
            # durably reserved.  A stale/pending reservation therefore blocks
            # automatic rerun after a possible process crash.
            authority.reserve_attempt(contract)
            try:
                outcome = executor.execute(task=task, contract=contract, policy=policy)
            except Tau2HarnessError:
                raise
            except Exception as exc:
                outcome = EpisodeAttemptOutcome.infrastructure(
                    contract=contract,
                    error_type=_safe_exception_type(exc),
                )
            executor_calls += 1
            if outcome.contract != contract:
                raise EpisodeContractError("executor returned an outcome for a different attempt")
            authority.record_attempt(outcome)
            # Read back immediately so a broken authority cannot cause an
            # attempt ID to be silently reused after restart.
            episode_attempts.append(outcome)
            persisted_history = _validated_history(
                authority=authority,
                episode=episode,
                config=config,
                policy_hash=policy_hash,
                prompt_hash=prompt_hash,
            )
            if persisted_history != tuple(episode_attempts):
                raise EpisodeAuthorityError(
                    "authority did not persist the exact append-only attempt history"
                )
            all_attempts.append(outcome)
            if outcome.disposition == "infrastructure_error":
                continue
            accepted_outcome = authority.accept_attempt(
                episode.episode_id, outcome.contract.attempt_id
            )
            if accepted_outcome != outcome:
                raise EpisodeAuthorityError("accepted outcome differs from executed attempt")
            if authority.load_accepted(episode.episode_id) != outcome:
                raise EpisodeAuthorityError("authority did not persist the accepted attempt")
            accepted.append(accepted_outcome)
            break

    completeness = compare_exact_matrix(
        expected, (outcome.contract.episode_id for outcome in accepted)
    )
    return EpisodeMatrixExecution(
        status=(
            "complete" if completeness.complete and not completeness.out_of_order else "incomplete"
        ),
        matrix_hash=expected.matrix_hash,
        accepted=tuple(accepted),
        attempts=tuple(all_attempts),
        completeness=completeness,
        executor_calls=executor_calls,
    )


def _optional_tau2_imports() -> dict[str, Any]:
    # The pinned tau2 revision calls python-dotenv directly at module import.
    # Disable it before asking GRACE's public helper to import LiteLLM under its
    # own fail-closed PRODUCTION-mode boundary.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    # LiteLLM otherwise performs an import-time HTTP fetch for a mutable model
    # cost map.  GRACE owns a dated pricing authority and requires imports and
    # offline qualification to remain network-free, so force LiteLLM's bundled
    # local map before its lazy production-mode import.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    try:
        from grace.providers.litellm import ensure_litellm_production

        ensure_litellm_production()
    except ProviderConfigurationError as exc:
        safe_reason = (
            "already imported outside PRODUCTION"
            if "outside PRODUCTION" in str(exc)
            else "PRODUCTION mode unavailable"
        )
        raise Tau2UnavailableError(
            f"LiteLLM production-mode initialization failed for tau2: {safe_reason}"
        ) from None
    try:
        import tau2.agent.llm_agent as llm_agent_module
        import tau2.user.user_simulator as user_simulator_module
        from tau2.agent.llm_agent import LLMAgent
        from tau2.domains.telecom.environment import get_environment_manual_policy
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.run import load_tasks
        from tau2.user.user_simulator import UserSimulator
        from tau2.utils.llm_utils import to_litellm_messages
    except ImportError as exc:
        raise Tau2UnavailableError(
            "tau2 reproduction dependency is unavailable; install the pinned reproduction extra"
        ) from exc
    return {
        "EvaluationType": EvaluationType,
        "LLMAgent": LLMAgent,
        "llm_agent_module": llm_agent_module,
        "Orchestrator": Orchestrator,
        "UserSimulator": UserSimulator,
        "user_simulator_module": user_simulator_module,
        "evaluate_simulation": evaluate_simulation,
        "get_environment_manual_policy": get_environment_manual_policy,
        "load_tasks": load_tasks,
        "to_litellm_messages": to_litellm_messages,
    }


def _git_revision(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return None if dirty else revision
    except (OSError, subprocess.CalledProcessError):
        return None


def _direct_url_revision() -> str | None:
    try:
        payload = json.loads(metadata.distribution("tau2").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        return None
    revision = payload.get("vcs_info", {}).get("commit_id")
    if isinstance(revision, str) and revision:
        return revision

    # Editable local installs omit ``vcs_info``.  Resolve only the explicit
    # package URL; never walk from cwd or site-packages into an unrelated repo.
    url = payload.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return _git_revision(Path(unquote(parsed.path)).resolve())


def installed_tau2_revision() -> str | None:
    """Discover an install-record revision without importing unverified tau2 code."""

    return _direct_url_revision()


def require_tau2_revision(expected_revision: str = PINNED_TAU2_REVISION) -> str:
    """Fail closed when the optional dependency cannot prove the pinned revision."""

    actual = installed_tau2_revision()
    if actual is None:
        raise Tau2RevisionError("installed tau2 revision is not verifiable")
    if actual != expected_revision:
        raise Tau2RevisionError(
            f"tau2 revision mismatch: expected {expected_revision}, got {actual}"
        )
    _optional_tau2_imports()
    return actual


def load_tau2_task_selection(
    manifest: TaskSelectionManifest,
) -> ResolvedTaskSelection:
    """Lazy-load the full domain corpus, then rebuild exact manifest order."""

    require_tau2_revision(manifest.benchmark_revision)
    api = _optional_tau2_imports()
    source = api["load_tasks"](task_set_name=manifest.domain, task_split_name=None)
    return resolve_task_selection(manifest, source, publication_status="public")


def get_tau2_initial_policy() -> str:
    """Return the pinned telecom policy from a fresh environment, lazily."""

    require_tau2_revision()
    api = _optional_tau2_imports()
    return api["get_environment_manual_policy"]().get_policy()


def _message_usage(messages: Sequence[Any]) -> EpisodeAccounting:
    agent_in = agent_out = user_in = user_out = 0
    agent_usage_complete = user_usage_complete = True
    agent_turns = user_turns = tool_calls = 0
    for message in messages:
        role = getattr(message, "role", None)
        usage = getattr(message, "usage", None)
        if role in {"assistant", "user"}:
            if role == "assistant":
                agent_turns += 1
            else:
                user_turns += 1

            usage_complete = False
            prompt = completion = 0
            if isinstance(usage, Mapping):
                prompt_raw = usage.get("prompt_tokens")
                completion_raw = usage.get("completion_tokens")
                if (
                    isinstance(prompt_raw, int)
                    and not isinstance(prompt_raw, bool)
                    and prompt_raw >= 0
                    and isinstance(completion_raw, int)
                    and not isinstance(completion_raw, bool)
                    and completion_raw >= 0
                ):
                    usage_complete = True
                    prompt = prompt_raw
                    completion = completion_raw

            if role == "assistant":
                agent_usage_complete = agent_usage_complete and usage_complete
                agent_in += prompt
                agent_out += completion
            else:
                user_usage_complete = user_usage_complete and usage_complete
                user_in += prompt
                user_out += completion
        calls = getattr(message, "tool_calls", None)
        if calls:
            tool_calls += len(calls)
    return EpisodeAccounting(
        agent_input_tokens=agent_in if agent_usage_complete else None,
        agent_output_tokens=agent_out if agent_usage_complete else None,
        user_input_tokens=user_in if user_usage_complete else None,
        user_output_tokens=user_out if user_usage_complete else None,
        agent_model_turns=agent_turns,
        user_model_turns=user_turns,
        model_turns=agent_turns + user_turns,
        tool_calls=tool_calls,
    )


def _sanitized_messages(messages: Sequence[Any]) -> tuple[TrajectoryMessage, ...]:
    sanitized: list[TrajectoryMessage] = []
    for message in messages:
        nested = getattr(message, "tool_messages", None)
        if nested:
            sanitized.extend(_sanitized_messages(nested))
            continue
        role = getattr(message, "role", None)
        if not isinstance(role, str):
            role = getattr(role, "value", None)
        if not isinstance(role, str) or not role:
            continue
        content = getattr(message, "content", None)
        calls: list[TrajectoryToolCall] = []
        for call in getattr(message, "tool_calls", None) or ():
            name = getattr(call, "name", None)
            arguments = getattr(call, "arguments", {})
            if isinstance(name, str) and name:
                calls.append(TrajectoryToolCall(name=name, arguments=arguments))
        sanitized.append(TrajectoryMessage(role=role, content=content, tool_calls=tuple(calls)))
    return tuple(sanitized)


def _safe_check_counts(checks: Any, *, passed_field: str) -> dict[str, JsonValue]:
    """Reduce evaluator objects to non-sensitive deterministic counts."""

    if checks is None:
        return {"present": False, "total": 0, "passed": 0}
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        raise EpisodeContractError("tau2 evaluator checks have an unexpected shape")
    values = tuple(checks)
    return {
        "present": True,
        "total": len(values),
        "passed": sum(getattr(check, passed_field, None) is True for check in values),
    }


def _reward_details(reward_info: Any) -> JsonValue:
    """Persist only an allowlisted evaluator summary.

    ``RewardInfo`` can contain task assertions, database values, action
    arguments, free-form justifications, and arbitrary ``info``.  None of
    those fields are copied into release artifacts.
    """

    reward = getattr(reward_info, "reward", None)
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        raise EpisodeContractError("tau2 evaluator reward must be numeric")
    allowed_basis = {
        "DB",
        "ENV_ASSERTION",
        "NL_ASSERTION",
        "ACTION",
        "COMMUNICATE",
    }
    basis_values: list[JsonValue] = []
    for item in getattr(reward_info, "reward_basis", None) or ():
        value = getattr(item, "value", item)
        if isinstance(value, str) and value in allowed_basis:
            basis_values.append(value)

    db_check = getattr(reward_info, "db_check", None)
    database: dict[str, JsonValue] = {
        "present": db_check is not None,
        "passed": (getattr(db_check, "db_match", None) is True if db_check is not None else None),
    }
    checks: dict[str, JsonValue] = {
        "database": database,
        "environment_assertions": _safe_check_counts(
            getattr(reward_info, "env_assertions", None), passed_field="met"
        ),
        "actions": _safe_check_counts(
            getattr(reward_info, "action_checks", None),
            passed_field="action_match",
        ),
        "natural_language_assertions": _safe_check_counts(
            getattr(reward_info, "nl_assertions", None), passed_field="met"
        ),
        "communication": _safe_check_counts(
            getattr(reward_info, "communicate_checks", None), passed_field="met"
        ),
    }
    details: dict[str, JsonValue] = {
        "reward": float(reward),
        "reward_basis": basis_values,
        "checks": checks,
    }
    return details


Tau2TurnRole = Literal["agent", "user"]
_TAU2_GENERATE_PATCH_LOCK = threading.Lock()
_INPUT_BOUND_OVERHEAD_BYTES = 512


def make_tau2_turn_logical_call_id(
    *,
    episode_attempt_id: str,
    role: Tau2TurnRole,
    turn_index: int,
    model: str,
    canonical_request_hash: str,
) -> str:
    """Bind one tau2 provider turn to exact request bytes and episode attempt."""

    validate_sha256(episode_attempt_id, field="episode_attempt_id")
    validate_sha256(canonical_request_hash, field="canonical_request_hash")
    if turn_index < 0:
        raise EpisodeContractError("tau2 role turn index must be non-negative")
    if not model:
        raise EpisodeContractError("tau2 turn model must not be empty")
    return canonical_sha256(
        {
            "canonical_request_hash": canonical_request_hash,
            "episode_attempt_id": episode_attempt_id,
            "model": model,
            "namespace": "grace.tau2-provider-turn.v1",
            "role": role,
            "turn_index": turn_index,
        }
    )


def _known_descriptor(
    *,
    model: str,
    role: Tau2TurnRole,
    runtime_args: Mapping[str, Any],
) -> ProviderDescriptor:
    model_id = model.rsplit("/", 1)[-1]
    if model.startswith("vertex_ai/"):
        project = runtime_args.get("vertex_project")
        location = runtime_args.get("vertex_location")
        return ProviderDescriptor(
            route="vertex_ai_adc",
            model_id=model_id,
            model=model,
            location=location if isinstance(location, str) and location else None,
            project_hash=(
                hashlib.sha256(project.encode("utf-8")).hexdigest()
                if isinstance(project, str) and project
                else None
            ),
        )
    if model.startswith("gemini/"):
        return ProviderDescriptor(route="google_ai_studio", model_id=model_id, model=model)
    if role == "user" and (model.startswith("gpt-") or model.startswith("openai/")):
        return ProviderDescriptor(route="openai", model_id=model_id, model=model)
    raise EpisodeContractError("tau2 per-turn hook requires an explicit supported route")


def _turn_usage(
    message: Any,
    *,
    descriptor: ProviderDescriptor,
    latency_seconds: float,
) -> UsageRecord:
    try:
        raw = getattr(message, "usage", None)
    except Exception:
        raw = None
    if isinstance(raw, Mapping):
        input_tokens = raw.get("prompt_tokens")
        output_tokens = raw.get("completion_tokens")
        if (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens >= 0
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        ):
            return UsageRecord(
                model=descriptor.model,
                route=descriptor.route,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usage_source="provider",
                latency_seconds=latency_seconds,
            )
    return UsageRecord(
        model=descriptor.model,
        route=descriptor.route,
        usage_source="missing",
        latency_seconds=latency_seconds,
    )


class Tau2EpisodeExecutor:
    """Real pinned tau2 executor with runtime-only provider options.

    tau2 invokes LiteLLM internally for each model turn, so this adapter wraps
    the pinned agent/user ``generate`` bindings and requires a pre-dispatch
    ``ProviderAttemptHook`` for every turn.  The caller additionally reserves
    the exact episode attempt in its episode authority before :meth:`execute`.
    """

    _CANONICAL_LLM_ARGS = frozenset(
        {
            "max_tokens",
            "max_completion_tokens",
            "max_retries",
            "messages",
            "model",
            "num_retries",
            "temperature",
            "timeout",
            "request_timeout",
            "tools",
        }
    )

    def __init__(
        self,
        *,
        expected_revision: str = PINNED_TAU2_REVISION,
        agent_runtime_args: Mapping[str, Any] | None = None,
        user_runtime_args: Mapping[str, Any] | None = None,
        attempt_hook: ProviderAttemptHook | None = None,
        agent_descriptor: ProviderDescriptor | None = None,
        user_descriptor: ProviderDescriptor | None = None,
    ):
        self.expected_revision = expected_revision
        self._agent_runtime_args = dict(agent_runtime_args or {})
        self._user_runtime_args = dict(user_runtime_args or {})
        self._attempt_hook = attempt_hook
        self._agent_descriptor = agent_descriptor
        self._user_descriptor = user_descriptor
        forbidden = self._CANONICAL_LLM_ARGS.intersection(
            self._agent_runtime_args.keys() | self._user_runtime_args.keys()
        )
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise EpisodeContractError(
                f"runtime LLM args cannot override canonical fields: {names}"
            )
        self._revision_verified = False

    def _verify(self) -> None:
        if not self._revision_verified:
            require_tau2_revision(self.expected_revision)
            self._revision_verified = True

    def _build_agent(self, task: Any, policy: str, contract: EpisodeAttemptContract):
        api = _optional_tau2_imports()
        environment = api["get_environment_manual_policy"]()
        agent_args = {
            **self._agent_runtime_args,
            "temperature": contract.agent_temperature,
            "num_retries": 0,
            "timeout": contract.model_turn_timeout_seconds,
            "max_tokens": contract.model_turn_max_tokens,
        }
        user_args = {
            **self._user_runtime_args,
            "temperature": contract.user_temperature,
            "num_retries": 0,
            "timeout": contract.model_turn_timeout_seconds,
            "max_tokens": contract.model_turn_max_tokens,
        }
        agent = api["LLMAgent"](
            tools=environment.get_tools(),
            domain_policy=policy,
            llm=contract.agent_model,
            llm_args=agent_args,
        )
        actual_prompt_hash = text_sha256(agent.system_prompt)
        if actual_prompt_hash != contract.prompt_hash:
            raise EpisodeContractError("tau2 agent prompt hash does not match run manifest")
        user = api["UserSimulator"](
            tools=environment.get_user_tools(),
            instructions=str(task.user_scenario),
            llm=contract.user_model,
            llm_args=user_args,
        )
        return api, environment, agent, user

    def prompt_hash(
        self,
        *,
        task: Any,
        policy: str,
        agent_model: str,
        user_model: str,
        model_turn_timeout_seconds: int = 300,
        model_turn_max_tokens: int = 8192,
    ) -> str:
        """Compute the actual pinned LLMAgent system prompt without a model call."""

        if model_turn_timeout_seconds < 1:
            raise EpisodeContractError("model-turn timeout must be positive")
        if model_turn_max_tokens < 1:
            raise EpisodeContractError("model-turn max tokens must be positive")
        self._verify()
        api = _optional_tau2_imports()
        environment = api["get_environment_manual_policy"]()
        agent = api["LLMAgent"](
            tools=environment.get_tools(),
            domain_policy=policy,
            llm=agent_model,
            llm_args={
                "temperature": 0.0,
                "num_retries": 0,
                "timeout": model_turn_timeout_seconds,
                "max_tokens": model_turn_max_tokens,
            },
        )
        # ``task`` and ``user_model`` are explicit to make call sites bind the same roles.
        if _task_id(task) == "" or not user_model:
            raise EpisodeContractError("task and user model must be explicit")
        return text_sha256(agent.system_prompt)

    def _turn_descriptor(self, *, role: Tau2TurnRole, model: str) -> ProviderDescriptor:
        provided = self._agent_descriptor if role == "agent" else self._user_descriptor
        runtime_args = self._agent_runtime_args if role == "agent" else self._user_runtime_args
        expected = _known_descriptor(model=model, role=role, runtime_args=runtime_args)
        if provided is not None:
            if (
                provided.model != expected.model
                or provided.model_id != expected.model_id
                or provided.route != expected.route
            ):
                raise EpisodeContractError(
                    f"tau2 {role} descriptor route/model does not match attempt contract"
                )
            return provided
        return expected

    @staticmethod
    def _canonical_turn_request(
        *,
        api: Mapping[str, Any],
        model: str,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        tool_choice: str | None,
        kwargs: Mapping[str, Any],
    ) -> tuple[str, int]:
        try:
            message_payload = api["to_litellm_messages"](list(messages))
            tool_payload = [tool.openai_schema for tool in tools] if tools else None
            effective_tool_choice = "auto" if tool_payload and tool_choice is None else tool_choice
            payload = {
                "generation": {
                    "max_tokens": kwargs["max_tokens"],
                    "num_retries": kwargs["num_retries"],
                    "seed": kwargs.get("seed"),
                    "temperature": kwargs["temperature"],
                    "timeout": kwargs["timeout"],
                    "tool_choice": effective_tool_choice,
                },
                "messages": message_payload,
                "model": model,
                "namespace": "grace.tau2-canonical-provider-request.v1",
                "tools": tool_payload,
            }
            canonical_request = canonical_json(payload)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise EpisodeContractError(
                "tau2 provider request could not be canonically bound"
            ) from None
        return (
            text_sha256(canonical_request),
            len(canonical_request.encode("utf-8")) + _INPUT_BOUND_OVERHEAD_BYTES,
        )

    @staticmethod
    def _settle_turn(
        hook: ProviderAttemptHook,
        reservation: object,
        *,
        attempt: ProviderAttemptRecord,
        usage: UsageRecord | None,
    ) -> None:
        try:
            hook.after_attempt(reservation, attempt=attempt, usage=usage)
        except Exception:
            raise Tau2ProviderHookError(
                "tau2 provider-attempt settlement failed; execution halted"
            ) from None

    @staticmethod
    def _provider_attempt_id(reservation: object) -> str:
        """Extract the immutable ledger identity before provider dispatch."""

        provider_attempt_id = getattr(reservation, "provider_attempt_id", None)
        if not isinstance(provider_attempt_id, str):
            raise Tau2ProviderHookError(
                "tau2 provider-attempt reservation lacks a durable attempt identity"
            )
        try:
            validate_sha256(provider_attempt_id, field="provider_attempt_id")
        except ValueError:
            raise Tau2ProviderHookError(
                "tau2 provider-attempt reservation has an invalid attempt identity"
            ) from None
        return provider_attempt_id

    def _turn_wrapper(
        self,
        *,
        api: Mapping[str, Any],
        role: Tau2TurnRole,
        original: Any,
        contract: EpisodeAttemptContract,
        descriptor: ProviderDescriptor,
        turn_indices: dict[Tau2TurnRole, int],
        turn_records: list[Tau2ProviderTurnRecord],
        owner_thread: int,
    ) -> Any:
        expected_model = contract.agent_model if role == "agent" else contract.user_model
        expected_temperature = (
            contract.agent_temperature if role == "agent" else contract.user_temperature
        )
        stage = "tau2_agent_turn" if role == "agent" else "tau2_user_turn"
        hook = self._attempt_hook
        assert hook is not None

        def wrapped(
            model: str,
            messages: Sequence[Any],
            tools: Sequence[Any] | None = None,
            tool_choice: str | None = None,
            **kwargs: Any,
        ) -> Any:
            if threading.get_ident() != owner_thread:
                raise Tau2ProviderHookError(
                    "concurrent tau2 generation is forbidden by the sequential profile"
                )
            canonical_values = {
                "max_tokens": contract.model_turn_max_tokens,
                "num_retries": 0,
                "temperature": expected_temperature,
                "timeout": contract.model_turn_timeout_seconds,
            }
            if model != expected_model or any(
                isinstance(kwargs.get(name), bool) or kwargs.get(name) != expected
                for name, expected in canonical_values.items()
            ):
                raise EpisodeContractError(
                    f"tau2 {role} turn changed canonical model/reliability inputs"
                )

            request_hash, input_bound = self._canonical_turn_request(
                api=api,
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                kwargs=kwargs,
            )
            if input_bound > contract.model_turn_max_input_bound:
                raise EpisodeContractError(
                    "tau2 canonical input bound exceeds the configured maximum"
                )
            turn_index = turn_indices[role]
            logical_call_id = make_tau2_turn_logical_call_id(
                episode_attempt_id=contract.attempt_id,
                role=role,
                turn_index=turn_index,
                model=model,
                canonical_request_hash=request_hash,
            )
            context = ProviderAttemptContext(
                descriptor=descriptor,
                stage=stage,
                logical_call_id=logical_call_id,
                attempt=1,
                input_token_bound=input_bound,
                temperature=expected_temperature,
                timeout_seconds=contract.model_turn_timeout_seconds,
                overall_timeout_seconds=contract.model_turn_timeout_seconds,
                elapsed_since_call_start_seconds=0.0,
                max_tokens=contract.model_turn_max_tokens,
            )
            try:
                reservation = hook.before_attempt(context)
            except Exception:
                raise Tau2ProviderHookError(
                    "tau2 provider-attempt reservation failed before dispatch"
                ) from None
            provider_attempt_id = self._provider_attempt_id(reservation)
            turn_indices[role] += 1

            started = time.perf_counter()
            try:
                response = original(
                    model=model,
                    messages=list(messages),
                    tools=list(tools) if tools is not None else None,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            except Exception as exc:
                latency = max(0.0, time.perf_counter() - started)
                attempt = ProviderAttemptRecord(
                    attempt=1,
                    temperature=expected_temperature,
                    timeout_seconds=contract.model_turn_timeout_seconds,
                    overall_timeout_seconds=contract.model_turn_timeout_seconds,
                    elapsed_since_call_start_seconds=latency,
                    latency_seconds=latency,
                    outcome="failed",
                    category=_safe_provider_attempt_category(exc),
                    retryable=False,
                )
                self._settle_turn(hook, reservation, attempt=attempt, usage=None)
                turn_records.append(
                    Tau2ProviderTurnRecord.create(
                        episode_attempt_id=contract.attempt_id,
                        role=role,
                        turn_index=turn_index,
                        canonical_request_hash=request_hash,
                        logical_call_id=logical_call_id,
                        provider_attempt_id=provider_attempt_id,
                        route=descriptor.route,
                        model=descriptor.model,
                        outcome="failed",
                    )
                )
                raise

            latency = max(0.0, time.perf_counter() - started)
            attempt = ProviderAttemptRecord(
                attempt=1,
                temperature=expected_temperature,
                timeout_seconds=contract.model_turn_timeout_seconds,
                overall_timeout_seconds=contract.model_turn_timeout_seconds,
                elapsed_since_call_start_seconds=latency,
                latency_seconds=latency,
                outcome="succeeded",
                category="tau2_model_turn",
                retryable=False,
            )
            usage = _turn_usage(response, descriptor=descriptor, latency_seconds=latency)
            self._settle_turn(hook, reservation, attempt=attempt, usage=usage)
            turn_records.append(
                Tau2ProviderTurnRecord.create(
                    episode_attempt_id=contract.attempt_id,
                    role=role,
                    turn_index=turn_index,
                    canonical_request_hash=request_hash,
                    logical_call_id=logical_call_id,
                    provider_attempt_id=provider_attempt_id,
                    route=descriptor.route,
                    model=descriptor.model,
                    outcome="succeeded",
                )
            )
            return response

        return wrapped

    @contextmanager
    def _instrumented_generate(
        self,
        *,
        api: Mapping[str, Any],
        contract: EpisodeAttemptContract,
        turn_records: list[Tau2ProviderTurnRecord] | None = None,
    ):
        if self._attempt_hook is None:
            raise EpisodeContractError(
                "formal tau2 execution requires a per-turn provider-attempt hook"
            )

        agent_module = api["llm_agent_module"]
        user_module = api["user_simulator_module"]
        agent_original = getattr(agent_module, "generate", None)
        user_original = getattr(user_module, "generate", None)
        if not callable(agent_original) or not callable(user_original):
            raise EpisodeContractError("pinned tau2 generate bindings are unavailable")
        agent_descriptor = self._turn_descriptor(role="agent", model=contract.agent_model)
        user_descriptor = self._turn_descriptor(role="user", model=contract.user_model)
        owner_thread = threading.get_ident()
        turn_indices: dict[Tau2TurnRole, int] = {"agent": 0, "user": 0}
        records = [] if turn_records is None else turn_records

        with _TAU2_GENERATE_PATCH_LOCK:
            agent_wrapper = self._turn_wrapper(
                api=api,
                role="agent",
                original=agent_original,
                contract=contract,
                descriptor=agent_descriptor,
                turn_indices=turn_indices,
                turn_records=records,
                owner_thread=owner_thread,
            )
            user_wrapper = self._turn_wrapper(
                api=api,
                role="user",
                original=user_original,
                contract=contract,
                descriptor=user_descriptor,
                turn_indices=turn_indices,
                turn_records=records,
                owner_thread=owner_thread,
            )
            binding_changed = False
            agent_patched = False
            user_patched = False
            try:
                setattr(agent_module, "generate", agent_wrapper)
                agent_patched = True
                setattr(user_module, "generate", user_wrapper)
                user_patched = True
                yield
            finally:
                if agent_patched:
                    binding_changed = getattr(agent_module, "generate", None) is not agent_wrapper
                    setattr(agent_module, "generate", agent_original)
                if user_patched:
                    binding_changed = (
                        getattr(user_module, "generate", None) is not user_wrapper
                        or binding_changed
                    )
                    setattr(user_module, "generate", user_original)
                if binding_changed:
                    raise Tau2ProviderHookError(
                        "tau2 generate binding changed during instrumented execution"
                    )

    def execute(
        self,
        *,
        task: Any,
        contract: EpisodeAttemptContract,
        policy: str,
    ) -> EpisodeAttemptOutcome:
        self._verify()
        if self._attempt_hook is None:
            raise EpisodeContractError(
                "formal tau2 execution requires a per-turn provider-attempt hook"
            )
        if _task_id(task) != contract.benchmark_task_id:
            raise EpisodeContractError("tau2 task does not match the attempt contract")
        if text_sha256(policy) != contract.policy_hash:
            raise EpisodeContractError("policy hash does not match the attempt contract")
        api, environment, agent, user = self._build_agent(task, policy, contract)
        turn_records: list[Tau2ProviderTurnRecord] = []
        try:
            orchestrator = api["Orchestrator"](
                domain="telecom",
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=contract.max_steps,
                max_errors=contract.max_errors,
                seed=contract.attempt_seed,
                validate_communication=contract.enforce_communication_protocol,
            )
            with self._instrumented_generate(
                api=api,
                contract=contract,
                turn_records=turn_records,
            ):
                simulation = orchestrator.run()
            reward_info = api["evaluate_simulation"](
                simulation=simulation,
                task=task,
                evaluation_type=api["EvaluationType"].ALL,
                solo_mode=False,
                domain="telecom",
            )
            simulation.reward_info = reward_info
        except Tau2HarnessError:
            raise
        except Exception as exc:
            return EpisodeAttemptOutcome.infrastructure(
                contract=contract,
                error_type=_safe_exception_type(exc),
                provider_turns=turn_records,
            )
        accounting = _message_usage(simulation.messages).model_copy(
            update={
                # tau2 maps an unpriceable LiteLLM response to numeric zero.
                # USD authority therefore remains the immutable per-turn
                # budget ledger; episode summaries must not fabricate zero.
                "agent_cost_usd": None,
                "user_cost_usd": None,
                "duration_seconds": simulation.duration,
            }
        )
        termination = simulation.termination_reason
        termination_reason = (
            termination.value if hasattr(termination, "value") else str(termination)
        )
        return EpisodeAttemptOutcome.behavioral(
            contract=contract,
            reward=float(reward_info.reward),
            termination_reason=termination_reason,
            messages=_sanitized_messages(simulation.messages),
            reward_details=_reward_details(reward_info),
            accounting=EpisodeAccounting.model_validate(accounting.model_dump(mode="json")),
            provider_turns=turn_records,
        )


__all__ = [
    "CANONICAL_TAU2_MODEL_TURN_MAX_INPUT_BOUND",
    "CANONICAL_TAU2_USER_MODEL",
    "EpisodeAccounting",
    "EpisodeArtifactAuthority",
    "EpisodeAttemptContract",
    "EpisodeAttemptOutcome",
    "EpisodeAuthorityError",
    "EpisodeContractError",
    "EpisodeExecutor",
    "EpisodeHarnessConfig",
    "EpisodeMatrixExecution",
    "LEGACY_TAU2_USER_MODEL_ALIAS",
    "Tau2EpisodeExecutor",
    "Tau2HarnessError",
    "Tau2ProviderHookError",
    "Tau2ProviderTurnRecord",
    "Tau2RevisionError",
    "Tau2UnavailableError",
    "execute_matrix_sequential",
    "get_tau2_initial_policy",
    "installed_tau2_revision",
    "load_tau2_task_selection",
    "make_tau2_turn_logical_call_id",
    "require_tau2_revision",
]
