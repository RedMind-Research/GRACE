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

"""Offline qualification of the exact sequential tau2 episode harness."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from grace.providers.base import (
    ProviderAttemptContext,
    ProviderAttemptRecord,
    UsageRecord,
)

from reproduction.tau2_telecom.harness.completeness import build_expected_matrix
from reproduction.tau2_telecom.harness.cost import derive_cost_summary
from reproduction.tau2_telecom.harness.evaluation import (
    _validate_eval_smoke_scope,
    run_evaluation_smoke,
)
from reproduction.tau2_telecom.harness.experience import run_experience_collection
from reproduction.tau2_telecom.harness.health import derive_health_summary
from reproduction.tau2_telecom.harness.identity import (
    canonical_sha256,
    make_matrix_hash,
    text_sha256,
)
from reproduction.tau2_telecom.harness.models import (
    ExpectedMatrix,
    TrajectoryMessage,
)
from reproduction.tau2_telecom.harness.tasks import (
    TaskResolutionError,
    load_task_selection_manifest,
    resolve_task_selection,
)
from reproduction.tau2_telecom.harness.tau2_interface import (
    CANONICAL_TAU2_MODEL_TURN_MAX_INPUT_BOUND,
    CANONICAL_TAU2_USER_MODEL,
    LEGACY_TAU2_USER_MODEL_ALIAS,
    EpisodeAccounting,
    EpisodeArtifactAuthority,
    EpisodeAttemptContract,
    EpisodeAttemptOutcome,
    EpisodeAuthorityError,
    EpisodeContractError,
    EpisodeHarnessConfig,
    Tau2EpisodeExecutor,
    Tau2ProviderHookError,
    execute_matrix_sequential,
    make_tau2_turn_logical_call_id,
)


DATA_DIR = Path(__file__).parents[2] / "reproduction/tau2_telecom/data"
POLICY = "Pinned telecom policy for the offline harness test."
PROMPT_HASH = text_sha256("rendered tau2 LLMAgent system prompt")


class FakeAuthority(EpisodeArtifactAuthority):
    """Small append-only authority used only by zero-network tests."""

    def __init__(self) -> None:
        self.attempts: dict[str, list[EpisodeAttemptOutcome]] = {}
        self.accepted: dict[str, EpisodeAttemptOutcome] = {}
        self.reservations: dict[str, EpisodeAttemptContract] = {}

    def load_accepted(self, episode_id: str) -> EpisodeAttemptOutcome | None:
        return self.accepted.get(episode_id)

    def load_attempts(self, episode_id: str) -> Sequence[EpisodeAttemptOutcome]:
        return tuple(self.attempts.get(episode_id, ()))

    def reserve_attempt(self, contract: EpisodeAttemptContract) -> None:
        if contract.attempt_id in self.reservations:
            raise EpisodeAuthorityError("attempt reservation already exists")
        self.reservations[contract.attempt_id] = contract

    def record_attempt(self, outcome: EpisodeAttemptOutcome) -> None:
        attempt_id = outcome.contract.attempt_id
        if self.reservations.get(attempt_id) != outcome.contract:
            raise EpisodeAuthorityError("outcome does not match a reserved attempt")
        for values in self.attempts.values():
            for existing in values:
                if existing.contract.attempt_id == attempt_id:
                    raise EpisodeAuthorityError("attempt ID was reused")
        self.attempts.setdefault(outcome.contract.episode_id, []).append(outcome)

    def accept_attempt(self, episode_id: str, attempt_id: str) -> EpisodeAttemptOutcome:
        matches = [
            item
            for item in self.attempts.get(episode_id, ())
            if item.contract.attempt_id == attempt_id
        ]
        if len(matches) != 1 or matches[0].disposition == "infrastructure_error":
            raise EpisodeAuthorityError("only one recorded behavioral attempt may be accepted")
        existing = self.accepted.get(episode_id)
        if existing is not None and existing != matches[0]:
            raise EpisodeAuthorityError("episode already accepted another attempt")
        self.accepted[episode_id] = matches[0]
        return matches[0]


OutcomeFactory = Callable[[Any, EpisodeAttemptContract], EpisodeAttemptOutcome]


def _behavioral(
    contract: EpisodeAttemptContract,
    *,
    reward: float = 1.0,
    marker: str = "ok",
) -> EpisodeAttemptOutcome:
    return EpisodeAttemptOutcome.behavioral(
        contract=contract,
        reward=reward,
        termination_reason="agent_stop",
        messages=(TrajectoryMessage(role="assistant", content=marker),),
        reward_details={"reward": reward, "checks": {}},
        accounting=EpisodeAccounting(
            agent_input_tokens=10,
            agent_output_tokens=2,
            user_input_tokens=8,
            user_output_tokens=3,
            agent_cost_usd=0.01,
            user_cost_usd=0.02,
            duration_seconds=1.0,
            agent_model_turns=1,
            user_model_turns=1,
            model_turns=2,
        ),
    )


class FakeExecutor:
    def __init__(self, factory: OutcomeFactory | None = None) -> None:
        self.factory = factory or (
            lambda task, contract: _behavioral(contract, marker=f"completed:{_task_id(task)}")
        )
        self.calls: list[tuple[str, EpisodeAttemptContract, str]] = []

    def execute(
        self,
        *,
        task: Any,
        contract: EpisodeAttemptContract,
        policy: str,
    ) -> EpisodeAttemptOutcome:
        self.calls.append((_task_id(task), contract, policy))
        return self.factory(task, contract)


def _task_id(task: Any) -> str:
    if isinstance(task, dict):
        return str(task["id"])
    return str(task.id)


def _profile(name: str) -> tuple[Any, tuple[dict[str, str], ...], Any, EpisodeHarnessConfig]:
    manifest = load_task_selection_manifest(DATA_DIR / name)
    source = tuple(
        {"id": slot.benchmark_task_id, "definition": f"task-{slot.split_slot}"}
        for slot in reversed(manifest.tasks)
    )
    resolved = resolve_task_selection(manifest, source, publication_status="public")
    config = EpisodeHarnessConfig.create(
        protocol_hash=text_sha256("tau2 communication protocol v1"),
        model_turn_timeout_seconds=300,
    )
    expected = build_expected_matrix(
        resolved.task_map,
        trials=(0,),
        seed=manifest.seed,
        checkpoint_state_id="checkpoint-state-v1",
        protocol_hash=config.protocol_hash,
        config_hash=config.config_hash,
        model_roles_hash=config.model_roles_hash,
        evaluator_hash=config.evaluator_hash,
    )
    return manifest, tuple(reversed(resolved.tasks)), expected, config


def _single_cell(expected: ExpectedMatrix) -> ExpectedMatrix:
    episode = expected.episodes[0]
    return ExpectedMatrix(
        episodes=(episode,),
        matrix_hash=make_matrix_hash((episode.episode_id,)),
    )


def test_experience_rebuilds_exact_order_and_binds_every_observation() -> None:
    manifest, reversed_tasks, expected, config = _profile("experience_smoke_v1.json")
    executor = FakeExecutor()
    authority = FakeAuthority()

    result = run_experience_collection(
        manifest=manifest,
        expected=expected,
        tasks=reversed_tasks,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=executor,
        authority=authority,
    )

    expected_task_ids = [item.benchmark_task_id for item in manifest.tasks]
    assert result.status == "complete"
    assert [item[0] for item in executor.calls] == expected_task_ids
    assert [item.benchmark_task_id for item in result.trajectories] == expected_task_ids
    assert config.workers == 1
    assert config.user_model == CANONICAL_TAU2_USER_MODEL
    assert (
        config.model_turn_max_input_bound == CANONICAL_TAU2_MODEL_TURN_MAX_INPUT_BOUND == 1_000_000
    )
    assert config.require_per_turn_attempt_hook is True
    assert LEGACY_TAU2_USER_MODEL_ALIAS == "gpt-4.1"
    for cell, outcome in zip(expected.episodes, result.execution.accepted, strict=True):
        contract = outcome.contract
        assert contract.episode_id == cell.episode_id
        assert contract.task_uid == cell.task_uid
        assert contract.benchmark_task_id == cell.benchmark_task_id
        assert contract.split_slot == cell.split_slot
        assert contract.trial == cell.trial
        assert contract.protocol_seed == cell.protocol_seed
        assert contract.checkpoint_state_id == cell.checkpoint_state_id
        assert contract.policy_hash == text_sha256(POLICY)
        assert contract.prompt_hash == PROMPT_HASH
        assert contract.model_roles_hash == config.model_roles_hash
        assert contract.evaluator_hash == config.evaluator_hash
        assert contract.config_hash == config.config_hash
        assert contract.model_turn_max_input_bound == 1_000_000
        assert contract.require_per_turn_attempt_hook is True


def test_public_smoke_profile_rejects_the_historical_moving_user_alias() -> None:
    with pytest.raises(ValueError, match="gpt-4.1-2025-04-14"):
        EpisodeHarnessConfig.create(
            protocol_hash=text_sha256("tau2 communication protocol v1"),
            user_model=LEGACY_TAU2_USER_MODEL_ALIAS,  # type: ignore[arg-type]
        )


def test_public_smoke_profile_cannot_disable_hook_or_change_input_bound() -> None:
    protocol_hash = text_sha256("tau2 communication protocol v1")
    with pytest.raises(ValueError):
        EpisodeHarnessConfig.create(
            protocol_hash=protocol_hash,
            require_per_turn_attempt_hook=False,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        EpisodeHarnessConfig.create(
            protocol_hash=protocol_hash,
            model_turn_max_input_bound=999_999,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="temperatures"):
        EpisodeHarnessConfig.create(
            protocol_hash=protocol_hash,
            attempt_agent_temperatures=(0.0, 0.7),
        )
    with pytest.raises(ValueError, match="user simulator temperature"):
        EpisodeHarnessConfig.create(protocol_hash=protocol_hash, user_temperature=0.1)


def test_input_bound_is_part_of_config_and_attempt_contract_hashes() -> None:
    _, _, expected, config = _profile("experience_smoke_v1.json")
    config_payload = EpisodeHarnessConfig._config_payload(
        model_turn_timeout_seconds=config.model_turn_timeout_seconds,
        model_turn_max_tokens=config.model_turn_max_tokens,
        model_turn_max_input_bound=config.model_turn_max_input_bound,
        require_per_turn_attempt_hook=config.require_per_turn_attempt_hook,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        retry_seed_stride=config.retry_seed_stride,
        attempt_agent_temperatures=config.attempt_agent_temperatures,
        user_temperature=config.user_temperature,
        enforce_communication_protocol=config.enforce_communication_protocol,
    )
    assert config_payload["model_turn_max_input_bound"] == 1_000_000
    assert config.config_hash == canonical_sha256(config_payload)

    contract = EpisodeAttemptContract.create(
        episode=expected.episodes[0],
        attempt_index=0,
        config=config,
        policy_hash=text_sha256(POLICY),
        prompt_hash=PROMPT_HASH,
    )
    contract_payload = contract.model_dump(mode="json")
    assert contract_payload["model_turn_max_input_bound"] == 1_000_000
    assert contract.contract_hash == canonical_sha256(
        EpisodeAttemptContract._payload(contract_payload)
    )


def test_infrastructure_retry_is_exactly_once_with_distinct_attempt_identity() -> None:
    _, tasks, expected_all, config = _profile("experience_smoke_v1.json")
    expected = _single_cell(expected_all)
    task = (
        next(item for item in tasks if _task_id(item) == expected.episodes[0].benchmark_task_id),
    )

    def transient_then_complete(
        _task: Any, contract: EpisodeAttemptContract
    ) -> EpisodeAttemptOutcome:
        if contract.attempt_index == 0:
            raise TimeoutError("provider timeout with secret details")
        return _behavioral(contract, reward=0.0, marker="behavioral failure is complete")

    executor = FakeExecutor(transient_then_complete)
    authority = FakeAuthority()
    result = execute_matrix_sequential(
        expected=expected,
        tasks=task,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=executor,
        authority=authority,
    )

    assert result.status == "complete"
    assert result.executor_calls == 2
    first, second = result.attempts
    assert first.disposition == "infrastructure_error"
    assert first.infrastructure_error_type == "timeout"
    assert second.disposition == "behavioral_failure"
    assert second.reward == 0.0
    assert first.contract.attempt_id != second.contract.attempt_id
    assert [first.contract.attempt_index, second.contract.attempt_index] == [0, 1]
    assert second.contract.attempt_seed == first.contract.attempt_seed + 7
    assert [first.contract.agent_temperature, second.contract.agent_temperature] == [
        0.0,
        0.5,
    ]
    assert [first.contract.user_temperature, second.contract.user_temperature] == [
        0.0,
        0.0,
    ]
    assert all(item.contract.tau2_inner_num_retries == 0 for item in result.attempts)
    assert all(item.contract.model_turn_timeout_seconds == 300 for item in result.attempts)
    assert all(item.contract.model_turn_max_tokens == 8192 for item in result.attempts)
    assert all(item.contract.model_turn_max_input_bound == 1_000_000 for item in result.attempts)
    assert all(item.contract.require_per_turn_attempt_hook is True for item in result.attempts)
    assert result.accepted == (second,)


def test_two_infrastructure_attempts_remain_incomplete_and_never_reexecute() -> None:
    _, tasks, expected_all, config = _profile("experience_smoke_v1.json")
    expected = _single_cell(expected_all)
    task = (
        next(item for item in tasks if _task_id(item) == expected.episodes[0].benchmark_task_id),
    )
    authority = FakeAuthority()
    first_executor = FakeExecutor(
        lambda _task, _contract: (_ for _ in ()).throw(TimeoutError("timeout"))
    )

    first = execute_matrix_sequential(
        expected=expected,
        tasks=task,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=first_executor,
        authority=authority,
    )
    assert first.status == "incomplete"
    assert first.executor_calls == 2
    assert len(first.attempts) == 2
    assert first.completeness.missing_episode_ids == (expected.episodes[0].episode_id,)

    no_call_executor = FakeExecutor(
        lambda _task, _contract: pytest.fail("exhausted episode was re-executed")
    )
    resumed = execute_matrix_sequential(
        expected=expected,
        tasks=task,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=no_call_executor,
        authority=authority,
    )
    assert resumed.status == "incomplete"
    assert resumed.executor_calls == 0
    assert no_call_executor.calls == []


def test_complete_resume_uses_accepted_authority_with_zero_executor_calls() -> None:
    _, tasks, expected, config = _profile("experience_smoke_v1.json")
    authority = FakeAuthority()
    first = execute_matrix_sequential(
        expected=expected,
        tasks=tasks,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=FakeExecutor(),
        authority=authority,
    )
    snapshot = (
        {key: tuple(value) for key, value in authority.attempts.items()},
        dict(authority.accepted),
        dict(authority.reservations),
    )
    no_call_executor = FakeExecutor(
        lambda _task, _contract: pytest.fail("accepted episode was re-executed")
    )

    resumed = execute_matrix_sequential(
        expected=expected,
        tasks=tasks,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=no_call_executor,
        authority=authority,
    )
    assert first.status == resumed.status == "complete"
    assert resumed.executor_calls == 0
    assert no_call_executor.calls == []
    assert snapshot == (
        {key: tuple(value) for key, value in authority.attempts.items()},
        authority.accepted,
        authority.reservations,
    )


@pytest.mark.parametrize("mode", ["duplicate", "wrong"])
def test_duplicate_or_wrong_source_task_id_fails_before_execution(mode: str) -> None:
    _, tasks, expected, config = _profile("experience_smoke_v1.json")
    source = list(tasks)
    if mode == "duplicate":
        source.append(source[0])
    else:
        source = source[1:] + [{"id": "not-a-selected-task"}]
    executor = FakeExecutor()

    with pytest.raises(TaskResolutionError):
        execute_matrix_sequential(
            expected=expected,
            tasks=source,
            policy=POLICY,
            prompt_hash=PROMPT_HASH,
            config=config,
            executor=executor,
            authority=FakeAuthority(),
        )
    assert executor.calls == []


def test_wrong_contract_outcome_is_rejected_without_authority_record() -> None:
    _, tasks, expected_all, config = _profile("experience_smoke_v1.json")
    expected = _single_cell(expected_all)
    task = (
        next(item for item in tasks if _task_id(item) == expected.episodes[0].benchmark_task_id),
    )
    wrong_contract = EpisodeAttemptContract.create(
        episode=expected_all.episodes[1],
        attempt_index=0,
        config=config,
        policy_hash=text_sha256(POLICY),
        prompt_hash=PROMPT_HASH,
    )
    executor = FakeExecutor(
        lambda _task, _contract: EpisodeAttemptOutcome.infrastructure(
            contract=wrong_contract, error_type="timeout"
        )
    )
    authority = FakeAuthority()

    with pytest.raises(EpisodeContractError, match="different attempt"):
        execute_matrix_sequential(
            expected=expected,
            tasks=task,
            policy=POLICY,
            prompt_hash=PROMPT_HASH,
            config=config,
            executor=executor,
            authority=authority,
        )
    assert authority.attempts == {}
    assert len(authority.reservations) == 1


def test_pending_attempt_reservation_blocks_dispatch_after_crash() -> None:
    _, tasks, expected_all, config = _profile("experience_smoke_v1.json")
    expected = _single_cell(expected_all)
    episode = expected.episodes[0]
    task = (next(item for item in tasks if _task_id(item) == episode.benchmark_task_id),)
    contract = EpisodeAttemptContract.create(
        episode=episode,
        attempt_index=0,
        config=config,
        policy_hash=text_sha256(POLICY),
        prompt_hash=PROMPT_HASH,
    )
    authority = FakeAuthority()
    authority.reserve_attempt(contract)  # Simulate a crash before outcome finalization.
    executor = FakeExecutor()

    with pytest.raises(EpisodeAuthorityError, match="reservation already exists"):
        execute_matrix_sequential(
            expected=expected,
            tasks=task,
            policy=POLICY,
            prompt_hash=PROMPT_HASH,
            config=config,
            executor=executor,
            authority=authority,
        )

    assert executor.calls == []
    assert authority.attempts == {}


def test_authority_fails_closed_if_attempt_follows_behavioral_terminal() -> None:
    _, tasks, expected_all, config = _profile("experience_smoke_v1.json")
    expected = _single_cell(expected_all)
    episode = expected.episodes[0]
    task = (next(item for item in tasks if _task_id(item) == episode.benchmark_task_id),)
    first_contract = EpisodeAttemptContract.create(
        episode=episode,
        attempt_index=0,
        config=config,
        policy_hash=text_sha256(POLICY),
        prompt_hash=PROMPT_HASH,
    )
    second_contract = EpisodeAttemptContract.create(
        episode=episode,
        attempt_index=1,
        config=config,
        policy_hash=text_sha256(POLICY),
        prompt_hash=PROMPT_HASH,
    )
    behavioral = _behavioral(first_contract)
    later = EpisodeAttemptOutcome.infrastructure(contract=second_contract, error_type="timeout")
    authority = FakeAuthority()
    authority.attempts[episode.episode_id] = [behavioral, later]
    authority.accepted[episode.episode_id] = behavioral

    with pytest.raises(EpisodeAuthorityError, match="cannot continue"):
        execute_matrix_sequential(
            expected=expected,
            tasks=task,
            policy=POLICY,
            prompt_hash=PROMPT_HASH,
            config=config,
            executor=FakeExecutor(),
            authority=authority,
        )


def test_eval_smoke_is_exactly_three_and_has_no_pass_metric() -> None:
    manifest, tasks, expected, config = _profile("eval_smoke_v1.json")
    rewards = {
        expected.episodes[0].benchmark_task_id: 0.0,
        expected.episodes[1].benchmark_task_id: 1.0,
        expected.episodes[2].benchmark_task_id: 0.5,
    }
    executor = FakeExecutor(
        lambda task, contract: _behavioral(contract, reward=rewards[_task_id(task)])
    )

    result = run_evaluation_smoke(
        manifest=manifest,
        expected=expected,
        tasks=tasks,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=executor,
        authority=FakeAuthority(),
    )

    assert result.summary.status == "complete"
    assert result.summary.expected_count == result.summary.accepted_count == 3
    assert [item.split_slot for item in result.summary.episodes] == [
        "eval-s020",
        "eval-s043",
        "eval-s060",
    ]
    assert [item.reward for item in result.summary.episodes] == [0.0, 1.0, 0.5]
    assert all(item.status == "completed" for item in result.summary.episodes)
    payload = result.summary.model_dump(mode="json")
    episode_keys = set().union(*(episode.keys() for episode in payload["episodes"]))
    assert not {"pass", "pass_rate", "success", "disposition"}.intersection(episode_keys)
    assert "paper_comparable" not in payload
    assert "reference_aggregate_eligible" not in payload


def test_eval_smoke_rejects_right_slots_with_any_other_task_id() -> None:
    manifest, _, expected, config = _profile("eval_smoke_v1.json")
    tampered_first = manifest.tasks[0].model_copy(
        update={"benchmark_task_id": "wrong-task-in-eval-slot-20"}
    )
    tampered = manifest.model_copy(update={"tasks": (tampered_first, *manifest.tasks[1:])})

    with pytest.raises(EpisodeContractError, match="fixed to EVAL20"):
        _validate_eval_smoke_scope(tampered, expected, config)


def test_eval_infrastructure_is_missing_not_a_zero_reward() -> None:
    manifest, tasks, expected, config = _profile("eval_smoke_v1.json")
    broken_id = expected.episodes[1].benchmark_task_id

    def partial(task: Any, contract: EpisodeAttemptContract) -> EpisodeAttemptOutcome:
        if _task_id(task) == broken_id:
            raise TimeoutError("timeout")
        return _behavioral(contract, reward=0.0)

    result = run_evaluation_smoke(
        manifest=manifest,
        expected=expected,
        tasks=tasks,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=FakeExecutor(partial),
        authority=FakeAuthority(),
    )

    assert result.summary.status == "incomplete"
    assert result.summary.accepted_count == 2
    assert result.summary.missing_episode_ids == (expected.episodes[1].episode_id,)
    assert all(item.benchmark_task_id != broken_id for item in result.summary.episodes)
    assert [item.reward for item in result.summary.episodes] == [0.0, 0.0]


@pytest.mark.parametrize(
    "field",
    [
        "max_tokens",
        "max_completion_tokens",
        "max_retries",
        "model",
        "temperature",
        "num_retries",
        "timeout",
        "request_timeout",
        "messages",
        "tools",
    ],
)
def test_runtime_args_cannot_override_canonical_llm_fields(field: str) -> None:
    with pytest.raises(EpisodeContractError, match="cannot override"):
        Tau2EpisodeExecutor(agent_runtime_args={field: "forbidden"})
    with pytest.raises(EpisodeContractError, match="cannot override"):
        Tau2EpisodeExecutor(user_runtime_args={field: "forbidden"})


def test_real_adapter_forces_timeout_temperature_and_zero_inner_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reproduction.tau2_telecom.harness import tau2_interface

    _, _, expected, config = _profile("experience_smoke_v1.json")
    contract = EpisodeAttemptContract.create(
        episode=expected.episodes[0],
        attempt_index=1,
        config=config,
        policy_hash=text_sha256(POLICY),
        prompt_hash=text_sha256("fake system prompt"),
    )
    captures: dict[str, dict[str, Any]] = {}

    class Environment:
        def get_tools(self) -> list[str]:
            return ["agent-tool"]

        def get_user_tools(self) -> list[str]:
            return ["user-tool"]

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            captures["agent"] = kwargs
            self.system_prompt = "fake system prompt"

    class User:
        def __init__(self, **kwargs: Any) -> None:
            captures["user"] = kwargs

    monkeypatch.setattr(
        tau2_interface,
        "_optional_tau2_imports",
        lambda: {
            "LLMAgent": Agent,
            "UserSimulator": User,
            "get_environment_manual_policy": Environment,
        },
    )
    adapter = Tau2EpisodeExecutor(
        agent_runtime_args={"vertex_location": "us-central1"},
        user_runtime_args={"api_base": "https://example.invalid"},
    )
    task = SimpleNamespace(id=contract.benchmark_task_id, user_scenario="scenario")
    adapter._build_agent(task, POLICY, contract)

    assert captures["agent"]["llm"] == "vertex_ai/gemini-2.5-flash"
    assert captures["user"]["llm"] == "gpt-4.1-2025-04-14"
    for role in ("agent", "user"):
        args = captures[role]["llm_args"]
        assert args["num_retries"] == 0
        assert args["timeout"] == 300
        assert args["max_tokens"] == 8192
    assert captures["agent"]["llm_args"]["temperature"] == 0.5
    assert captures["user"]["llm_args"]["temperature"] == 0.0


class RecordingTurnHook:
    def __init__(self, *, fail_before: bool = False) -> None:
        self.fail_before = fail_before
        self.contexts: list[ProviderAttemptContext] = []
        self.settlements: list[tuple[object, ProviderAttemptRecord, UsageRecord | None]] = []

    def before_attempt(self, context: ProviderAttemptContext) -> object:
        self.contexts.append(context)
        if self.fail_before:
            raise RuntimeError("budget denied")
        return SimpleNamespace(
            provider_attempt_id=canonical_sha256(
                {
                    "logical_call_id": context.logical_call_id,
                    "namespace": "test.tau2-provider-attempt.v1",
                }
            )
        )

    def after_attempt(
        self,
        reservation: object,
        *,
        attempt: ProviderAttemptRecord,
        usage: UsageRecord | None,
    ) -> None:
        self.settlements.append((reservation, attempt, usage))


def _turn_contract() -> EpisodeAttemptContract:
    _, _, expected, config = _profile("experience_smoke_v1.json")
    return EpisodeAttemptContract.create(
        episode=expected.episodes[0],
        attempt_index=1,
        config=config,
        policy_hash=text_sha256(POLICY),
        prompt_hash=PROMPT_HASH,
    )


def _turn_kwargs(contract: EpisodeAttemptContract, *, role: str) -> dict[str, Any]:
    return {
        "temperature": (
            contract.agent_temperature if role == "agent" else contract.user_temperature
        ),
        "num_retries": 0,
        "timeout": contract.model_turn_timeout_seconds,
        "max_tokens": contract.model_turn_max_tokens,
        "seed": contract.attempt_seed,
    }


def test_per_turn_hook_binds_vertex_and_openai_dispatches_and_restores_wrappers() -> None:
    contract = _turn_contract()
    hook = RecordingTurnHook()
    original_calls: list[str] = []

    def agent_original(**_kwargs: Any) -> Any:
        original_calls.append("agent")
        return SimpleNamespace(usage={"prompt_tokens": 11, "completion_tokens": 3})

    def user_original(**_kwargs: Any) -> Any:
        original_calls.append("user")
        return SimpleNamespace(usage={"prompt_tokens": 7, "completion_tokens": 2})

    agent_module = SimpleNamespace(generate=agent_original)
    user_module = SimpleNamespace(generate=user_original)
    api = {
        "llm_agent_module": agent_module,
        "user_simulator_module": user_module,
        "to_litellm_messages": lambda messages: list(messages),
    }
    adapter = Tau2EpisodeExecutor(attempt_hook=hook)
    agent_messages = [{"role": "system", "content": "agent policy"}]
    user_messages = [{"role": "system", "content": "private user scenario"}]
    tool = SimpleNamespace(openai_schema={"type": "function", "function": {"name": "lookup"}})
    turn_records = []

    with adapter._instrumented_generate(
        api=api,
        contract=contract,
        turn_records=turn_records,
    ):
        assert agent_module.generate is not agent_original
        assert user_module.generate is not user_original
        agent_module.generate(
            model=contract.agent_model,
            messages=agent_messages,
            tools=[tool],
            **_turn_kwargs(contract, role="agent"),
        )
        user_module.generate(
            model=contract.user_model,
            messages=user_messages,
            **_turn_kwargs(contract, role="user"),
        )

    assert agent_module.generate is agent_original
    assert user_module.generate is user_original
    assert original_calls == ["agent", "user"]
    assert [item.stage for item in hook.contexts] == [
        "tau2_agent_turn",
        "tau2_user_turn",
    ]
    assert [item.descriptor.route for item in hook.contexts] == [
        "vertex_ai_adc",
        "openai",
    ]
    assert [item.descriptor.model for item in hook.contexts] == [
        "vertex_ai/gemini-2.5-flash",
        "gpt-4.1-2025-04-14",
    ]
    assert all(item.attempt == 1 for item in hook.contexts)
    assert all(item.max_tokens == 8192 for item in hook.contexts)
    assert all(item.input_token_bound > 512 for item in hook.contexts)
    assert hook.contexts[0].logical_call_id != hook.contexts[1].logical_call_id
    assert "private user scenario" not in str(hook.contexts)

    agent_hash, _ = adapter._canonical_turn_request(
        api=api,
        model=contract.agent_model,
        messages=agent_messages,
        tools=[tool],
        tool_choice=None,
        kwargs=_turn_kwargs(contract, role="agent"),
    )
    assert hook.contexts[0].logical_call_id == make_tau2_turn_logical_call_id(
        episode_attempt_id=contract.attempt_id,
        role="agent",
        turn_index=0,
        model=contract.agent_model,
        canonical_request_hash=agent_hash,
    )
    assert [item[1].outcome for item in hook.settlements] == [
        "succeeded",
        "succeeded",
    ]
    assert [item[2].route for item in hook.settlements if item[2] is not None] == [
        "vertex_ai_adc",
        "openai",
    ]
    assert [item[2].input_tokens for item in hook.settlements if item[2] is not None] == [
        11,
        7,
    ]
    assert [(item.role, item.turn_index, item.outcome) for item in turn_records] == [
        ("agent", 0, "succeeded"),
        ("user", 0, "succeeded"),
    ]
    assert [item.logical_call_id for item in turn_records] == [
        context.logical_call_id for context in hook.contexts
    ]
    assert [item.provider_attempt_id for item in turn_records] == [
        settlement[0].provider_attempt_id for settlement in hook.settlements
    ]


@pytest.mark.parametrize("role", ["agent", "user"])
def test_per_turn_input_bound_stops_before_hook_and_provider(role: str) -> None:
    contract = _turn_contract()
    hook = RecordingTurnHook()
    provider_calls = 0

    def original(**_kwargs: Any) -> Any:
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=None)

    agent_module = SimpleNamespace(generate=original)
    user_module = SimpleNamespace(generate=original)
    api = {
        "llm_agent_module": agent_module,
        "user_simulator_module": user_module,
        "to_litellm_messages": lambda messages: list(messages),
    }
    adapter = Tau2EpisodeExecutor(attempt_hook=hook)
    selected_module = agent_module if role == "agent" else user_module
    model = contract.agent_model if role == "agent" else contract.user_model

    with pytest.raises(EpisodeContractError, match="canonical input bound"):
        with adapter._instrumented_generate(api=api, contract=contract):
            selected_module.generate(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "x" * contract.model_turn_max_input_bound,
                    }
                ],
                **_turn_kwargs(contract, role=role),
            )

    assert provider_calls == 0
    assert hook.contexts == []
    assert hook.settlements == []
    assert agent_module.generate is original
    assert user_module.generate is original


def test_per_turn_pre_hook_failure_makes_zero_provider_calls_and_restores() -> None:
    contract = _turn_contract()
    hook = RecordingTurnHook(fail_before=True)
    provider_calls = 0

    def original(**_kwargs: Any) -> Any:
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=None)

    agent_module = SimpleNamespace(generate=original)
    user_module = SimpleNamespace(generate=original)
    api = {
        "llm_agent_module": agent_module,
        "user_simulator_module": user_module,
        "to_litellm_messages": lambda messages: list(messages),
    }
    adapter = Tau2EpisodeExecutor(attempt_hook=hook)

    with pytest.raises(Tau2ProviderHookError, match="before dispatch"):
        with adapter._instrumented_generate(api=api, contract=contract):
            agent_module.generate(
                model=contract.agent_model,
                messages=[{"role": "system", "content": "policy"}],
                **_turn_kwargs(contract, role="agent"),
            )

    assert provider_calls == 0
    assert hook.settlements == []
    assert agent_module.generate is original
    assert user_module.generate is original


def test_provider_generation_failure_settles_failed_and_never_leaks_wrapper() -> None:
    contract = _turn_contract()
    hook = RecordingTurnHook()

    def failing_original(**_kwargs: Any) -> Any:
        raise TimeoutError("provider timeout with private response")

    agent_module = SimpleNamespace(generate=failing_original)
    user_module = SimpleNamespace(generate=failing_original)
    api = {
        "llm_agent_module": agent_module,
        "user_simulator_module": user_module,
        "to_litellm_messages": lambda messages: list(messages),
    }
    adapter = Tau2EpisodeExecutor(attempt_hook=hook)
    turn_records = []

    with pytest.raises(TimeoutError):
        with adapter._instrumented_generate(
            api=api,
            contract=contract,
            turn_records=turn_records,
        ):
            agent_module.generate(
                model=contract.agent_model,
                messages=[{"role": "system", "content": "policy"}],
                **_turn_kwargs(contract, role="agent"),
            )

    assert len(hook.settlements) == 1
    _, attempt, usage = hook.settlements[0]
    assert attempt.outcome == "failed"
    assert attempt.category == "overall_timeout"
    assert usage is None
    assert len(turn_records) == 1
    assert turn_records[0].outcome == "failed"
    assert turn_records[0].provider_attempt_id == hook.settlements[0][0].provider_attempt_id
    assert agent_module.generate is failing_original
    assert user_module.generate is failing_original


def test_execute_propagates_pre_turn_hook_failure_instead_of_infra_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reproduction.tau2_telecom.harness import tau2_interface

    contract = _turn_contract()
    hook = RecordingTurnHook(fail_before=True)
    provider_calls = 0

    def original_generate(**_kwargs: Any) -> Any:
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(usage=None)

    agent_module = SimpleNamespace(generate=original_generate)
    user_module = SimpleNamespace(generate=original_generate)

    class Environment:
        def get_tools(self) -> list[Any]:
            return []

        def get_user_tools(self) -> list[Any]:
            return []

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            self.system_prompt = "rendered tau2 LLMAgent system prompt"
            self.llm = kwargs["llm"]
            self.llm_args = kwargs["llm_args"]

    class User:
        def __init__(self, **kwargs: Any) -> None:
            self.llm = kwargs["llm"]
            self.llm_args = kwargs["llm_args"]

    class Orchestrator:
        def __init__(self, **kwargs: Any) -> None:
            self.agent = kwargs["agent"]

        def run(self) -> Any:
            agent_module.generate(
                model=self.agent.llm,
                messages=[{"role": "system", "content": "policy"}],
                **self.agent.llm_args,
            )
            raise AssertionError("pre-hook failure should prevent this line")

    api = {
        "EvaluationType": SimpleNamespace(ALL="all"),
        "LLMAgent": Agent,
        "Orchestrator": Orchestrator,
        "UserSimulator": User,
        "evaluate_simulation": lambda **_kwargs: pytest.fail("evaluation was reached"),
        "get_environment_manual_policy": Environment,
        "llm_agent_module": agent_module,
        "to_litellm_messages": lambda messages: list(messages),
        "user_simulator_module": user_module,
    }
    monkeypatch.setattr(tau2_interface, "_optional_tau2_imports", lambda: api)
    adapter = Tau2EpisodeExecutor(attempt_hook=hook)
    monkeypatch.setattr(adapter, "_verify", lambda: None)
    task = SimpleNamespace(id=contract.benchmark_task_id, user_scenario="scenario")

    with pytest.raises(Tau2ProviderHookError, match="before dispatch"):
        adapter.execute(task=task, contract=contract, policy=POLICY)

    assert provider_calls == 0
    assert agent_module.generate is original_generate
    assert user_module.generate is original_generate


def test_formal_real_adapter_refuses_uninstrumented_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _turn_contract().model_copy(update={"require_per_turn_attempt_hook": False})
    adapter = Tau2EpisodeExecutor()
    monkeypatch.setattr(adapter, "_verify", lambda: None)
    task = SimpleNamespace(id=contract.benchmark_task_id, user_scenario="scenario")

    with pytest.raises(EpisodeContractError, match="per-turn provider-attempt hook"):
        adapter.execute(task=task, contract=contract, policy=POLICY)


def test_execute_settles_provider_failure_then_returns_episode_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reproduction.tau2_telecom.harness import tau2_interface

    contract = _turn_contract()
    hook = RecordingTurnHook()

    def failing_generate(**_kwargs: Any) -> Any:
        raise TimeoutError("provider timeout with private response")

    agent_module = SimpleNamespace(generate=failing_generate)
    user_module = SimpleNamespace(generate=failing_generate)

    class Environment:
        def get_tools(self) -> list[Any]:
            return []

        def get_user_tools(self) -> list[Any]:
            return []

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            self.system_prompt = "rendered tau2 LLMAgent system prompt"
            self.llm = kwargs["llm"]
            self.llm_args = kwargs["llm_args"]

    class User:
        def __init__(self, **kwargs: Any) -> None:
            self.llm = kwargs["llm"]
            self.llm_args = kwargs["llm_args"]

    class Orchestrator:
        def __init__(self, **kwargs: Any) -> None:
            self.agent = kwargs["agent"]

        def run(self) -> Any:
            return agent_module.generate(
                model=self.agent.llm,
                messages=[{"role": "system", "content": "policy"}],
                **self.agent.llm_args,
            )

    api = {
        "EvaluationType": SimpleNamespace(ALL="all"),
        "LLMAgent": Agent,
        "Orchestrator": Orchestrator,
        "UserSimulator": User,
        "evaluate_simulation": lambda **_kwargs: pytest.fail("evaluation was reached"),
        "get_environment_manual_policy": Environment,
        "llm_agent_module": agent_module,
        "to_litellm_messages": lambda messages: list(messages),
        "user_simulator_module": user_module,
    }
    monkeypatch.setattr(tau2_interface, "_optional_tau2_imports", lambda: api)
    adapter = Tau2EpisodeExecutor(attempt_hook=hook)
    monkeypatch.setattr(adapter, "_verify", lambda: None)
    task = SimpleNamespace(id=contract.benchmark_task_id, user_scenario="scenario")

    outcome = adapter.execute(task=task, contract=contract, policy=POLICY)

    assert outcome.disposition == "infrastructure_error"
    assert outcome.infrastructure_error_type == "timeout"
    assert len(outcome.provider_turns) == 1
    assert outcome.provider_turns[0].outcome == "failed"
    assert len(hook.settlements) == 1
    assert hook.settlements[0][1].outcome == "failed"
    assert hook.settlements[0][1].category == "overall_timeout"
    assert hook.settlements[0][2] is None
    assert agent_module.generate is failing_generate
    assert user_module.generate is failing_generate


def test_health_and_cost_are_deterministic_views_not_resume_authority() -> None:
    _, tasks, expected_all, config = _profile("experience_smoke_v1.json")
    expected = _single_cell(expected_all)
    task = (
        next(item for item in tasks if _task_id(item) == expected.episodes[0].benchmark_task_id),
    )

    def retry_then_complete(_task: Any, contract: EpisodeAttemptContract) -> EpisodeAttemptOutcome:
        if contract.attempt_index == 0:
            return EpisodeAttemptOutcome.infrastructure(
                contract=contract,
                error_type="timeout",
            )
        return _behavioral(contract, reward=0.0)

    execution = execute_matrix_sequential(
        expected=expected,
        tasks=task,
        policy=POLICY,
        prompt_hash=PROMPT_HASH,
        config=config,
        executor=FakeExecutor(retry_then_complete),
        authority=FakeAuthority(),
    )
    health = derive_health_summary(execution)
    rebuilt_health = derive_health_summary(execution)
    costs = derive_cost_summary(execution.attempts)

    assert health == rebuilt_health
    assert health.source == "derived_from_immutable_authority"
    assert health.infrastructure_attempts == 1
    assert health.behavioral_failures == 1
    assert not hasattr(health, "load_accepted")
    assert not hasattr(costs, "load_attempts")
    # Infrastructure accounting was unavailable, so aggregate usage/cost must
    # remain unknown rather than being fabricated as zero.
    assert costs.agent.input_tokens is None
    assert costs.user_simulator.output_tokens is None
    assert costs.total_cost_usd is None

    completed_only = derive_cost_summary(execution.accepted)
    assert completed_only.agent.input_tokens == 10
    assert completed_only.agent.output_tokens == 2
    assert completed_only.user_simulator.input_tokens == 8
    assert completed_only.user_simulator.output_tokens == 3
    assert completed_only.total_cost_usd == pytest.approx(0.03)

    empty = derive_cost_summary(())
    assert empty.agent.input_tokens is None
    assert empty.agent.output_tokens is None
    assert empty.agent.cost_usd is None
    assert empty.user_simulator.input_tokens is None
    assert empty.total_cost_usd is None
