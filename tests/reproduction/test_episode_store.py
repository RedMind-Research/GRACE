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
import multiprocessing
import os
from pathlib import Path
from queue import Empty

import pytest

from reproduction.tau2_telecom.harness.episode_store import (
    AcceptedEpisodePointer,
    EpisodeConflictError,
    EpisodeIndeterminateError,
    EpisodeIntegrityError,
    EpisodeLeaseHeldError,
    EpisodeRecoveryError,
    EpisodeTerminalError,
    FileEpisodeArtifactAuthority,
    new_owner_token,
)
from reproduction.tau2_telecom.harness.identity import (
    PINNED_TAU2_REVISION,
    canonical_sha256,
    make_episode_id,
    make_matrix_hash,
    make_task_uid,
    text_sha256,
)
from reproduction.tau2_telecom.harness.models import (
    EpisodeCell,
    ExpectedMatrix,
    TrajectoryMessage,
)
from reproduction.tau2_telecom.harness.tau2_interface import (
    EpisodeArtifactAuthority,
    EpisodeAttemptContract,
    EpisodeAttemptOutcome,
    EpisodeHarnessConfig,
    execute_matrix_sequential,
)


RUN_ID = "episode-authority-test"
POLICY = "Verify the account before making changes."
PROMPT_HASH = text_sha256("tau2-agent-system-prompt")


def _config(*, max_steps: int = 200) -> EpisodeHarnessConfig:
    return EpisodeHarnessConfig.create(
        protocol_hash=canonical_sha256({"protocol": "episode-store-test"}),
        max_steps=max_steps,
    )


def _episode(
    config: EpisodeHarnessConfig | None = None,
    *,
    benchmark_task_id: str = "[test]episode_store[PERSONA:None]",
    episode_id_override: str | None = None,
) -> EpisodeCell:
    config = config or _config()
    task_uid = make_task_uid(
        benchmark_task_id,
        benchmark_revision=PINNED_TAU2_REVISION,
    )
    task_definition_hash = canonical_sha256(
        {"definition": {"id": benchmark_task_id}, "namespace": "grace.task-definition.v1"}
    )
    episode_id = make_episode_id(
        task_uid=task_uid,
        task_definition_hash=task_definition_hash,
        split_slot="eval-s000",
        trial=0,
        protocol_seed=1024,
        checkpoint_state_id="state-10",
        benchmark_revision=PINNED_TAU2_REVISION,
        protocol_hash=config.protocol_hash,
        config_hash=config.config_hash,
        model_roles_hash=config.model_roles_hash,
        evaluator_hash=config.evaluator_hash,
    )
    return EpisodeCell(
        episode_id=episode_id_override or episode_id,
        task_uid=task_uid,
        task_definition_hash=task_definition_hash,
        benchmark_task_id=benchmark_task_id,
        split_slot="eval-s000",
        trial=0,
        protocol_seed=1024,
        checkpoint_state_id="state-10",
    )


def _contract(
    *,
    attempt_index: int,
    config: EpisodeHarnessConfig | None = None,
    episode: EpisodeCell | None = None,
    policy: str = POLICY,
    prompt_hash: str = PROMPT_HASH,
) -> EpisodeAttemptContract:
    config = config or _config()
    episode = episode or _episode(config)
    return EpisodeAttemptContract.create(
        episode=episode,
        attempt_index=attempt_index,
        config=config,
        policy_hash=text_sha256(policy),
        prompt_hash=prompt_hash,
    )


def _behavioral(
    contract: EpisodeAttemptContract,
    *,
    reward: float = 1.0,
) -> EpisodeAttemptOutcome:
    return EpisodeAttemptOutcome.behavioral(
        contract=contract,
        reward=reward,
        termination_reason="user_stop",
        messages=(TrajectoryMessage(role="assistant", content="Completed safely."),),
    )


def _infrastructure(contract: EpisodeAttemptContract) -> EpisodeAttemptOutcome:
    return EpisodeAttemptOutcome.infrastructure(contract=contract, error_type="timeout")


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _reserve_and_record(
    store: FileEpisodeArtifactAuthority,
    outcome: EpisodeAttemptOutcome,
) -> None:
    store.reserve_attempt(outcome.contract)
    store.record_attempt(outcome)


def _record_worker(
    root: str,
    payload: dict[str, object],
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    store = FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    outcome = EpisodeAttemptOutcome.model_validate(payload)
    start.wait(timeout=10)
    try:
        store.record_attempt(outcome)
    except EpisodeConflictError:
        results.put("conflict")
    except Exception as exc:  # pragma: no cover - surfaced by the parent test
        results.put(f"error:{type(exc).__name__}")
    else:
        results.put("ok")


def _crash_after_paid_execution_worker(
    root: str,
    contract_payload: dict[str, object],
    paid_marker: str,
) -> None:
    store = FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    contract = EpisodeAttemptContract.model_validate(contract_payload)
    store.reserve_attempt(contract)
    Path(paid_marker).write_text("executor-dispatched", encoding="utf-8")
    os._exit(23)


def _lease_worker(
    root: str,
    owner_token: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    store = FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    start.wait(timeout=10)
    try:
        lease = store.acquire_run_lease(owner_token)
    except EpisodeLeaseHeldError:
        results.put(("held", owner_token))
    except Exception as exc:  # pragma: no cover - surfaced by the parent test
        results.put((f"error:{type(exc).__name__}", owner_token))
    else:
        results.put(("ok", lease.owner_token))


def test_protocol_round_trip_acceptance_and_derived_files_are_not_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    store = FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    assert isinstance(store, EpisodeArtifactAuthority)
    episode = _episode()
    first = _infrastructure(_contract(attempt_index=0, episode=episode))
    second = _behavioral(_contract(attempt_index=1, episode=episode), reward=0.0)

    (root / "derived").mkdir()
    (root / "derived" / "attempts.jsonl").write_text(
        json.dumps({"episode_id": episode.episode_id, "accepted": True}) + "\n",
        encoding="utf-8",
    )
    (root / "health.json").write_text('{"accepted": 99}', encoding="utf-8")
    (root / "cost.json").write_text('{"usd": 0}', encoding="utf-8")
    assert store.load_attempts(episode.episode_id) == ()
    assert store.load_accepted(episode.episode_id) is None

    _reserve_and_record(store, first)
    assert tuple(store.load_attempts(episode.episode_id)) == (first,)
    with pytest.raises(EpisodeConflictError, match="infrastructure"):
        store.accept_attempt(episode.episode_id, first.contract.attempt_id)

    _reserve_and_record(store, second)
    accepted = store.accept_attempt(episode.episode_id, second.contract.attempt_id)
    assert accepted == second
    assert tuple(store.load_attempts(episode.episode_id)) == (first, second)
    assert store.load_accepted(episode.episode_id) == second
    pointer_path = root / "episode_authority" / "episodes" / episode.episode_id / "accepted.json"
    pointer = AcceptedEpisodePointer.model_validate_json(pointer_path.read_text())
    assert pointer.outcome_id == second.outcome_id
    assert len(pointer.outcome_file_hash) == 64


def test_same_attempt_is_idempotent_but_conflict_unknown_and_post_terminal_fail(
    tmp_path: Path,
) -> None:
    store = FileEpisodeArtifactAuthority(tmp_path, run_id=RUN_ID)
    episode = _episode()
    contract0 = _contract(attempt_index=0, episode=episode)
    failed = _behavioral(contract0, reward=0.0)
    conflicting = _behavioral(contract0, reward=1.0)

    _reserve_and_record(store, failed)
    before = _tree_snapshot(tmp_path)
    store.record_attempt(failed)
    assert _tree_snapshot(tmp_path) == before
    with pytest.raises(EpisodeConflictError, match="different outcome"):
        store.record_attempt(conflicting)
    with pytest.raises(EpisodeConflictError, match="unknown"):
        store.accept_attempt(episode.episode_id, "f" * 64)

    later = _infrastructure(_contract(attempt_index=1, episode=episode))
    with pytest.raises(EpisodeTerminalError, match="terminal"):
        store.reserve_attempt(later.contract)


def test_outcome_without_exact_pre_execution_reservation_is_rejected(
    tmp_path: Path,
) -> None:
    store = FileEpisodeArtifactAuthority(tmp_path, run_id=RUN_ID)
    outcome = _behavioral(_contract(attempt_index=0))

    with pytest.raises(EpisodeConflictError, match="no pre-execution reservation"):
        store.record_attempt(outcome)


def test_wrong_task_config_and_policy_fingerprints_fail_closed(tmp_path: Path) -> None:
    config = _config()
    canonical_episode = _episode(config)

    wrong_task_episode = _episode(
        config,
        benchmark_task_id="[wrong]task[PERSONA:None]",
        episode_id_override=canonical_episode.episode_id,
    )
    wrong_task = _infrastructure(
        _contract(attempt_index=0, config=config, episode=wrong_task_episode)
    )
    with pytest.raises(EpisodeIntegrityError, match="identity"):
        FileEpisodeArtifactAuthority(tmp_path / "wrong-task", run_id=RUN_ID).reserve_attempt(
            wrong_task.contract
        )

    wrong_config = _config(max_steps=199)
    wrong_config_outcome = _infrastructure(
        _contract(
            attempt_index=0,
            config=wrong_config,
            episode=canonical_episode,
        )
    )
    with pytest.raises(EpisodeIntegrityError, match="identity"):
        FileEpisodeArtifactAuthority(tmp_path / "wrong-config", run_id=RUN_ID).reserve_attempt(
            wrong_config_outcome.contract
        )

    store = FileEpisodeArtifactAuthority(tmp_path / "wrong-policy", run_id=RUN_ID)
    first = _infrastructure(_contract(attempt_index=0, config=config, episode=canonical_episode))
    _reserve_and_record(store, first)
    mismatched_policy = _behavioral(
        _contract(
            attempt_index=1,
            config=config,
            episode=canonical_episode,
            policy="A different policy fingerprint.",
        )
    )
    with pytest.raises(EpisodeConflictError, match="fingerprints"):
        store.reserve_attempt(mismatched_policy.contract)


@pytest.mark.parametrize(
    "target",
    ["outcome", "outcome_deleted", "reservation_deleted", "manifest", "accepted"],
)
def test_tampered_authority_records_are_rejected(tmp_path: Path, target: str) -> None:
    store = FileEpisodeArtifactAuthority(tmp_path, run_id=RUN_ID)
    episode = _episode()
    outcome = _behavioral(_contract(attempt_index=0, episode=episode))
    _reserve_and_record(store, outcome)
    store.accept_attempt(episode.episode_id, outcome.contract.attempt_id)
    episode_dir = tmp_path / "episode_authority" / "episodes" / episode.episode_id
    paths = {
        "outcome": (episode_dir / "attempts" / outcome.contract.attempt_id / "outcome.json"),
        "outcome_deleted": (
            episode_dir / "attempts" / outcome.contract.attempt_id / "outcome.json"
        ),
        "reservation_deleted": (
            episode_dir / "attempts" / outcome.contract.attempt_id / "reservation.json"
        ),
        "manifest": episode_dir / "episode.json",
        "accepted": episode_dir / "accepted.json",
    }
    path = paths[target]
    if target in {"outcome_deleted", "reservation_deleted"}:
        path.unlink()
        with pytest.raises(EpisodeIntegrityError):
            store.load_attempts(episode.episode_id)
        with pytest.raises(EpisodeIntegrityError):
            store.load_accepted(episode.episode_id)
        return
    payload = json.loads(path.read_text())
    if target == "outcome":
        payload["termination_reason"] = "tampered"
    elif target == "manifest":
        payload["run_id"] = "tampered-run"
    else:
        payload["outcome_file_hash"] = "0" * 64
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(EpisodeIntegrityError):
        store.load_attempts(episode.episode_id)
    with pytest.raises(EpisodeIntegrityError):
        store.load_accepted(episode.episode_id)


def test_path_symlink_and_debris_are_rejected(tmp_path: Path) -> None:
    store = FileEpisodeArtifactAuthority(tmp_path / "path", run_id=RUN_ID)
    with pytest.raises(EpisodeIntegrityError, match="canonical"):
        store.load_attempts("../escape")

    episode = _episode()
    outcome = _behavioral(_contract(attempt_index=0, episode=episode))
    _reserve_and_record(store, outcome)
    attempts_dir = (
        tmp_path / "path" / "episode_authority" / "episodes" / episode.episode_id / "attempts"
    )
    (attempts_dir / "debris.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(EpisodeIntegrityError, match="debris"):
        store.load_attempts(episode.episode_id)

    symlink_root = tmp_path / "symlink"
    symlink_store = FileEpisodeArtifactAuthority(symlink_root, run_id=RUN_ID)
    _reserve_and_record(symlink_store, outcome)
    outcome_path = (
        symlink_root
        / "episode_authority"
        / "episodes"
        / episode.episode_id
        / "attempts"
        / outcome.contract.attempt_id
        / "outcome.json"
    )
    external = tmp_path / "external.json"
    external.write_bytes(outcome_path.read_bytes())
    outcome_path.unlink()
    outcome_path.symlink_to(external)
    with pytest.raises(EpisodeIntegrityError, match="regular file"):
        symlink_store.load_attempts(episode.episode_id)


def test_multiprocess_same_attempt_is_idempotent_and_conflict_has_one_winner(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    episode = _episode()
    contract = _contract(attempt_index=0, episode=episode)
    same = _behavioral(contract, reward=0.0)

    same_root = tmp_path / "same"
    same_authority = FileEpisodeArtifactAuthority(same_root, run_id=RUN_ID)
    same_authority.reserve_attempt(contract)
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_record_worker,
            args=(str(same_root), same.model_dump(mode="json"), start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert sorted(results.get(timeout=2) for _ in processes) == ["ok", "ok"]
    same_store = FileEpisodeArtifactAuthority(same_root, run_id=RUN_ID)
    assert tuple(same_store.load_attempts(episode.episode_id)) == (same,)

    conflict_root = tmp_path / "conflict"
    conflict_authority = FileEpisodeArtifactAuthority(conflict_root, run_id=RUN_ID)
    conflict_authority.reserve_attempt(contract)
    start = context.Event()
    results = context.Queue()
    conflicting = (_behavioral(contract, reward=0.0), _behavioral(contract, reward=1.0))
    processes = [
        context.Process(
            target=_record_worker,
            args=(str(conflict_root), item.model_dump(mode="json"), start, results),
        )
        for item in conflicting
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert sorted(results.get(timeout=2) for _ in processes) == ["conflict", "ok"]


def test_hard_crash_temp_attempt_is_never_accepted_or_silently_retried(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "crash"
    FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    episode = _episode()
    outcome = _behavioral(_contract(attempt_index=0, episode=episode))
    paid_marker = tmp_path / "paid-execution.marker"
    process = context.Process(
        target=_crash_after_paid_execution_worker,
        args=(str(root), outcome.contract.model_dump(mode="json"), str(paid_marker)),
    )
    process.start()
    process.join(timeout=15)
    assert process.exitcode == 23
    assert paid_marker.read_text(encoding="utf-8") == "executor-dispatched"

    store = FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    with pytest.raises(EpisodeIndeterminateError, match="manual resolution"):
        store.load_attempts(episode.episode_id)
    with pytest.raises(EpisodeIndeterminateError):
        store.load_accepted(episode.episode_id)
    with pytest.raises(EpisodeIndeterminateError, match="dispatch blocked"):
        store.reserve_attempt(outcome.contract)
    episode_dir = root / "episode_authority" / "episodes" / episode.episode_id
    assert not (episode_dir / "accepted.json").exists()


def test_pending_reservation_stops_runner_before_executor_and_manual_resolution_is_additive(
    tmp_path: Path,
) -> None:
    config = _config()
    episode = _episode(config)
    contract0 = _contract(attempt_index=0, config=config, episode=episode)
    store = FileEpisodeArtifactAuthority(tmp_path, run_id=RUN_ID)
    owner = new_owner_token()
    store.acquire_run_lease(owner)
    store.acquire_episode_claim(episode.episode_id, owner)
    store.reserve_attempt(contract0)

    class CountingExecutor:
        calls = 0

        def execute(self, **_kwargs: object) -> EpisodeAttemptOutcome:
            self.calls += 1
            return _behavioral(contract0)

    executor = CountingExecutor()
    expected = ExpectedMatrix(
        episodes=(episode,),
        matrix_hash=make_matrix_hash((episode.episode_id,)),
    )
    with pytest.raises(EpisodeIndeterminateError, match="manual resolution"):
        execute_matrix_sequential(
            expected=expected,
            tasks=({"id": episode.benchmark_task_id},),
            policy=POLICY,
            prompt_hash=PROMPT_HASH,
            config=config,
            executor=executor,
            authority=store,
        )
    assert executor.calls == 0
    store.release_episode_claim(episode.episode_id, owner)

    indeterminate = store.mark_episode_attempt_indeterminate(
        episode.episode_id,
        contract0.attempt_id,
        owner_token=owner,
        reason_code="operator_confirmed_unknown_result",
    )
    assert indeterminate.disposition == "infrastructure_error"
    assert indeterminate.infrastructure_error_type == (
        "indeterminate_operator_confirmed_unknown_result"
    )
    assert indeterminate.accounting.agent_input_tokens is None
    assert indeterminate.accounting.agent_output_tokens is None
    assert indeterminate.accounting.user_input_tokens is None
    assert indeterminate.accounting.user_output_tokens is None
    assert indeterminate.accounting.agent_cost_usd is None
    assert indeterminate.accounting.user_cost_usd is None
    attempt_dir = (
        tmp_path
        / "episode_authority"
        / "episodes"
        / episode.episode_id
        / "attempts"
        / contract0.attempt_id
    )
    assert (attempt_dir / "reservation.json").is_file()
    assert (attempt_dir / "outcome.json").is_file()
    before = _tree_snapshot(tmp_path)
    assert (
        store.mark_episode_attempt_indeterminate(
            episode.episode_id,
            contract0.attempt_id,
            owner_token=owner,
            reason_code="operator_confirmed_unknown_result",
        )
        == indeterminate
    )
    assert _tree_snapshot(tmp_path) == before

    store.acquire_episode_claim(episode.episode_id, owner)
    contract1 = _contract(attempt_index=1, config=config, episode=episode)
    store.reserve_attempt(contract1)
    completed = _behavioral(contract1)
    store.record_attempt(completed)
    assert store.accept_attempt(episode.episode_id, contract1.attempt_id) == completed
    assert tuple(store.load_attempts(episode.episode_id)) == (indeterminate, completed)
    store.release_episode_claim(episode.episode_id, owner)
    store.release_run_lease(owner)


def test_completed_resume_and_idempotent_publish_are_zero_mutation(tmp_path: Path) -> None:
    store = FileEpisodeArtifactAuthority(tmp_path, run_id=RUN_ID)
    episode = _episode()
    first = _infrastructure(_contract(attempt_index=0, episode=episode))
    second = _behavioral(_contract(attempt_index=1, episode=episode))
    _reserve_and_record(store, first)
    _reserve_and_record(store, second)
    store.accept_attempt(episode.episode_id, second.contract.attempt_id)
    before = _tree_snapshot(tmp_path)

    assert tuple(store.load_attempts(episode.episode_id)) == (first, second)
    assert store.load_accepted(episode.episode_id) == second
    store.record_attempt(second)
    assert store.accept_attempt(episode.episode_id, second.contract.attempt_id) == second

    assert _tree_snapshot(tmp_path) == before


def test_lease_claim_never_auto_steals_and_requires_explicit_recovery(tmp_path: Path) -> None:
    store = FileEpisodeArtifactAuthority(tmp_path, run_id=RUN_ID)
    owner = new_owner_token()
    other = new_owner_token()
    operator = new_owner_token()
    episode_id = _episode().episode_id
    lease = store.acquire_run_lease(
        owner,
        started_at_utc="2000-01-01T00:00:00Z",
        pid_diagnostic=999999,
    )
    assert lease.owner_token == owner
    with pytest.raises(EpisodeLeaseHeldError, match="cannot authorize stealing"):
        store.acquire_run_lease(other)

    claim = store.acquire_episode_claim(
        episode_id,
        owner,
        started_at_utc="2000-01-01T00:00:00Z",
        pid_diagnostic=999999,
    )
    assert claim.owner_token == owner
    with pytest.raises(EpisodeLeaseHeldError, match="claims"):
        store.release_run_lease(owner)
    with pytest.raises(EpisodeRecoveryError, match="expectation"):
        store.manual_recover_episode_claim(
            episode_id,
            expected_owner_token=other,
            operator_token=operator,
            reason_code="operator_reviewed_stale_owner",
        )

    claim_recovery = store.manual_recover_episode_claim(
        episode_id,
        expected_owner_token=owner,
        operator_token=operator,
        reason_code="operator_reviewed_stale_owner",
        recovered_at_utc="2026-07-11T00:00:00Z",
    )
    assert claim_recovery.target_kind == "episode_claim"
    lease_recovery = store.manual_recover_run_lease(
        expected_owner_token=owner,
        operator_token=operator,
        reason_code="operator_reviewed_stale_owner",
        recovered_at_utc="2026-07-11T00:01:00Z",
    )
    assert lease_recovery.target_kind == "run_lease"
    assert len(store.load_manual_recoveries()) == 2
    assert store.load_run_lease() is None
    assert store.load_episode_claims() == ()
    assert store.acquire_run_lease(other).owner_token == other
    store.release_run_lease(other)


def test_multiprocess_run_lease_has_exactly_one_owner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "lease-race"
    FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    tokens = (new_owner_token(), new_owner_token())
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_lease_worker, args=(str(root), token, start, results))
        for token in tokens
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    try:
        outcomes = [results.get(timeout=2) for _ in processes]
    except Empty as exc:  # pragma: no cover - diagnostic for broken child process
        raise AssertionError("lease worker did not report outcome") from exc
    assert sorted(item[0] for item in outcomes) == ["held", "ok"]
    store = FileEpisodeArtifactAuthority(root, run_id=RUN_ID)
    lease = store.load_run_lease()
    assert lease is not None
    assert lease.owner_token == next(item[1] for item in outcomes if item[0] == "ok")
