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

"""Immutable filesystem authority for tau2 episode attempts and acceptance.

Provider-attempt reservations live in :mod:`attempt_store`; this module is a
separate episode-level authority.  Resume reads only ``episode_authority``
records.  JSONL, health, cost, and other derived files are never consulted.

Every published model is canonical JSON written to an exclusive temporary
file, fsynced, and atomically hard-linked into its create-only final path.
Recognized crash debris is deliberately *not* ignored: it blocks automatic
resume until an operator investigates.  Persistent run leases and episode
claims are never stolen based on PID or elapsed time; explicit recovery APIs
require the exact prior owner token and write an immutable recovery record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .identity import (
    canonical_json,
    canonical_sha256,
    make_episode_id,
    make_task_uid,
    validate_sha256,
)
from .tau2_interface import (
    EpisodeArtifactAuthority,
    EpisodeAttemptContract,
    EpisodeAttemptOutcome,
)


try:
    import fcntl
except ImportError:  # pragma: no cover - the supported release runners are POSIX.
    fcntl = None  # type: ignore[assignment]


FrozenModelConfig = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1, max_length=512)]
FaultInjector = Callable[[str, Path], None]

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REASON_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_HEX_FILE = re.compile(r"^([0-9a-f]{64})\.json$")
_MAX_RECORD_BYTES = 50_000_000
_MAX_ATTEMPTS = 2


class EpisodeStoreError(RuntimeError):
    """Base class for safe episode-authority failures."""


class EpisodeConflictError(EpisodeStoreError):
    """Raised when create-only episode content conflicts with authority."""


class EpisodeIntegrityError(EpisodeStoreError):
    """Raised for malformed, tampered, incomplete, or debris-bearing authority."""


class EpisodeTerminalError(EpisodeStoreError):
    """Raised when an episode's behavioral terminal state forbids another attempt."""


class EpisodeIndeterminateError(EpisodeStoreError):
    """Raised when a reserved attempt lacks a durable terminal outcome."""


class EpisodeLeaseHeldError(EpisodeStoreError):
    """Raised when a persistent lease/claim belongs to another owner."""


class EpisodeRecoveryError(EpisodeStoreError):
    """Raised when explicit human recovery expectations do not match authority."""


def _safe_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise EpisodeStoreError("run_id is not a safe identifier")
    return value


def _safe_reason(value: str) -> str:
    if not _REASON_CODE.fullmatch(value):
        raise EpisodeRecoveryError("recovery reason_code is not a safe identifier")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("timestamp must be ISO-8601 UTC") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must use UTC")
    return value


def _model_hash(model: BaseModel, *, exclude: set[str]) -> str:
    return canonical_sha256(model.model_dump(mode="json", exclude=exclude))


def _payload_bytes(model: BaseModel) -> bytes:
    return canonical_json(model.model_dump(mode="json")).encode("utf-8") + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def new_owner_token() -> str:
    """Return a cryptographically random opaque 64-hex lease owner token."""

    return secrets.token_hex(32)


class EpisodeScope(BaseModel):
    """Attempt-invariant task, policy, provider-role, and runtime fingerprints."""

    model_config = FrozenModelConfig

    episode_id: NonEmpty
    task_uid: NonEmpty
    task_definition_hash: NonEmpty
    benchmark_task_id: NonEmpty
    split_slot: NonEmpty
    trial: int = Field(ge=0)
    protocol_seed: int
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
    max_steps: int = Field(ge=1)
    max_errors: int = Field(ge=1)
    enforce_communication_protocol: bool = False

    @classmethod
    def from_contract(cls, contract: EpisodeAttemptContract) -> EpisodeScope:
        fields = cls.model_fields
        return cls.model_validate({name: getattr(contract, name) for name in fields})

    @model_validator(mode="after")
    def validate_identities(self) -> EpisodeScope:
        for name in (
            "episode_id",
            "task_uid",
            "task_definition_hash",
            "policy_hash",
            "prompt_hash",
            "protocol_hash",
            "model_roles_hash",
            "evaluator_hash",
            "config_hash",
        ):
            validate_sha256(getattr(self, name), field=name)
        expected_task_uid = make_task_uid(
            self.benchmark_task_id,
            benchmark_revision=self.benchmark_revision,
            domain="telecom",
        )
        if self.task_uid != expected_task_uid:
            raise ValueError("task_uid does not match benchmark task/revision")
        expected_episode_id = make_episode_id(
            task_uid=self.task_uid,
            task_definition_hash=self.task_definition_hash,
            split_slot=self.split_slot,
            trial=self.trial,
            protocol_seed=self.protocol_seed,
            checkpoint_state_id=self.checkpoint_state_id,
            benchmark_revision=self.benchmark_revision,
            protocol_hash=self.protocol_hash,
            config_hash=self.config_hash,
            model_roles_hash=self.model_roles_hash,
            evaluator_hash=self.evaluator_hash,
        )
        if self.episode_id != expected_episode_id:
            raise ValueError("episode_id does not match task/runtime fingerprints")
        return self

    @property
    def scope_hash(self) -> str:
        return canonical_sha256(
            {
                "namespace": "grace.episode-authority-scope.v1",
                "scope": self.model_dump(mode="json"),
            }
        )


class EpisodeAuthorityRun(BaseModel):
    """Deterministic marker preventing two run IDs from sharing an authority root."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-authority-run.v1"] = "grace.episode-authority-run.v1"
    run_id: NonEmpty
    authority_hash: NonEmpty

    @classmethod
    def create(cls, run_id: str) -> EpisodeAuthorityRun:
        _safe_run_id(run_id)
        values = {
            "schema_version": "grace.episode-authority-run.v1",
            "run_id": run_id,
        }
        return cls.model_validate({**values, "authority_hash": canonical_sha256(values)})

    @model_validator(mode="after")
    def validate_authority(self) -> EpisodeAuthorityRun:
        _safe_run_id(self.run_id)
        validate_sha256(self.authority_hash, field="authority_hash")
        if self.authority_hash != _model_hash(self, exclude={"authority_hash"}):
            raise ValueError("authority_hash does not match run marker")
        return self


class EpisodeAuthorityManifest(BaseModel):
    """Immutable attempt-invariant fingerprint for one episode directory."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-authority-manifest.v1"] = (
        "grace.episode-authority-manifest.v1"
    )
    run_id: NonEmpty
    episode_id: NonEmpty
    scope: EpisodeScope
    scope_hash: NonEmpty
    manifest_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        contract: EpisodeAttemptContract,
    ) -> EpisodeAuthorityManifest:
        scope = EpisodeScope.from_contract(contract)
        values = {
            "schema_version": "grace.episode-authority-manifest.v1",
            "run_id": run_id,
            "episode_id": scope.episode_id,
            "scope": scope,
            "scope_hash": scope.scope_hash,
        }
        return cls.model_validate({**values, "manifest_hash": canonical_sha256(values)})

    @model_validator(mode="after")
    def validate_manifest(self) -> EpisodeAuthorityManifest:
        _safe_run_id(self.run_id)
        for name in ("episode_id", "scope_hash", "manifest_hash"):
            validate_sha256(getattr(self, name), field=name)
        if self.episode_id != self.scope.episode_id:
            raise ValueError("episode manifest identity differs from scope")
        if self.scope_hash != self.scope.scope_hash:
            raise ValueError("scope_hash does not match episode scope")
        if self.manifest_hash != _model_hash(self, exclude={"manifest_hash"}):
            raise ValueError("manifest_hash does not match episode manifest")
        return self


class EpisodeAttemptReservation(BaseModel):
    """Create-only pre-dispatch authority for one exact attempt contract."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-attempt-reservation.v1"] = (
        "grace.episode-attempt-reservation.v1"
    )
    run_id: NonEmpty
    episode_id: NonEmpty
    attempt_id: NonEmpty
    scope_hash: NonEmpty
    contract: EpisodeAttemptContract
    reserved_at_utc: NonEmpty
    pid_diagnostic: int = Field(ge=1)
    reservation_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        contract: EpisodeAttemptContract,
        reserved_at_utc: str,
        pid_diagnostic: int,
    ) -> EpisodeAttemptReservation:
        scope = EpisodeScope.from_contract(contract)
        values = {
            "schema_version": "grace.episode-attempt-reservation.v1",
            "run_id": run_id,
            "episode_id": contract.episode_id,
            "attempt_id": contract.attempt_id,
            "scope_hash": scope.scope_hash,
            "contract": contract,
            "reserved_at_utc": reserved_at_utc,
            "pid_diagnostic": pid_diagnostic,
        }
        return cls.model_validate({**values, "reservation_hash": canonical_sha256(values)})

    @model_validator(mode="after")
    def validate_reservation(self) -> EpisodeAttemptReservation:
        _safe_run_id(self.run_id)
        for name in (
            "episode_id",
            "attempt_id",
            "scope_hash",
            "reservation_hash",
        ):
            validate_sha256(getattr(self, name), field=name)
        _validate_utc(self.reserved_at_utc)
        scope = EpisodeScope.from_contract(self.contract)
        if self.episode_id != self.contract.episode_id:
            raise ValueError("reservation episode differs from contract")
        if self.attempt_id != self.contract.attempt_id:
            raise ValueError("reservation attempt differs from contract")
        if self.scope_hash != scope.scope_hash:
            raise ValueError("reservation scope_hash differs from contract")
        if self.reservation_hash != _model_hash(self, exclude={"reservation_hash"}):
            raise ValueError("reservation_hash does not match reservation content")
        return self


class EpisodeAttemptAuthorityRecord(BaseModel):
    """Verified reservation and its optional immutable outcome."""

    model_config = FrozenModelConfig

    reservation: EpisodeAttemptReservation
    outcome: EpisodeAttemptOutcome | None = None

    @property
    def pending(self) -> bool:
        return self.outcome is None


class AcceptedEpisodePointer(BaseModel):
    """Create-only pointer binding acceptance to exact immutable outcome bytes."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.accepted-episode-pointer.v1"] = (
        "grace.accepted-episode-pointer.v1"
    )
    episode_id: NonEmpty
    attempt_id: NonEmpty
    attempt_index: int = Field(ge=0, le=1)
    outcome_id: NonEmpty
    outcome_file_hash: NonEmpty
    pointer_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        outcome: EpisodeAttemptOutcome,
        outcome_file_hash: str,
    ) -> AcceptedEpisodePointer:
        values = {
            "schema_version": "grace.accepted-episode-pointer.v1",
            "episode_id": outcome.contract.episode_id,
            "attempt_id": outcome.contract.attempt_id,
            "attempt_index": outcome.contract.attempt_index,
            "outcome_id": outcome.outcome_id,
            "outcome_file_hash": outcome_file_hash,
        }
        return cls.model_validate({**values, "pointer_hash": canonical_sha256(values)})

    @model_validator(mode="after")
    def validate_pointer(self) -> AcceptedEpisodePointer:
        for name in (
            "episode_id",
            "attempt_id",
            "outcome_id",
            "outcome_file_hash",
            "pointer_hash",
        ):
            validate_sha256(getattr(self, name), field=name)
        if self.pointer_hash != _model_hash(self, exclude={"pointer_hash"}):
            raise ValueError("pointer_hash does not match accepted pointer")
        return self


class RunExecutionLease(BaseModel):
    """Persistent owner record; PID and time are diagnostic, never liveness proof."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.run-execution-lease.v1"] = "grace.run-execution-lease.v1"
    run_id: NonEmpty
    owner_token: NonEmpty
    started_at_utc: NonEmpty
    pid_diagnostic: int = Field(ge=1)
    lease_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        owner_token: str,
        started_at_utc: str,
        pid_diagnostic: int,
    ) -> RunExecutionLease:
        values = {
            "schema_version": "grace.run-execution-lease.v1",
            "run_id": run_id,
            "owner_token": owner_token,
            "started_at_utc": started_at_utc,
            "pid_diagnostic": pid_diagnostic,
        }
        return cls.model_validate({**values, "lease_hash": canonical_sha256(values)})

    @model_validator(mode="after")
    def validate_lease(self) -> RunExecutionLease:
        _safe_run_id(self.run_id)
        validate_sha256(self.owner_token, field="owner_token")
        validate_sha256(self.lease_hash, field="lease_hash")
        _validate_utc(self.started_at_utc)
        if self.lease_hash != _model_hash(self, exclude={"lease_hash"}):
            raise ValueError("lease_hash does not match run lease")
        return self


def _claim_id(*, run_id: str, episode_id: str) -> str:
    return canonical_sha256(
        {
            "episode_id": episode_id,
            "namespace": "grace.episode-execution-claim.v1",
            "run_id": run_id,
        }
    )


class EpisodeExecutionClaim(BaseModel):
    """Persistent per-episode claim held under one active run lease owner."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.episode-execution-claim.v1"] = "grace.episode-execution-claim.v1"
    claim_id: NonEmpty
    run_id: NonEmpty
    episode_id: NonEmpty
    owner_token: NonEmpty
    started_at_utc: NonEmpty
    pid_diagnostic: int = Field(ge=1)
    claim_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        episode_id: str,
        owner_token: str,
        started_at_utc: str,
        pid_diagnostic: int,
    ) -> EpisodeExecutionClaim:
        values = {
            "schema_version": "grace.episode-execution-claim.v1",
            "claim_id": _claim_id(run_id=run_id, episode_id=episode_id),
            "run_id": run_id,
            "episode_id": episode_id,
            "owner_token": owner_token,
            "started_at_utc": started_at_utc,
            "pid_diagnostic": pid_diagnostic,
        }
        return cls.model_validate({**values, "claim_hash": canonical_sha256(values)})

    @model_validator(mode="after")
    def validate_claim(self) -> EpisodeExecutionClaim:
        _safe_run_id(self.run_id)
        for name in ("claim_id", "episode_id", "owner_token", "claim_hash"):
            validate_sha256(getattr(self, name), field=name)
        _validate_utc(self.started_at_utc)
        if self.claim_id != _claim_id(run_id=self.run_id, episode_id=self.episode_id):
            raise ValueError("claim_id does not match run/episode")
        if self.claim_hash != _model_hash(self, exclude={"claim_hash"}):
            raise ValueError("claim_hash does not match episode claim")
        return self


class ManualRecoveryRecord(BaseModel):
    """Immutable audit record for an explicit operator-directed unlock."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.manual-control-recovery.v1"] = "grace.manual-control-recovery.v1"
    recovery_id: NonEmpty
    run_id: NonEmpty
    target_kind: Literal["run_lease", "episode_claim"]
    target_id: NonEmpty
    target_record_hash: NonEmpty
    expected_owner_token: NonEmpty
    operator_token: NonEmpty
    reason_code: NonEmpty
    recovered_at_utc: NonEmpty
    recovery_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        target_kind: Literal["run_lease", "episode_claim"],
        target_id: str,
        target_record_hash: str,
        expected_owner_token: str,
        operator_token: str,
        reason_code: str,
        recovered_at_utc: str,
    ) -> ManualRecoveryRecord:
        identity = {
            "expected_owner_token": expected_owner_token,
            "namespace": "grace.manual-control-recovery-id.v1",
            "operator_token": operator_token,
            "reason_code": reason_code,
            "run_id": run_id,
            "target_id": target_id,
            "target_kind": target_kind,
            "target_record_hash": target_record_hash,
        }
        values = {
            "schema_version": "grace.manual-control-recovery.v1",
            "recovery_id": canonical_sha256(identity),
            "run_id": run_id,
            "target_kind": target_kind,
            "target_id": target_id,
            "target_record_hash": target_record_hash,
            "expected_owner_token": expected_owner_token,
            "operator_token": operator_token,
            "reason_code": reason_code,
            "recovered_at_utc": recovered_at_utc,
        }
        return cls.model_validate({**values, "recovery_hash": canonical_sha256(values)})

    @model_validator(mode="after")
    def validate_recovery(self) -> ManualRecoveryRecord:
        _safe_run_id(self.run_id)
        _safe_reason(self.reason_code)
        for name in (
            "recovery_id",
            "target_id",
            "target_record_hash",
            "expected_owner_token",
            "operator_token",
            "recovery_hash",
        ):
            validate_sha256(getattr(self, name), field=name)
        _validate_utc(self.recovered_at_utc)
        identity = {
            "expected_owner_token": self.expected_owner_token,
            "namespace": "grace.manual-control-recovery-id.v1",
            "operator_token": self.operator_token,
            "reason_code": self.reason_code,
            "run_id": self.run_id,
            "target_id": self.target_id,
            "target_kind": self.target_kind,
            "target_record_hash": self.target_record_hash,
        }
        if self.recovery_id != canonical_sha256(identity):
            raise ValueError("recovery_id does not match recovery target")
        if self.recovery_hash != _model_hash(self, exclude={"recovery_hash"}):
            raise ValueError("recovery_hash does not match recovery record")
        return self


class FileEpisodeArtifactAuthority(EpisodeArtifactAuthority):
    """Filesystem-backed implementation of ``EpisodeArtifactAuthority``."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        self.run_id = _safe_run_id(run_id)
        self.authority_dir = self.root / "episode_authority"
        self.episodes_dir = self.authority_dir / "episodes"
        self.run_manifest_path = self.authority_dir / "run.json"
        self.control_dir = self.root / "control"
        self.claims_dir = self.control_dir / "claims"
        self.recoveries_dir = self.control_dir / "recoveries"
        self.lease_path = self.control_dir / "run_lease.json"
        self._lock_path = self.root / ".episode-authority.lock"
        self._fault_injector = fault_injector

        for path in (
            self.root,
            self.authority_dir,
            self.episodes_dir,
            self.control_dir,
            self.claims_dir,
            self.recoveries_dir,
        ):
            self._ensure_directory(path)
        marker = EpisodeAuthorityRun.create(self.run_id)
        # Initialization publishes through the same process lock as every later
        # mutation.  Without this boundary, another constructor can observe the
        # create-only publisher's temporary file and mistake it for debris.
        with self._locked(exclusive=True, validate=False):
            self._atomic_create_model(
                self.run_manifest_path,
                marker,
                event="before_run_manifest_publish",
                identical_ok=True,
            )
            loaded_marker = self._read_model(self.run_manifest_path, EpisodeAuthorityRun)
            if loaded_marker != marker:
                raise EpisodeIntegrityError("episode authority root belongs to another run")
            self._validate_namespace_layout()

    def _fault(self, event: str, path: Path) -> None:
        if self._fault_injector is None:
            return
        try:
            self._fault_injector(event, path)
        except EpisodeStoreError:
            raise
        except Exception:
            raise EpisodeStoreError("episode authority fault boundary failed") from None

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = path.lstat()
        except OSError:
            raise EpisodeStoreError("episode authority directory could not be created") from None
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise EpisodeIntegrityError("episode authority path is not a real directory")

    @staticmethod
    def _require_directory(path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError:
            raise EpisodeIntegrityError("episode authority directory is missing") from None
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise EpisodeIntegrityError("episode authority entry is not a real directory")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def _locked(self, *, exclusive: bool, validate: bool = True) -> Iterator[None]:
        if fcntl is None:
            raise EpisodeStoreError("episode authority requires POSIX file locking")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._lock_path, flags, 0o600)
        except OSError:
            raise EpisodeStoreError("episode authority lock could not be opened") from None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            if validate:
                self._validate_namespace_layout()
            yield
        except EpisodeStoreError:
            raise
        except OSError:
            raise EpisodeStoreError("episode authority lock operation failed") from None
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _validate_namespace_layout(self) -> None:
        """Reject authority/control debris while ignoring derived sibling namespaces."""

        try:
            authority_names = {path.name for path in self.authority_dir.iterdir()}
            control_names = {path.name for path in self.control_dir.iterdir()}
        except OSError:
            raise EpisodeIntegrityError("episode authority namespace could not be read") from None
        if authority_names != {"episodes", "run.json"}:
            raise EpisodeIntegrityError("episode authority namespace contains debris")
        marker = self._read_model(self.run_manifest_path, EpisodeAuthorityRun)
        if marker.run_id != self.run_id:
            raise EpisodeIntegrityError("episode authority run marker mismatch")
        if not {"claims", "recoveries"} <= control_names or not control_names <= {
            "claims",
            "recoveries",
            "run_lease.json",
        }:
            raise EpisodeIntegrityError("episode control namespace contains debris")
        self._require_directory(self.episodes_dir)
        self._require_directory(self.claims_dir)
        self._require_directory(self.recoveries_dir)
        for directory, label in (
            (self.episodes_dir, "episode"),
            (self.claims_dir, "claim"),
            (self.recoveries_dir, "recovery"),
        ):
            try:
                children = tuple(directory.iterdir())
            except OSError:
                raise EpisodeIntegrityError(
                    f"episode {label} namespace could not be enumerated"
                ) from None
            for child in children:
                if directory == self.episodes_dir:
                    try:
                        validate_sha256(child.name, field="episode_id")
                        self._require_directory(child)
                    except Exception:
                        raise EpisodeIntegrityError(
                            "episode namespace contains an invalid path"
                        ) from None
                elif _HEX_FILE.fullmatch(child.name) is None:
                    raise EpisodeIntegrityError(f"episode {label} namespace contains debris")
                else:
                    try:
                        metadata = child.lstat()
                    except OSError:
                        raise EpisodeIntegrityError(
                            f"episode {label} record could not be inspected"
                        ) from None
                    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                        raise EpisodeIntegrityError(f"episode {label} record is not a regular file")

    def _atomic_create_model(
        self,
        path: Path,
        model: BaseModel,
        *,
        event: str,
        identical_ok: bool,
    ) -> bool:
        payload = _payload_bytes(model)
        if len(payload) > _MAX_RECORD_BYTES:
            raise EpisodeStoreError("episode authority record exceeds size limit")
        if path.exists():
            existing = self._read_bytes(path)
            if identical_ok and existing == payload:
                return False
            raise EpisodeConflictError("create-only episode authority record already exists")

        temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        published = False
        try:
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._fault(event, path)
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                existing = self._read_bytes(path)
                if identical_ok and existing == payload:
                    return False
                raise EpisodeConflictError(
                    "create-only episode authority record already exists"
                ) from None
            published = True
            self._fsync_directory(path.parent)
            return True
        except EpisodeStoreError:
            raise
        except OSError:
            raise EpisodeStoreError("episode authority record could not be published") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                if not published:
                    raise EpisodeStoreError(
                        "episode authority temporary record could not be cleaned"
                    ) from None

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise EpisodeIntegrityError("episode authority record is not a regular file")
            if metadata.st_size > _MAX_RECORD_BYTES:
                raise EpisodeIntegrityError("episode authority record exceeds size limit")
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            try:
                chunks: list[bytes] = []
                remaining = _MAX_RECORD_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(65_536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
            finally:
                os.close(descriptor)
            payload = b"".join(chunks)
        except EpisodeIntegrityError:
            raise
        except OSError:
            raise EpisodeIntegrityError("episode authority record could not be read") from None
        if len(payload) > _MAX_RECORD_BYTES:
            raise EpisodeIntegrityError("episode authority record exceeds size limit")
        return payload

    @classmethod
    def _read_model(cls, path: Path, model_type: type[BaseModel]) -> Any:
        payload = cls._read_bytes(path)
        try:
            model = model_type.model_validate(json.loads(payload))
        except (ValidationError, ValueError, json.JSONDecodeError):
            raise EpisodeIntegrityError("episode authority record failed validation") from None
        if payload != _payload_bytes(model):
            raise EpisodeIntegrityError("episode authority record is not canonical JSON")
        return model

    @staticmethod
    def _episode_path(episodes_dir: Path, episode_id: str) -> Path:
        try:
            validate_sha256(episode_id, field="episode_id")
        except ValueError:
            raise EpisodeIntegrityError("episode_id must be canonical 64-hex") from None
        return episodes_dir / episode_id

    @staticmethod
    def _attempt_path(attempts_dir: Path, attempt_id: str) -> Path:
        try:
            validate_sha256(attempt_id, field="attempt_id")
        except ValueError:
            raise EpisodeIntegrityError("attempt_id must be canonical 64-hex") from None
        return attempts_dir / attempt_id

    def _ensure_episode_manifest(
        self,
        contract: EpisodeAttemptContract,
    ) -> EpisodeAuthorityManifest:
        episode_id = contract.episode_id
        episode_path = self._episode_path(self.episodes_dir, episode_id)
        created = not episode_path.exists()
        self._ensure_directory(episode_path)
        attempts_path = episode_path / "attempts"
        self._ensure_directory(attempts_path)
        if created:
            self._fault("after_episode_directory_created", episode_path)
        expected = EpisodeAuthorityManifest.create(
            run_id=self.run_id,
            contract=contract,
        )
        manifest_path = episode_path / "episode.json"
        if manifest_path.exists():
            loaded = self._read_model(manifest_path, EpisodeAuthorityManifest)
            if loaded != expected:
                raise EpisodeConflictError("attempt fingerprints differ from episode authority")
            return loaded
        self._atomic_create_model(
            manifest_path,
            expected,
            event="before_episode_manifest_publish",
            identical_ok=True,
        )
        loaded = self._read_model(manifest_path, EpisodeAuthorityManifest)
        if loaded != expected:
            raise EpisodeConflictError("attempt fingerprints differ from episode authority")
        return loaded

    def _load_episode_unlocked(
        self,
        episode_id: str,
        *,
        allow_pending: bool,
    ) -> tuple[
        EpisodeAuthorityManifest | None,
        tuple[EpisodeAttemptAuthorityRecord, ...],
        EpisodeAttemptOutcome | None,
    ]:
        episode_path = self._episode_path(self.episodes_dir, episode_id)
        if not episode_path.exists():
            return None, (), None
        self._require_directory(episode_path)
        try:
            names = {child.name for child in episode_path.iterdir()}
        except OSError:
            raise EpisodeIntegrityError("episode authority directory could not be read") from None
        required = {"episode.json", "attempts"}
        allowed = required | {"accepted.json"}
        if not required <= names or not names <= allowed:
            raise EpisodeIntegrityError("episode authority is incomplete or contains debris")
        self._require_directory(episode_path / "attempts")
        manifest = self._read_model(episode_path / "episode.json", EpisodeAuthorityManifest)
        if manifest.run_id != self.run_id or manifest.episode_id != episode_id:
            raise EpisodeIntegrityError("episode manifest run/path identity mismatch")

        records: list[EpisodeAttemptAuthorityRecord] = []
        attempts_path = episode_path / "attempts"
        try:
            children = sorted(attempts_path.iterdir(), key=lambda path: path.name)
        except OSError:
            raise EpisodeIntegrityError("episode attempts could not be enumerated") from None
        seen_indices: set[int] = set()
        for child in children:
            try:
                validate_sha256(child.name, field="attempt_id")
                self._require_directory(child)
            except Exception:
                raise EpisodeIntegrityError("episode attempts contain debris or invalid paths")
            try:
                attempt_names = {path.name for path in child.iterdir()}
            except OSError:
                raise EpisodeIntegrityError(
                    "attempt authority directory could not be read"
                ) from None
            if "reservation.json" not in attempt_names or not attempt_names <= {
                "reservation.json",
                "outcome.json",
            }:
                raise EpisodeIntegrityError("attempt authority is incomplete or contains debris")
            reservation = self._read_model(child / "reservation.json", EpisodeAttemptReservation)
            if reservation.run_id != self.run_id:
                raise EpisodeIntegrityError("attempt reservation belongs to another run")
            if reservation.attempt_id != child.name:
                raise EpisodeIntegrityError("attempt directory identity mismatch")
            if reservation.episode_id != episode_id:
                raise EpisodeIntegrityError("attempt reservation belongs to another episode")
            if EpisodeScope.from_contract(reservation.contract) != manifest.scope:
                raise EpisodeIntegrityError("attempt reservation fingerprints differ from manifest")
            index = reservation.contract.attempt_index
            if index in seen_indices:
                raise EpisodeIntegrityError("episode attempt history contains duplicates")
            seen_indices.add(index)
            outcome: EpisodeAttemptOutcome | None = None
            if "outcome.json" in attempt_names:
                outcome = self._read_model(child / "outcome.json", EpisodeAttemptOutcome)
                if outcome.contract != reservation.contract:
                    raise EpisodeIntegrityError(
                        "attempt outcome does not bind exact reservation contract"
                    )
            records.append(
                EpisodeAttemptAuthorityRecord(
                    reservation=reservation,
                    outcome=outcome,
                )
            )
        records.sort(key=lambda item: item.reservation.contract.attempt_index)
        indices = [item.reservation.contract.attempt_index for item in records]
        if indices != list(range(len(indices))) or len(indices) > _MAX_ATTEMPTS:
            raise EpisodeIntegrityError("episode attempt history is not contiguous")
        pending = [index for index, record in enumerate(records) if record.pending]
        if len(pending) > 1 or (pending and pending[0] != len(records) - 1):
            raise EpisodeIntegrityError("pending reservation is not the final attempt")
        behavioral = [
            index
            for index, record in enumerate(records)
            if record.outcome is not None and record.outcome.disposition != "infrastructure_error"
        ]
        if len(behavioral) > 1 or (behavioral and behavioral[0] != len(records) - 1):
            raise EpisodeIntegrityError("episode history continues after behavioral terminal")

        accepted: EpisodeAttemptOutcome | None = None
        if "accepted.json" in names:
            pointer = self._read_model(episode_path / "accepted.json", AcceptedEpisodePointer)
            if pointer.episode_id != episode_id:
                raise EpisodeIntegrityError("accepted pointer belongs to another episode")
            matches = [
                record.outcome
                for record in records
                if record.reservation.attempt_id == pointer.attempt_id
                and record.outcome is not None
            ]
            if len(matches) != 1:
                raise EpisodeIntegrityError("accepted pointer references unknown attempt")
            accepted = matches[0]
            if accepted.disposition == "infrastructure_error":
                raise EpisodeIntegrityError("accepted pointer references infrastructure attempt")
            outcome_path = self._attempt_path(attempts_path, pointer.attempt_id) / "outcome.json"
            file_hash = _sha256_bytes(self._read_bytes(outcome_path))
            if (
                pointer.attempt_index != accepted.contract.attempt_index
                or pointer.outcome_id != accepted.outcome_id
                or pointer.outcome_file_hash != file_hash
            ):
                raise EpisodeIntegrityError("accepted pointer does not bind exact outcome")
        if pending and not allow_pending:
            raise EpisodeIndeterminateError(
                "reserved episode attempt lacks outcome; manual resolution required"
            )
        return manifest, tuple(records), accepted

    def load_accepted(self, episode_id: str) -> EpisodeAttemptOutcome | None:
        with self._locked(exclusive=False):
            _, _, accepted = self._load_episode_unlocked(episode_id, allow_pending=False)
            return accepted

    def load_attempts(self, episode_id: str) -> Sequence[EpisodeAttemptOutcome]:
        with self._locked(exclusive=False):
            _, records, _ = self._load_episode_unlocked(episode_id, allow_pending=False)
            return tuple(record.outcome for record in records if record.outcome is not None)

    def reserve_attempt(self, contract: EpisodeAttemptContract) -> None:
        """Persist the exact contract before a paid tau2 execution can begin."""

        if not isinstance(contract, EpisodeAttemptContract):
            raise EpisodeIntegrityError("reserve_attempt requires EpisodeAttemptContract")
        try:
            contract = EpisodeAttemptContract.model_validate(contract.model_dump(mode="json"))
            EpisodeScope.from_contract(contract)
        except (ValidationError, ValueError):
            raise EpisodeIntegrityError("attempt contract identity is invalid") from None
        with self._locked(exclusive=True):
            manifest = self._ensure_episode_manifest(contract)
            _, records, accepted = self._load_episode_unlocked(
                contract.episode_id, allow_pending=True
            )
            if EpisodeScope.from_contract(contract) != manifest.scope:
                raise EpisodeConflictError("attempt fingerprints differ from episode authority")
            existing = next(
                (
                    record
                    for record in records
                    if record.reservation.attempt_id == contract.attempt_id
                ),
                None,
            )
            if existing is not None:
                if existing.reservation.contract != contract:
                    raise EpisodeConflictError("same attempt_id has different reservation contract")
                if existing.pending:
                    raise EpisodeIndeterminateError(
                        "attempt is already reserved without outcome; dispatch blocked"
                    )
                raise EpisodeConflictError("attempt is already terminal and cannot be reserved")
            if any(record.pending for record in records):
                raise EpisodeIndeterminateError(
                    "prior attempt reservation lacks outcome; dispatch blocked"
                )
            if accepted is not None or any(
                record.outcome is not None and record.outcome.disposition != "infrastructure_error"
                for record in records
            ):
                raise EpisodeTerminalError("episode is terminal; reservation forbidden")
            expected_index = len(records)
            if contract.attempt_index != expected_index:
                raise EpisodeConflictError("attempt reservations must be contiguous and ordered")
            if expected_index >= _MAX_ATTEMPTS:
                raise EpisodeTerminalError("episode attempt envelope is exhausted")

            attempts_path = self._episode_path(self.episodes_dir, contract.episode_id) / "attempts"
            attempt_path = self._attempt_path(attempts_path, contract.attempt_id)
            self._ensure_directory(attempt_path)
            self._fault("after_attempt_directory_created", attempt_path)
            reservation = EpisodeAttemptReservation.create(
                run_id=self.run_id,
                contract=contract,
                reserved_at_utc=_utc_now(),
                pid_diagnostic=os.getpid(),
            )
            self._atomic_create_model(
                attempt_path / "reservation.json",
                reservation,
                event="before_attempt_reservation_publish",
                identical_ok=False,
            )
            _, verified, _ = self._load_episode_unlocked(contract.episode_id, allow_pending=True)
            if not any(record.reservation == reservation and record.pending for record in verified):
                raise EpisodeIntegrityError(
                    "published attempt reservation failed authority verification"
                )

    def record_attempt(self, outcome: EpisodeAttemptOutcome) -> None:
        if not isinstance(outcome, EpisodeAttemptOutcome):
            raise EpisodeIntegrityError("record_attempt requires EpisodeAttemptOutcome")
        try:
            outcome = EpisodeAttemptOutcome.model_validate(outcome.model_dump(mode="json"))
            EpisodeScope.from_contract(outcome.contract)
        except (ValidationError, ValueError):
            raise EpisodeIntegrityError("attempt outcome/contract identity is invalid") from None
        with self._locked(exclusive=True):
            manifest, records, _ = self._load_episode_unlocked(
                outcome.contract.episode_id, allow_pending=True
            )
            if manifest is None:
                raise EpisodeConflictError("attempt outcome has no pre-execution reservation")
            if EpisodeScope.from_contract(outcome.contract) != manifest.scope:
                raise EpisodeConflictError("attempt fingerprints differ from episode authority")
            existing = next(
                (
                    record
                    for record in records
                    if record.reservation.attempt_id == outcome.contract.attempt_id
                ),
                None,
            )
            if existing is None:
                raise EpisodeConflictError("attempt outcome has no exact pre-execution reservation")
            if existing.reservation.contract != outcome.contract:
                raise EpisodeConflictError("attempt outcome contract differs from reservation")
            if existing.outcome is not None:
                if existing.outcome == outcome:
                    return
                raise EpisodeConflictError("same attempt_id has different outcome content")
            attempt_path = (
                self._episode_path(self.episodes_dir, outcome.contract.episode_id)
                / "attempts"
                / outcome.contract.attempt_id
            )
            self._atomic_create_model(
                attempt_path / "outcome.json",
                outcome,
                event="before_outcome_publish",
                identical_ok=False,
            )
            _, verified, _ = self._load_episode_unlocked(
                outcome.contract.episode_id, allow_pending=True
            )
            if not any(record.outcome == outcome for record in verified):
                raise EpisodeIntegrityError("published attempt failed authority verification")

    def mark_episode_attempt_indeterminate(
        self,
        episode_id: str,
        attempt_id: str,
        *,
        owner_token: str,
        reason_code: str,
    ) -> EpisodeAttemptOutcome:
        """Explicitly settle a pending reservation as infrastructure-indeterminate.

        The caller must own the persistent run lease and, when a claim exists,
        it must have the same owner.  The reservation is retained permanently;
        this method only adds the immutable outcome required to authorize the
        next attempt index.
        """

        validate_sha256(episode_id, field="episode_id")
        validate_sha256(attempt_id, field="attempt_id")
        validate_sha256(owner_token, field="owner_token")
        _safe_reason(reason_code)
        safe_error_type = f"indeterminate_{reason_code}"
        with self._locked(exclusive=True):
            lease = self._load_run_lease_unlocked()
            if lease is None or lease.owner_token != owner_token:
                raise EpisodeLeaseHeldError(
                    "indeterminate resolution requires active run lease owner"
                )
            claims = self._load_claims_unlocked()
            episode_claims = [claim for claim in claims if claim.episode_id == episode_id]
            if episode_claims and (
                len(episode_claims) != 1 or episode_claims[0].owner_token != owner_token
            ):
                raise EpisodeLeaseHeldError(
                    "indeterminate resolution conflicts with episode claim owner"
                )
            manifest, records, accepted = self._load_episode_unlocked(
                episode_id, allow_pending=True
            )
            if manifest is None or accepted is not None:
                raise EpisodeConflictError(
                    "indeterminate resolution target is absent or already accepted"
                )
            matches = [record for record in records if record.reservation.attempt_id == attempt_id]
            if len(matches) != 1:
                raise EpisodeConflictError("indeterminate resolution requires exact reservation")
            record = matches[0]
            if record.outcome is not None:
                if (
                    record.outcome.disposition == "infrastructure_error"
                    and record.outcome.infrastructure_error_type == safe_error_type
                ):
                    return record.outcome
                raise EpisodeConflictError("attempt reservation already has an outcome")
            outcome = EpisodeAttemptOutcome.infrastructure(
                contract=record.reservation.contract,
                error_type=safe_error_type,
            )
            attempt_path = self._attempt_path(
                self._episode_path(self.episodes_dir, episode_id) / "attempts",
                attempt_id,
            )
            self._atomic_create_model(
                attempt_path / "outcome.json",
                outcome,
                event="before_indeterminate_outcome_publish",
                identical_ok=False,
            )
            _, verified, _ = self._load_episode_unlocked(episode_id, allow_pending=False)
            if not any(item.outcome == outcome for item in verified):
                raise EpisodeIntegrityError("indeterminate outcome failed authority verification")
            return outcome

    def accept_attempt(self, episode_id: str, attempt_id: str) -> EpisodeAttemptOutcome:
        self._episode_path(self.episodes_dir, episode_id)
        self._attempt_path(Path("."), attempt_id)
        with self._locked(exclusive=True):
            manifest, records, accepted = self._load_episode_unlocked(
                episode_id, allow_pending=False
            )
            if manifest is None:
                raise EpisodeConflictError("cannot accept an unknown episode/attempt")
            if accepted is not None:
                if accepted.contract.attempt_id == attempt_id:
                    return accepted
                raise EpisodeConflictError("episode already accepted another attempt")
            matches = [
                record.outcome
                for record in records
                if record.reservation.attempt_id == attempt_id and record.outcome is not None
            ]
            if len(matches) != 1:
                raise EpisodeConflictError("cannot accept an unknown attempt")
            outcome = matches[0]
            if outcome.disposition == "infrastructure_error":
                raise EpisodeConflictError("infrastructure attempts cannot be accepted")
            outcome_path = (
                self._episode_path(self.episodes_dir, episode_id)
                / "attempts"
                / attempt_id
                / "outcome.json"
            )
            pointer = AcceptedEpisodePointer.create(
                outcome=outcome,
                outcome_file_hash=_sha256_bytes(self._read_bytes(outcome_path)),
            )
            self._atomic_create_model(
                self._episode_path(self.episodes_dir, episode_id) / "accepted.json",
                pointer,
                event="before_accept_publish",
                identical_ok=True,
            )
            _, _, verified = self._load_episode_unlocked(episode_id, allow_pending=False)
            if verified != outcome:
                raise EpisodeIntegrityError("accepted outcome failed authority verification")
            return verified

    def load_run_lease(self) -> RunExecutionLease | None:
        with self._locked(exclusive=False):
            return self._load_run_lease_unlocked()

    def _load_run_lease_unlocked(self) -> RunExecutionLease | None:
        if not self.lease_path.exists():
            return None
        lease = self._read_model(self.lease_path, RunExecutionLease)
        if lease.run_id != self.run_id:
            raise EpisodeIntegrityError("run lease belongs to another run")
        return lease

    def acquire_run_lease(
        self,
        owner_token: str,
        *,
        started_at_utc: str | None = None,
        pid_diagnostic: int | None = None,
    ) -> RunExecutionLease:
        validate_sha256(owner_token, field="owner_token")
        with self._locked(exclusive=True):
            existing = self._load_run_lease_unlocked()
            if existing is not None:
                if existing.owner_token == owner_token:
                    return existing
                raise EpisodeLeaseHeldError(
                    "run lease is held; PID/time are diagnostic and cannot authorize stealing"
                )
            lease = RunExecutionLease.create(
                run_id=self.run_id,
                owner_token=owner_token,
                started_at_utc=started_at_utc or _utc_now(),
                pid_diagnostic=pid_diagnostic or os.getpid(),
            )
            self._atomic_create_model(
                self.lease_path,
                lease,
                event="before_run_lease_publish",
                identical_ok=False,
            )
            return self._load_run_lease_unlocked() or lease

    def load_episode_claims(self) -> tuple[EpisodeExecutionClaim, ...]:
        with self._locked(exclusive=False):
            return self._load_claims_unlocked()

    def _load_claims_unlocked(self) -> tuple[EpisodeExecutionClaim, ...]:
        claims: list[EpisodeExecutionClaim] = []
        try:
            children = sorted(self.claims_dir.iterdir(), key=lambda path: path.name)
        except OSError:
            raise EpisodeIntegrityError("episode claims could not be enumerated") from None
        for child in children:
            match = _HEX_FILE.fullmatch(child.name)
            if match is None:
                raise EpisodeIntegrityError("episode claim directory contains debris")
            claim = self._read_model(child, EpisodeExecutionClaim)
            if claim.claim_id != match.group(1) or claim.run_id != self.run_id:
                raise EpisodeIntegrityError("episode claim path/run identity mismatch")
            claims.append(claim)
        episode_ids = [claim.episode_id for claim in claims]
        if len(episode_ids) != len(set(episode_ids)):
            raise EpisodeIntegrityError("duplicate episode claims exist")
        if claims:
            lease = self._load_run_lease_unlocked()
            if lease is None or any(claim.owner_token != lease.owner_token for claim in claims):
                raise EpisodeIntegrityError("episode claims are not bound to active lease owner")
        return tuple(claims)

    def load_manual_recoveries(self) -> tuple[ManualRecoveryRecord, ...]:
        """Load every immutable explicit recovery record in canonical ID order."""

        with self._locked(exclusive=False):
            records: list[ManualRecoveryRecord] = []
            try:
                children = sorted(self.recoveries_dir.iterdir(), key=lambda path: path.name)
            except OSError:
                raise EpisodeIntegrityError("manual recoveries could not be enumerated") from None
            for child in children:
                match = _HEX_FILE.fullmatch(child.name)
                if match is None:
                    raise EpisodeIntegrityError("manual recovery directory contains debris")
                record = self._read_model(child, ManualRecoveryRecord)
                if record.recovery_id != match.group(1) or record.run_id != self.run_id:
                    raise EpisodeIntegrityError("manual recovery path/run identity mismatch")
                records.append(record)
            return tuple(records)

    def release_run_lease(self, owner_token: str) -> None:
        validate_sha256(owner_token, field="owner_token")
        with self._locked(exclusive=True):
            lease = self._load_run_lease_unlocked()
            if lease is None or lease.owner_token != owner_token:
                raise EpisodeLeaseHeldError("run lease is absent or owned by another token")
            if self._load_claims_unlocked():
                raise EpisodeLeaseHeldError("release episode claims before releasing run lease")
            try:
                self.lease_path.unlink()
                self._fsync_directory(self.control_dir)
            except OSError:
                raise EpisodeStoreError("run lease could not be released") from None

    def acquire_episode_claim(
        self,
        episode_id: str,
        owner_token: str,
        *,
        started_at_utc: str | None = None,
        pid_diagnostic: int | None = None,
    ) -> EpisodeExecutionClaim:
        validate_sha256(episode_id, field="episode_id")
        validate_sha256(owner_token, field="owner_token")
        with self._locked(exclusive=True):
            lease = self._load_run_lease_unlocked()
            if lease is None or lease.owner_token != owner_token:
                raise EpisodeLeaseHeldError("episode claim requires the active run lease owner")
            claim_id = _claim_id(run_id=self.run_id, episode_id=episode_id)
            path = self.claims_dir / f"{claim_id}.json"
            if path.exists():
                existing = self._read_model(path, EpisodeExecutionClaim)
                if existing.owner_token == owner_token and existing.episode_id == episode_id:
                    return existing
                raise EpisodeLeaseHeldError("episode claim belongs to another owner")
            claim = EpisodeExecutionClaim.create(
                run_id=self.run_id,
                episode_id=episode_id,
                owner_token=owner_token,
                started_at_utc=started_at_utc or _utc_now(),
                pid_diagnostic=pid_diagnostic or os.getpid(),
            )
            self._atomic_create_model(
                path,
                claim,
                event="before_episode_claim_publish",
                identical_ok=False,
            )
            return self._read_model(path, EpisodeExecutionClaim)

    def release_episode_claim(self, episode_id: str, owner_token: str) -> None:
        validate_sha256(episode_id, field="episode_id")
        validate_sha256(owner_token, field="owner_token")
        with self._locked(exclusive=True):
            claim_id = _claim_id(run_id=self.run_id, episode_id=episode_id)
            path = self.claims_dir / f"{claim_id}.json"
            if not path.exists():
                raise EpisodeLeaseHeldError("episode claim is absent")
            claim = self._read_model(path, EpisodeExecutionClaim)
            if claim.owner_token != owner_token:
                raise EpisodeLeaseHeldError("episode claim belongs to another owner")
            try:
                path.unlink()
                self._fsync_directory(self.claims_dir)
            except OSError:
                raise EpisodeStoreError("episode claim could not be released") from None

    def _publish_recovery(self, recovery: ManualRecoveryRecord) -> None:
        path = self.recoveries_dir / f"{recovery.recovery_id}.json"
        self._atomic_create_model(
            path,
            recovery,
            event="before_manual_recovery_publish",
            identical_ok=True,
        )

    def manual_recover_episode_claim(
        self,
        episode_id: str,
        *,
        expected_owner_token: str,
        operator_token: str,
        reason_code: str,
        recovered_at_utc: str | None = None,
    ) -> ManualRecoveryRecord:
        validate_sha256(episode_id, field="episode_id")
        validate_sha256(expected_owner_token, field="expected_owner_token")
        validate_sha256(operator_token, field="operator_token")
        _safe_reason(reason_code)
        with self._locked(exclusive=True):
            claim_id = _claim_id(run_id=self.run_id, episode_id=episode_id)
            path = self.claims_dir / f"{claim_id}.json"
            if not path.exists():
                raise EpisodeRecoveryError("manual recovery target claim is absent")
            claim = self._read_model(path, EpisodeExecutionClaim)
            if claim.owner_token != expected_owner_token:
                raise EpisodeRecoveryError("manual recovery owner expectation does not match")
            recovery = ManualRecoveryRecord.create(
                run_id=self.run_id,
                target_kind="episode_claim",
                target_id=claim.claim_id,
                target_record_hash=claim.claim_hash,
                expected_owner_token=expected_owner_token,
                operator_token=operator_token,
                reason_code=reason_code,
                recovered_at_utc=recovered_at_utc or _utc_now(),
            )
            self._publish_recovery(recovery)
            try:
                path.unlink()
                self._fsync_directory(self.claims_dir)
            except OSError:
                raise EpisodeRecoveryError("manual claim recovery could not finalize") from None
            return recovery

    def manual_recover_run_lease(
        self,
        *,
        expected_owner_token: str,
        operator_token: str,
        reason_code: str,
        recovered_at_utc: str | None = None,
    ) -> ManualRecoveryRecord:
        validate_sha256(expected_owner_token, field="expected_owner_token")
        validate_sha256(operator_token, field="operator_token")
        _safe_reason(reason_code)
        with self._locked(exclusive=True):
            lease = self._load_run_lease_unlocked()
            if lease is None:
                raise EpisodeRecoveryError("manual recovery target lease is absent")
            if lease.owner_token != expected_owner_token:
                raise EpisodeRecoveryError("manual recovery owner expectation does not match")
            if self._load_claims_unlocked():
                raise EpisodeRecoveryError("recover episode claims before run lease")
            recovery = ManualRecoveryRecord.create(
                run_id=self.run_id,
                target_kind="run_lease",
                target_id=lease.lease_hash,
                target_record_hash=lease.lease_hash,
                expected_owner_token=expected_owner_token,
                operator_token=operator_token,
                reason_code=reason_code,
                recovered_at_utc=recovered_at_utc or _utc_now(),
            )
            self._publish_recovery(recovery)
            try:
                self.lease_path.unlink()
                self._fsync_directory(self.control_dir)
            except OSError:
                raise EpisodeRecoveryError("manual lease recovery could not finalize") from None
            return recovery


# Concise alias for callers that prefer the storage-oriented name.
EpisodeStore = FileEpisodeArtifactAuthority


__all__ = [
    "AcceptedEpisodePointer",
    "EpisodeAuthorityManifest",
    "EpisodeAuthorityRun",
    "EpisodeAttemptAuthorityRecord",
    "EpisodeAttemptReservation",
    "EpisodeConflictError",
    "EpisodeExecutionClaim",
    "EpisodeIntegrityError",
    "EpisodeIndeterminateError",
    "EpisodeLeaseHeldError",
    "EpisodeRecoveryError",
    "EpisodeScope",
    "EpisodeStore",
    "EpisodeStoreError",
    "EpisodeTerminalError",
    "FileEpisodeArtifactAuthority",
    "ManualRecoveryRecord",
    "RunExecutionLease",
    "new_owner_token",
]
