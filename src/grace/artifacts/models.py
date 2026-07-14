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

"""Serializable result, report, and artifact-manifest contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from grace.graph.models import GraceState
from grace.providers.base import ProviderAttemptRecord, UsageRecord
from grace.schemas.models import NetworkSchema


JsonObject = dict[str, JsonValue]


class ResultStatus(str, Enum):
    """Non-fatal public call outcomes.

    Fatal outcomes raise a public GRACE exception and therefore never masquerade
    as a successfully returned result.
    """

    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"


class ValidationStatus(str, Enum):
    """Distinct validation termination conditions."""

    PASSED = "passed"
    CONVERGED = "converged"
    STABLE_STOP = "stable_stop"
    CAP_REACHED = "cap_reached"
    FAILED = "failed"


class ReconstructionStatus(str, Enum):
    """Deterministic patch-application outcomes."""

    APPLIED = "applied"
    NO_OP = "no_op"
    COMPLETED_WITH_FALLBACKS = "completed_with_fallbacks"


class ValidationReport(BaseModel):
    """Typed summary of deterministic and model-assisted validation.

    ``schema_valid`` is the final deterministic invariant and must be true for
    a returned state.  ``fidelity_valid`` is optional because evolution-time
    structural validation does not run P2G fidelity analysis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ValidationStatus
    schema_valid: bool
    fidelity_valid: bool | None = None
    rounds: int = Field(default=0, ge=0)
    errors: tuple[JsonObject, ...] = ()
    warnings: tuple[str, ...] = ()
    forced_drops: tuple[JsonObject, ...] = ()
    rejected_operations: tuple[JsonObject, ...] = ()
    examined_node_ids: tuple[str, ...] = ()
    history: tuple[JsonObject, ...] = ()


class ReconstructionReport(BaseModel):
    """Patch telemetry proving how an instruction was reconstructed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ReconstructionStatus
    input_instruction_hash: str
    output_instruction_hash: str
    proposed_operations: tuple[JsonObject, ...] = ()
    applied_operations: tuple[JsonValue, ...] = ()
    skipped_operations: tuple[JsonObject, ...] = ()
    appended_operations: tuple[JsonValue, ...] = ()
    warnings: tuple[str, ...] = ()


class InitializationResult(BaseModel):
    """Public result of initial instruction-to-graph construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: GraceState
    validation_report: ValidationReport
    usage: tuple[UsageRecord, ...] = ()
    attempts: tuple[ProviderAttemptRecord, ...] = ()
    status: ResultStatus = ResultStatus.COMPLETED
    warnings: tuple[str, ...] = ()
    artifact_location: Path | None = None

    @model_validator(mode="after")
    def validate_completed_state(self) -> InitializationResult:
        self.state.validate_integrity()
        if not self.validation_report.schema_valid:
            raise ValueError("a returned initialization state must be schema-valid")
        requires_warning = bool(
            self.warnings
            or self.validation_report.warnings
            or self.validation_report.rejected_operations
            or self.validation_report.forced_drops
            or self.validation_report.status
            in {ValidationStatus.STABLE_STOP, ValidationStatus.CAP_REACHED}
        )
        if requires_warning and self.status != ResultStatus.COMPLETED_WITH_WARNINGS:
            raise ValueError("warning conditions require completed_with_warnings status")
        return self


class EvolutionResult(BaseModel):
    """Public result of one diagnosis-driven GRACE evolution step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: GraceState
    change_log: tuple[JsonObject, ...]
    validation_report: ValidationReport
    reconstruction_report: ReconstructionReport
    provenance: JsonObject = Field(default_factory=dict)
    usage: tuple[UsageRecord, ...] = ()
    attempts: tuple[ProviderAttemptRecord, ...] = ()
    status: ResultStatus = ResultStatus.COMPLETED
    warnings: tuple[str, ...] = ()
    artifact_location: Path | None = None

    @model_validator(mode="after")
    def validate_completed_state(self) -> EvolutionResult:
        self.state.validate_integrity()
        if not self.validation_report.schema_valid:
            raise ValueError("a returned evolution state must be schema-valid")
        if self.reconstruction_report.output_instruction_hash != self.state.instruction_hash:
            raise ValueError("reconstruction output hash does not match the returned instruction")
        requires_warning = bool(
            self.warnings
            or self.validation_report.warnings
            or self.validation_report.rejected_operations
            or self.validation_report.forced_drops
            or self.validation_report.status
            in {ValidationStatus.STABLE_STOP, ValidationStatus.CAP_REACHED}
            or self.reconstruction_report.warnings
            or self.reconstruction_report.skipped_operations
            or self.reconstruction_report.appended_operations
            or self.reconstruction_report.status == ReconstructionStatus.COMPLETED_WITH_FALLBACKS
        )
        if requires_warning and self.status != ResultStatus.COMPLETED_WITH_WARNINGS:
            raise ValueError("warning conditions require completed_with_warnings status")
        return self


class ArtifactManifest(BaseModel):
    """Common immutable manifest fields for checkpoint and audit stores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_format_version: str = Field(min_length=1)
    grace_version: str = Field(min_length=1)
    state_format_version: str = Field(min_length=1)
    mode: Literal["checkpoint", "audit"]
    result_kind: Literal["initialization", "evolution"] = "initialization"
    run_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    parent_state_id: str | None = None
    step: int = Field(ge=0)
    schema_id: str = Field(min_length=1)
    schema_hash: str = Field(min_length=1)
    schema_file: str = Field(default="schema.json", min_length=1)
    instruction_hash: str = Field(min_length=1)
    graph_hash: str = Field(min_length=1)
    config_hash: str | None = None
    input_hashes: dict[str, str] = Field(default_factory=dict)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    models: tuple[str, ...] = ()
    usage: tuple[UsageRecord, ...] = ()
    attempts: tuple[ProviderAttemptRecord, ...] = ()
    status: ResultStatus
    warnings: tuple[str, ...] = ()
    files: dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lineage_and_timestamp(self) -> ArtifactManifest:
        if self.step == 0 and self.parent_state_id is not None:
            raise ValueError("step 0 manifest must not have a parent_state_id")
        if self.step > 0 and not self.parent_state_id:
            raise ValueError("manifest after step 0 requires parent_state_id")
        if self.parent_state_id == self.state_id:
            raise ValueError("manifest state cannot be its own parent")
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone offset")
        return self

    def canonical_payload(self) -> JsonObject:
        """Return the canonical manifest payload covered by ``manifest_hash``.

        A manifest cannot include its own digest in the digest preimage.  The
        persisted hash therefore covers every other field, including all file
        hashes, lineage, prompt provenance, model usage, and format versions.
        """

        return self.model_dump(mode="json", exclude={"manifest_hash"})

    def computed_manifest_hash(self) -> str:
        """Compute the SHA-256 digest for the canonical manifest payload."""

        encoded = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_integrity_hash(self) -> ArtifactManifest:
        """Return a frozen copy carrying its recomputable integrity digest."""

        return self.model_copy(update={"manifest_hash": self.computed_manifest_hash()})

    def validate_integrity_hash(self) -> bool:
        """Validate the persisted digest and raise on a missing or stale hash."""

        if self.manifest_hash is None:
            raise ValueError("persisted artifact manifest is missing manifest_hash")
        if self.manifest_hash != self.computed_manifest_hash():
            raise ValueError("artifact manifest_hash does not match manifest content")
        return True


class PromptProvenance(BaseModel):
    """Hashes derived from the exact prompt snapshot sent by the caller.

    ``captured=False`` is an explicit assertion that no prompt snapshot was
    supplied (for example, a deterministic no-op).  It is different from a
    captured but empty mapping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    captured: bool
    snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    entry_order: tuple[str, ...] = ()
    entry_hashes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capture_state(self) -> PromptProvenance:
        if self.captured:
            if self.snapshot_hash is None:
                raise ValueError("captured prompt provenance requires snapshot_hash")
            if len(set(self.entry_order)) != len(self.entry_order):
                raise ValueError("prompt entry_order cannot contain duplicates")
            if set(self.entry_hashes) != set(self.entry_order):
                raise ValueError("prompt entry_order must match entry_hashes keys")
        elif self.snapshot_hash is not None or self.entry_order or self.entry_hashes:
            raise ValueError("uncaptured prompt provenance cannot contain hashes")
        return self


class ArtifactRecord(BaseModel):
    """A fully verified, self-contained artifact record returned by the store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    manifest: ArtifactManifest
    state: GraceState
    schema_snapshot: NetworkSchema
    result_payload: JsonObject
    prompt_provenance: PromptProvenance | None = None
    audit_payload: JsonObject | None = None


__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "EvolutionResult",
    "InitializationResult",
    "PromptProvenance",
    "ReconstructionReport",
    "ReconstructionStatus",
    "ResultStatus",
    "ValidationReport",
    "ValidationStatus",
]
