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

"""Typed deterministic contracts for the tau2 telecom reproduction harness."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from grace.providers.base import UsageRecord

from .identity import (
    canonical_sha256,
    make_matrix_hash,
    make_split_slot,
    make_task_uid,
    validate_sha256,
)


FrozenModelConfig = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1)]
SplitKind = Literal["experience", "eval"]


class TaskSlotSpec(BaseModel):
    """One ordered task locator in a frozen public manifest."""

    model_config = FrozenModelConfig

    split: SplitKind
    slot: int = Field(ge=0)
    benchmark_task_id: NonEmpty
    batch: int | None = Field(default=None, ge=0)
    phase: Literal["A", "B"] | None = None

    @model_validator(mode="after")
    def validate_split_shape(self) -> TaskSlotSpec:
        if self.split == "experience" and self.batch is None:
            raise ValueError("experience task slots require a batch")
        if self.split == "eval" and self.batch is not None:
            raise ValueError("eval task slots must not contain a batch")
        if self.split == "eval" and self.phase is not None:
            raise ValueError("eval task slots must not contain an experience phase")
        return self

    @property
    def split_slot(self) -> str:
        return make_split_slot(self.split, slot=self.slot, batch=self.batch)


class TaskMapEntry(BaseModel):
    """Immutable mapping between a semantic benchmark ID and safe identifiers."""

    model_config = FrozenModelConfig

    benchmark: NonEmpty = "tau2"
    benchmark_revision: NonEmpty
    domain: NonEmpty = "telecom"
    benchmark_task_id: NonEmpty
    task_uid: NonEmpty
    split: SplitKind
    slot: int = Field(ge=0)
    batch: int | None = Field(default=None, ge=0)
    phase: Literal["A", "B"] | None = None
    split_slot: NonEmpty
    task_definition_hash: NonEmpty
    publication_status: Literal["public", "sanitized", "private"] = "private"

    @model_validator(mode="after")
    def validate_identifiers(self) -> TaskMapEntry:
        expected_slot = make_split_slot(self.split, slot=self.slot, batch=self.batch)
        if self.split_slot != expected_slot:
            raise ValueError(f"split_slot must be {expected_slot}")
        expected_uid = make_task_uid(
            self.benchmark_task_id,
            benchmark_revision=self.benchmark_revision,
            domain=self.domain,
            benchmark=self.benchmark,
        )
        if self.task_uid != expected_uid:
            raise ValueError("task_uid does not match the canonical task preimage")
        validate_sha256(self.task_definition_hash, field="task_definition_hash")
        return self


class TaskSelectionManifest(BaseModel):
    """Ordered public task selection independent of the tau2 Python package."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.task-selection.v1"] = "grace.task-selection.v1"
    benchmark: NonEmpty = "tau2"
    benchmark_revision: NonEmpty
    domain: NonEmpty = "telecom"
    seed: int
    tasks: tuple[TaskSlotSpec, ...]

    @model_validator(mode="after")
    def validate_unique_slots_and_tasks(self) -> TaskSelectionManifest:
        locators = [task.split_slot for task in self.tasks]
        task_ids = [task.benchmark_task_id for task in self.tasks]
        if len(locators) != len(set(locators)):
            raise ValueError("task selection contains duplicate split slots")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task selection contains duplicate benchmark task IDs")
        return self


class EpisodeCell(BaseModel):
    """One expected task/trial cell; retries never change this identity."""

    model_config = FrozenModelConfig

    episode_id: NonEmpty
    task_uid: NonEmpty
    task_definition_hash: NonEmpty
    benchmark_task_id: NonEmpty
    split_slot: NonEmpty
    trial: int = Field(ge=0)
    protocol_seed: int
    checkpoint_state_id: NonEmpty

    @model_validator(mode="after")
    def validate_hashes(self) -> EpisodeCell:
        validate_sha256(self.episode_id, field="episode_id")
        validate_sha256(self.task_uid, field="task_uid")
        validate_sha256(self.task_definition_hash, field="task_definition_hash")
        return self


class ExpectedMatrix(BaseModel):
    """The exact ordered cells a stage must accept before completion."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.expected-matrix.v1"] = "grace.expected-matrix.v1"
    episodes: tuple[EpisodeCell, ...]
    matrix_hash: NonEmpty

    @model_validator(mode="after")
    def validate_exact_ordered_set(self) -> ExpectedMatrix:
        ids = [episode.episode_id for episode in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("expected matrix contains duplicate episode IDs")
        if self.matrix_hash != make_matrix_hash(ids):
            raise ValueError("matrix_hash does not match ordered episode IDs")
        return self


class CompletenessReport(BaseModel):
    """Deterministic comparison of observed keys to an exact expected matrix."""

    model_config = FrozenModelConfig

    matrix_hash: NonEmpty
    expected_count: int = Field(ge=0)
    observed_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    missing_episode_ids: tuple[str, ...] = ()
    unknown_episode_ids: tuple[str, ...] = ()
    duplicate_episode_ids: tuple[str, ...] = ()
    out_of_order: bool = False
    complete: bool = False


class TrajectoryToolCall(BaseModel):
    """Sanitized tool-call representation; opaque provider call IDs are excluded."""

    model_config = FrozenModelConfig

    name: NonEmpty
    arguments: JsonValue = Field(default_factory=dict)


class TrajectoryMessage(BaseModel):
    """A public-safe trajectory message used as diagnosis input."""

    model_config = FrozenModelConfig

    role: NonEmpty
    content: str | None = None
    tool_calls: tuple[TrajectoryToolCall, ...] = ()


class TrajectoryRecord(BaseModel):
    """One sanitized, identity-bound episode trajectory."""

    model_config = FrozenModelConfig

    episode_id: NonEmpty
    task_uid: NonEmpty
    task_definition_hash: NonEmpty
    benchmark_task_id: NonEmpty
    split_slot: NonEmpty
    trial: int = Field(ge=0)
    protocol_seed: int
    success: bool
    termination_reason: NonEmpty
    messages: tuple[TrajectoryMessage, ...]

    @model_validator(mode="after")
    def validate_identifiers(self) -> TrajectoryRecord:
        validate_sha256(self.episode_id, field="episode_id")
        validate_sha256(self.task_uid, field="task_uid")
        validate_sha256(self.task_definition_hash, field="task_definition_hash")
        return self

    @property
    def substantive(self) -> bool:
        return any(
            (message.content is not None and bool(message.content.strip()))
            or bool(message.tool_calls)
            for message in self.messages
        )

    @property
    def trajectory_hash(self) -> str:
        return canonical_sha256(
            {
                "namespace": "grace.trajectory.v1",
                "trajectory": self.model_dump(mode="json"),
            }
        )


class EligibilityStatus(str, Enum):
    SUCCESSFUL_CONTROL = "successful_control"
    ELIGIBLE = "eligible"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    INVALID_INPUT = "invalid_input"


class EligibilityDecision(BaseModel):
    """Deterministic pre-call classification for one expected episode."""

    model_config = FrozenModelConfig

    episode_id: str | None = None
    split_slot: str | None = None
    status: EligibilityStatus
    reason_code: NonEmpty
    detail: NonEmpty


class DiagnosisFinding(BaseModel):
    model_config = FrozenModelConfig

    finding_id: NonEmpty
    affected_subtasks: tuple[str, ...] = ()
    description: NonEmpty
    proposed_instruction: NonEmpty


class FailureReflection(BaseModel):
    """Normalized accepted Phase-1 provider output."""

    model_config = FrozenModelConfig

    task_id: NonEmpty
    subtasks_attempted: tuple[str, ...] = ()
    subtasks_missed: tuple[str, ...] = ()
    findings: tuple[DiagnosisFinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_findings(self) -> FailureReflection:
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("reflection contains duplicate finding IDs")
        return self


class DiagnosisTheme(BaseModel):
    model_config = FrozenModelConfig

    theme_id: NonEmpty
    title: NonEmpty
    frequency: int = Field(ge=1)
    affected_subtask_types: tuple[str, ...] = ()
    root_cause: NonEmpty
    recommended_actions: tuple[NonEmpty, ...] = Field(min_length=1)


class SynthesisResponse(BaseModel):
    """Accepted Phase-2 provider output before deterministic evidence is attached."""

    model_config = FrozenModelConfig

    themes: tuple[DiagnosisTheme, ...] = Field(min_length=1)
    summary: NonEmpty

    @model_validator(mode="after")
    def validate_unique_themes(self) -> SynthesisResponse:
        theme_ids = [theme.theme_id for theme in self.themes]
        if len(theme_ids) != len(set(theme_ids)):
            raise ValueError("synthesis contains duplicate theme IDs")
        return self


class ToolErrorEvidenceItem(BaseModel):
    model_config = FrozenModelConfig

    tool: NonEmpty
    count: int = Field(ge=1)
    task_count: int = Field(ge=1)
    task_ids: tuple[str, ...]


class ObservedToolErrorEvidence(BaseModel):
    model_config = FrozenModelConfig

    total_tool_not_found_calls: int = Field(ge=0)
    unique_tool_not_found_names: int = Field(ge=0)
    tools: tuple[ToolErrorEvidenceItem, ...] = ()


class DiagnosisSynthesis(BaseModel):
    model_config = FrozenModelConfig

    themes: tuple[DiagnosisTheme, ...]
    summary: NonEmpty
    observed_tool_error_evidence: ObservedToolErrorEvidence


class ReflectionArtifact(BaseModel):
    """One collision-free reflection cache entry."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.reflection-artifact.v1"] = "grace.reflection-artifact.v1"
    reflection_id: NonEmpty
    episode_id: NonEmpty
    task_uid: NonEmpty
    benchmark_task_id: NonEmpty
    trajectory_hash: NonEmpty
    input_fingerprint: NonEmpty
    reflection: FailureReflection
    usage: UsageRecord

    @model_validator(mode="after")
    def validate_hash_fields(self) -> ReflectionArtifact:
        for name, value in {
            "reflection_id": self.reflection_id,
            "episode_id": self.episode_id,
            "task_uid": self.task_uid,
            "trajectory_hash": self.trajectory_hash,
            "input_fingerprint": self.input_fingerprint,
        }.items():
            validate_sha256(value, field=name)
        if self.reflection.task_id != self.benchmark_task_id:
            raise ValueError("reflection task_id does not match its benchmark task ID")
        return self


class SynthesisArtifact(BaseModel):
    """One accepted Phase-2 output bound to a diagnosis fingerprint."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.synthesis-artifact.v1"] = "grace.synthesis-artifact.v1"
    diagnosis_fingerprint: NonEmpty
    synthesis: DiagnosisSynthesis
    usage: UsageRecord

    @model_validator(mode="after")
    def validate_fingerprint(self) -> SynthesisArtifact:
        validate_sha256(self.diagnosis_fingerprint, field="diagnosis_fingerprint")
        return self


class DiagnosisStatus(str, Enum):
    RUNNING = "running"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    NO_UPDATE = "no_update"


class DiagnosisErrorRecord(BaseModel):
    """Safe persisted error classification without provider exception text."""

    model_config = FrozenModelConfig

    stage: Literal["input", "reflection", "synthesis"]
    logical_id: str | None = None
    error_type: NonEmpty
    safe_message: NonEmpty
    usage: UsageRecord | None = None


class DiagnosisManifest(BaseModel):
    """Resume authority for one diagnosis batch."""

    model_config = FrozenModelConfig

    schema_version: Literal["grace.diagnosis-manifest.v1"] = "grace.diagnosis-manifest.v1"
    status: DiagnosisStatus
    diagnosis_fingerprint: NonEmpty
    expected_matrix_hash: NonEmpty
    policy_hash: NonEmpty
    diagnosis_prompt_hash: NonEmpty
    config_hash: NonEmpty
    model: NonEmpty
    classifications: tuple[EligibilityDecision, ...]
    ordered_eligible_episode_ids: tuple[str, ...]
    ordered_reflection_ids: tuple[str, ...]
    completed_reflection_ids: tuple[str, ...] = ()
    failed_reflection_ids: tuple[str, ...] = ()
    synthesis_accepted: bool = False
    errors: tuple[DiagnosisErrorRecord, ...] = ()

    @model_validator(mode="after")
    def validate_manifest_sets(self) -> DiagnosisManifest:
        for name, value in {
            "diagnosis_fingerprint": self.diagnosis_fingerprint,
            "expected_matrix_hash": self.expected_matrix_hash,
            "policy_hash": self.policy_hash,
            "diagnosis_prompt_hash": self.diagnosis_prompt_hash,
            "config_hash": self.config_hash,
        }.items():
            validate_sha256(value, field=name)
        expected_reflections = set(self.ordered_reflection_ids)
        completed = set(self.completed_reflection_ids)
        failed = set(self.failed_reflection_ids)
        if len(expected_reflections) != len(self.ordered_reflection_ids):
            raise ValueError("ordered_reflection_ids contains duplicates")
        if completed & failed:
            raise ValueError("a reflection cannot be both completed and failed")
        if not completed <= expected_reflections or not failed <= expected_reflections:
            raise ValueError("reflection state contains an unknown reflection ID")
        if self.status == DiagnosisStatus.COMPLETE:
            if completed != expected_reflections or failed or not self.synthesis_accepted:
                raise ValueError("complete diagnosis requires every reflection and synthesis")
        if self.status == DiagnosisStatus.NO_UPDATE:
            if expected_reflections or self.synthesis_accepted:
                raise ValueError("no_update diagnosis cannot contain reflections or synthesis")
        return self


class DiagnosisResult(BaseModel):
    """Returned diagnosis outcome; only COMPLETE may enter Evolution."""

    model_config = FrozenModelConfig

    status: DiagnosisStatus
    manifest: DiagnosisManifest
    reflections: tuple[ReflectionArtifact, ...] = ()
    synthesis: DiagnosisSynthesis | None = None
    report: str | None = None
    usage: tuple[UsageRecord, ...] = ()
    artifact_dir: Path

    @model_validator(mode="after")
    def bind_result_to_manifest(self) -> DiagnosisResult:
        if self.status != self.manifest.status:
            raise ValueError("result status must match manifest status")
        if self.status == DiagnosisStatus.COMPLETE:
            if self.synthesis is None or not self.report:
                raise ValueError("complete diagnosis requires synthesis and report")
        if self.status == DiagnosisStatus.NO_UPDATE and self.usage:
            raise ValueError("no_update diagnosis must use zero provider calls")
        return self


__all__ = [
    "CompletenessReport",
    "DiagnosisErrorRecord",
    "DiagnosisFinding",
    "DiagnosisManifest",
    "DiagnosisResult",
    "DiagnosisStatus",
    "DiagnosisSynthesis",
    "DiagnosisTheme",
    "EligibilityDecision",
    "EligibilityStatus",
    "EpisodeCell",
    "ExpectedMatrix",
    "FailureReflection",
    "ObservedToolErrorEvidence",
    "ReflectionArtifact",
    "SynthesisArtifact",
    "SynthesisResponse",
    "TaskMapEntry",
    "TaskSelectionManifest",
    "TaskSlotSpec",
    "ToolErrorEvidenceItem",
    "TrajectoryMessage",
    "TrajectoryRecord",
    "TrajectoryToolCall",
]
