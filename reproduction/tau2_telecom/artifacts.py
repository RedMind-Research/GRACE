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

"""Create-only workflow artifacts and safe resume for Tau2 telecom runs."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .experiment import Method, Tau2TelecomEvaluationPlan
from .harness.identity import canonical_json, canonical_sha256, text_sha256, validate_sha256
from .provenance import Tau2TelecomSourceProvenance


Frozen = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1)]
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise ValueError("run_id must contain only letters, digits, '.', '_', or '-'")
    return value


class Tau2TelecomRunContract(BaseModel):
    """Immutable inputs that define one resumable experiment run."""

    model_config = Frozen

    schema_version: Literal["grace.tau2-telecom-run.v1"] = "grace.tau2-telecom-run.v1"
    run_id: NonEmpty
    method: Method
    seed: int
    evaluation: Tau2TelecomEvaluationPlan
    experiment_config_hash: NonEmpty
    task_selection_hash: NonEmpty
    initial_instruction_hash: NonEmpty
    provenance: Tau2TelecomSourceProvenance
    run_fingerprint: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        method: Method,
        seed: int,
        evaluation: Tau2TelecomEvaluationPlan,
        experiment_config_hash: str,
        task_selection_hash: str,
        initial_instruction: str,
        provenance: Tau2TelecomSourceProvenance,
    ) -> Tau2TelecomRunContract:
        values: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "evaluation": evaluation.model_dump(mode="json"),
            "experiment_config_hash": experiment_config_hash,
            "task_selection_hash": task_selection_hash,
            "initial_instruction_hash": text_sha256(initial_instruction),
            "provenance": provenance.model_dump(mode="json"),
        }
        fingerprint = canonical_sha256(
            {"namespace": "grace.tau2-telecom-run-fingerprint.v1", **values}
        )
        return cls(**values, run_fingerprint=fingerprint)

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Tau2TelecomRunContract:
        _validate_run_id(self.run_id)
        for name in (
            "experiment_config_hash",
            "task_selection_hash",
            "initial_instruction_hash",
            "run_fingerprint",
        ):
            validate_sha256(getattr(self, name), field=name)
        payload = self.model_dump(mode="json", exclude={"schema_version", "run_fingerprint"})
        expected = canonical_sha256(
            {"namespace": "grace.tau2-telecom-run-fingerprint.v1", **payload}
        )
        if self.run_fingerprint != expected:
            raise ValueError("run_fingerprint does not match the immutable run inputs")
        return self


class Tau2TelecomCheckpoint(BaseModel):
    """Method-neutral instruction checkpoint; 0 is the P2G initial state."""

    model_config = Frozen

    schema_version: Literal["grace.tau2-telecom-checkpoint.v1"] = "grace.tau2-telecom-checkpoint.v1"
    method: Method
    checkpoint_index: int = Field(ge=0, le=10)
    state_id: NonEmpty
    parent_state_id: str | None = None
    instruction: NonEmpty
    instruction_hash: NonEmpty
    method_state: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Tau2TelecomCheckpoint:
        validate_sha256(self.state_id, field="state_id")
        validate_sha256(self.instruction_hash, field="instruction_hash")
        if self.parent_state_id is not None:
            validate_sha256(self.parent_state_id, field="parent_state_id")
        if self.instruction_hash != text_sha256(self.instruction):
            raise ValueError("instruction_hash does not match instruction")
        if (self.checkpoint_index == 0) != (self.parent_state_id is None):
            raise ValueError("only checkpoint 0 may omit parent_state_id")
        expected = canonical_sha256(
            {
                "checkpoint_index": self.checkpoint_index,
                "instruction_hash": self.instruction_hash,
                "method": self.method,
                "method_state": self.method_state,
                "namespace": "grace.tau2-telecom-checkpoint-state.v1",
                "parent_state_id": self.parent_state_id,
            }
        )
        if self.state_id != expected:
            raise ValueError("state_id does not match checkpoint content")
        return self

    @classmethod
    def create(
        cls,
        *,
        method: Method,
        checkpoint_index: int,
        instruction: str,
        parent_state_id: str | None,
        method_state: dict[str, JsonValue] | None = None,
    ) -> Tau2TelecomCheckpoint:
        instruction_hash = text_sha256(instruction)
        state = method_state or {}
        state_id = canonical_sha256(
            {
                "checkpoint_index": checkpoint_index,
                "instruction_hash": instruction_hash,
                "method": method,
                "method_state": state,
                "namespace": "grace.tau2-telecom-checkpoint-state.v1",
                "parent_state_id": parent_state_id,
            }
        )
        return cls(
            method=method,
            checkpoint_index=checkpoint_index,
            state_id=state_id,
            parent_state_id=parent_state_id,
            instruction=instruction,
            instruction_hash=instruction_hash,
            method_state=state,
        )


class Tau2TelecomStageArtifact(BaseModel):
    """Identity-bound opaque output for experience, diagnosis, or evaluation."""

    model_config = Frozen

    schema_version: Literal["grace.tau2-telecom-stage.v1"] = "grace.tau2-telecom-stage.v1"
    stage: Literal["experience", "diagnosis", "evaluation"]
    checkpoint_index: int = Field(ge=0, le=10)
    update_index: int | None = Field(default=None, ge=1, le=10)
    checkpoint_state_id: NonEmpty
    payload: JsonValue
    artifact_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        stage: Literal["experience", "diagnosis", "evaluation"],
        checkpoint_index: int,
        checkpoint_state_id: str,
        payload: JsonValue,
        update_index: int | None = None,
    ) -> Tau2TelecomStageArtifact:
        values = {
            "stage": stage,
            "checkpoint_index": checkpoint_index,
            "update_index": update_index,
            "checkpoint_state_id": checkpoint_state_id,
            "payload": payload,
        }
        return cls(
            stage=stage,
            checkpoint_index=checkpoint_index,
            update_index=update_index,
            checkpoint_state_id=checkpoint_state_id,
            payload=payload,
            artifact_hash=canonical_sha256({"namespace": "grace.tau2-telecom-stage.v1", **values}),
        )

    @model_validator(mode="after")
    def validate_artifact(self) -> Tau2TelecomStageArtifact:
        validate_sha256(self.checkpoint_state_id, field="checkpoint_state_id")
        validate_sha256(self.artifact_hash, field="artifact_hash")
        if self.stage == "evaluation" and self.update_index is not None:
            raise ValueError("evaluation artifacts do not have an update_index")
        if self.stage != "evaluation" and self.update_index is None:
            raise ValueError("experience and diagnosis artifacts require update_index")
        expected = canonical_sha256(
            {
                "namespace": "grace.tau2-telecom-stage.v1",
                "stage": self.stage,
                "checkpoint_index": self.checkpoint_index,
                "update_index": self.update_index,
                "checkpoint_state_id": self.checkpoint_state_id,
                "payload": self.payload,
            }
        )
        if self.artifact_hash != expected:
            raise ValueError("artifact_hash does not match stage content")
        return self


class Tau2TelecomArtifactLayout:
    """Minimal create-only filesystem layout used by the public runner."""

    def __init__(self, root: str | Path, run_id: str) -> None:
        _validate_run_id(run_id)
        self.run_dir = Path(root).expanduser().resolve() / run_id

    @property
    def contract_path(self) -> Path:
        return self.run_dir / "run.json"

    def checkpoint_path(self, index: int) -> Path:
        return self.run_dir / "checkpoints" / str(index) / "checkpoint.json"

    def stage_path(self, stage: str, index: int) -> Path:
        if stage == "evaluation":
            return self.run_dir / "evaluations" / f"checkpoint-{index}" / "result.json"
        return self.run_dir / "updates" / str(index) / stage / "result.json"

    @staticmethod
    def _read(path: Path, model: type[BaseModel]) -> BaseModel | None:
        if not path.exists():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid or corrupt workflow artifact: {path}") from exc

    @staticmethod
    def _create(path: Path, value: BaseModel) -> None:
        data = (canonical_json(value.model_dump(mode="json")) + "\n").encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"resume artifact conflicts with current inputs: {path}")
            return
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"concurrent artifact conflicts with current inputs: {path}")
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def bind_run(self, contract: Tau2TelecomRunContract) -> None:
        self._create(self.contract_path, contract)

    def load_contract(self) -> Tau2TelecomRunContract | None:
        value = self._read(self.contract_path, Tau2TelecomRunContract)
        return None if value is None else Tau2TelecomRunContract.model_validate(value)

    def load_checkpoint(self, index: int) -> Tau2TelecomCheckpoint | None:
        value = self._read(self.checkpoint_path(index), Tau2TelecomCheckpoint)
        return None if value is None else Tau2TelecomCheckpoint.model_validate(value)

    def save_checkpoint(self, checkpoint: Tau2TelecomCheckpoint) -> None:
        self._create(self.checkpoint_path(checkpoint.checkpoint_index), checkpoint)

    def load_stage(self, stage: str, index: int) -> Tau2TelecomStageArtifact | None:
        value = self._read(self.stage_path(stage, index), Tau2TelecomStageArtifact)
        return None if value is None else Tau2TelecomStageArtifact.model_validate(value)

    def save_stage(self, artifact: Tau2TelecomStageArtifact) -> None:
        index = (
            artifact.checkpoint_index if artifact.stage == "evaluation" else artifact.update_index
        )
        assert index is not None
        self._create(self.stage_path(artifact.stage, index), artifact)


__all__ = [
    "Tau2TelecomArtifactLayout",
    "Tau2TelecomCheckpoint",
    "Tau2TelecomRunContract",
    "Tau2TelecomStageArtifact",
]
