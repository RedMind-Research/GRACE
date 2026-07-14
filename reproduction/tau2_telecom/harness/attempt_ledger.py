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

"""Small create-only provider-attempt ledger for formal reproduction runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grace.providers.base import ProviderAttemptContext, ProviderAttemptRecord, UsageRecord

from .identity import canonical_json, canonical_sha256, validate_sha256


Frozen = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1)]


class ProviderAttemptReservation(BaseModel):
    model_config = Frozen
    schema_version: Literal["grace.provider-attempt-reservation.v1"] = (
        "grace.provider-attempt-reservation.v1"
    )
    provider_attempt_id: NonEmpty
    context: ProviderAttemptContext

    @staticmethod
    def _identity_payload(context: ProviderAttemptContext) -> dict[str, object]:
        """Exclude wall-clock progress while binding every dispatch input."""

        return {
            "attempt": context.attempt,
            "descriptor": context.descriptor.model_dump(mode="json"),
            "input_token_bound": context.input_token_bound,
            "input_token_bound_method": context.input_token_bound_method,
            "logical_call_id": context.logical_call_id,
            "max_tokens": context.max_tokens,
            "namespace": "grace.provider-attempt-reservation.v1",
            "overall_timeout_seconds": context.overall_timeout_seconds,
            "stage": context.stage,
            "temperature": context.temperature,
            "timeout_seconds": context.timeout_seconds,
        }

    @classmethod
    def create(cls, context: ProviderAttemptContext) -> ProviderAttemptReservation:
        attempt_id = canonical_sha256(cls._identity_payload(context))
        return cls(provider_attempt_id=attempt_id, context=context)

    @model_validator(mode="after")
    def validate_id(self) -> ProviderAttemptReservation:
        validate_sha256(self.provider_attempt_id, field="provider_attempt_id")
        expected = canonical_sha256(self._identity_payload(self.context))
        if self.provider_attempt_id != expected:
            raise ValueError("provider_attempt_id does not match reservation context")
        return self


class ProviderAttemptSettlement(BaseModel):
    model_config = Frozen
    schema_version: Literal["grace.provider-attempt-settlement.v1"] = (
        "grace.provider-attempt-settlement.v1"
    )
    provider_attempt_id: NonEmpty
    attempt: ProviderAttemptRecord
    usage: UsageRecord | None = None


class FileProviderAttemptLedger:
    """Fail-closed hook: reserve before dispatch and settle exactly once after."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.pending = self.root / "pending"
        self.settled = self.root / "settled"
        self.pending.mkdir(parents=True, exist_ok=True)
        self.settled.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_create_only(path: Path, model: BaseModel) -> None:
        data = (canonical_json(model.model_dump(mode="json")) + "\n").encode()
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise ValueError(f"provider attempt already exists: {path.stem}") from None
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def before_attempt(self, context: ProviderAttemptContext) -> object:
        reservation = ProviderAttemptReservation.create(context)
        attempt_id = reservation.provider_attempt_id
        if (self.settled / f"{attempt_id}.json").exists():
            raise ValueError("provider attempt was already settled")
        self._write_create_only(self.pending / f"{attempt_id}.json", reservation)
        return reservation

    def after_attempt(
        self,
        reservation: object,
        *,
        attempt: ProviderAttemptRecord,
        usage: UsageRecord | None,
    ) -> None:
        if not isinstance(reservation, ProviderAttemptReservation):
            raise ValueError("provider attempt reservation has an invalid type")
        attempt_id = reservation.provider_attempt_id
        pending = self.pending / f"{attempt_id}.json"
        if not pending.exists():
            raise ValueError("provider attempt reservation is not pending")
        settlement = ProviderAttemptSettlement(
            provider_attempt_id=attempt_id,
            attempt=attempt,
            usage=usage,
        )
        self._write_create_only(self.settled / f"{attempt_id}.json", settlement)
        pending.unlink()


__all__ = ["FileProviderAttemptLedger", "ProviderAttemptReservation"]
