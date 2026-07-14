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

"""Deterministic application of anchored instruction patches.

The reconstruction model proposes localized edits, but it never writes the
next instruction directly.  This module applies only unambiguous anchors to a
copy of the previous instruction and records every proposal and outcome.  As a
result, text outside an accepted edit is preserved byte-for-byte.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from grace.artifacts.models import ReconstructionReport, ReconstructionStatus


class ReplaceSpan(BaseModel):
    """Replace the one occurrence of ``anchor`` with ``new_text``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["ReplaceSpan"] = "ReplaceSpan"
    anchor: str
    new_text: str


class InsertAfter(BaseModel):
    """Insert ``new_text`` after the one occurrence of ``anchor``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["InsertAfter"] = "InsertAfter"
    anchor: str
    new_text: str


class DeleteSpan(BaseModel):
    """Delete the one occurrence of ``anchor``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["DeleteSpan"] = "DeleteSpan"
    anchor: str


PatchOperation: TypeAlias = Annotated[
    ReplaceSpan | InsertAfter | DeleteSpan,
    Field(discriminator="op"),
]
TelemetryRecord: TypeAlias = dict[str, JsonValue]

_PATCH_OPERATION_ADAPTER: TypeAdapter[PatchOperation] = TypeAdapter(PatchOperation)
_KNOWN_OPERATIONS = frozenset({"ReplaceSpan", "InsertAfter", "DeleteSpan"})


def _sha256(text: str) -> str:
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("instruction text must be valid UTF-8") from None
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any, *, _seen: set[int] | None = None) -> JsonValue:
    """Return a JSON-safe snapshot for telemetry, including malformed inputs."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)

    seen = set() if _seen is None else _seen
    marker = id(value)
    if marker in seen:
        return "<recursive>"

    if isinstance(value, BaseModel):
        seen.add(marker)
        try:
            return _json_safe(value.model_dump(mode="json"), _seen=seen)
        finally:
            seen.remove(marker)
    if isinstance(value, Mapping):
        seen.add(marker)
        try:
            return {str(key): _json_safe(item, _seen=seen) for key, item in value.items()}
        finally:
            seen.remove(marker)
    if isinstance(value, (list, tuple)):
        seen.add(marker)
        try:
            return [_json_safe(item, _seen=seen) for item in value]
        finally:
            seen.remove(marker)
    return repr(value)


def _proposal_record(index: int, value: object) -> TelemetryRecord:
    return {"index": index, "operation": _json_safe(value)}


def _validation_errors(error: ValidationError) -> list[JsonValue]:
    """Keep useful validation details without persisting arbitrary raw objects."""

    return [
        {
            "type": item["type"],
            "location": ".".join(str(part) for part in item["loc"]),
            "message": item["msg"],
        }
        for item in error.errors(include_input=False, include_url=False)
    ]


def _parse_operation(
    value: object,
) -> tuple[PatchOperation | None, str | None, list[JsonValue]]:
    payload: object
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="python")
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        return None, "malformed_operation", []

    raw_op = payload.get("op") if isinstance(payload, dict) else None
    if isinstance(raw_op, str) and raw_op not in _KNOWN_OPERATIONS:
        return None, "unknown_op", []

    try:
        return _PATCH_OPERATION_ADAPTER.validate_python(payload), None, []
    except ValidationError as error:
        return None, "malformed_operation", _validation_errors(error)


class PatchApplicationResult(BaseModel):
    """Updated instruction plus a complete partition of patch outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction: str
    status: ReconstructionStatus
    input_instruction_hash: str
    output_instruction_hash: str
    proposed_operations: tuple[TelemetryRecord, ...] = ()
    applied_operations: tuple[TelemetryRecord, ...] = ()
    skipped_operations: tuple[TelemetryRecord, ...] = ()
    appended_operations: tuple[TelemetryRecord, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_telemetry_and_output_hash(self) -> PatchApplicationResult:
        outcome_count = (
            len(self.applied_operations)
            + len(self.skipped_operations)
            + len(self.appended_operations)
        )
        if outcome_count != len(self.proposed_operations):
            raise ValueError("each proposed patch operation must have exactly one outcome")
        if self.output_instruction_hash != _sha256(self.instruction):
            raise ValueError("output_instruction_hash does not match instruction")
        return self

    @property
    def output_instruction(self) -> str:
        """Descriptive alias for the reconstructed ``instruction`` field."""

        return self.instruction

    def to_reconstruction_report(self) -> ReconstructionReport:
        """Build the public artifact report without losing patch telemetry."""

        return ReconstructionReport(
            status=self.status,
            input_instruction_hash=self.input_instruction_hash,
            output_instruction_hash=self.output_instruction_hash,
            proposed_operations=self.proposed_operations,
            applied_operations=self.applied_operations,
            skipped_operations=self.skipped_operations,
            appended_operations=self.appended_operations,
            warnings=self.warnings,
        )


def apply_patch_operations(
    previous_instruction: str,
    operations: Iterable[object],
) -> PatchApplicationResult:
    """Apply typed or raw patch operations sequentially and deterministically.

    Replace, insert, and delete operations are accepted only when their anchor
    occurs exactly once in the instruction as it exists at that point in the
    sequence.  A non-empty ``InsertAfter`` with a missing or ambiguous anchor is
    appended without trimming or rewriting the existing instruction.  Other
    ambiguous, missing, unknown, or malformed operations are skipped.
    """

    text = previous_instruction
    proposed: list[TelemetryRecord] = []
    applied: list[TelemetryRecord] = []
    skipped: list[TelemetryRecord] = []
    appended: list[TelemetryRecord] = []

    for index, raw_operation in enumerate(operations):
        proposal = _proposal_record(index, raw_operation)
        proposed.append(proposal)

        operation, parse_reason, errors = _parse_operation(raw_operation)
        if operation is None:
            record: TelemetryRecord = {**proposal, "reason": parse_reason or "malformed_operation"}
            if errors:
                record["validation_errors"] = errors
            skipped.append(record)
            continue

        operation_payload = operation.model_dump(mode="json")
        anchor_count = text.count(operation.anchor) if operation.anchor else 0
        outcome_base: TelemetryRecord = {
            "index": index,
            "operation": operation_payload,
            "anchor_count": anchor_count,
        }

        if anchor_count == 1:
            if isinstance(operation, ReplaceSpan):
                text = text.replace(operation.anchor, operation.new_text, 1)
                action = "replaced"
            elif isinstance(operation, InsertAfter):
                replacement = operation.anchor + "\n" + operation.new_text
                text = text.replace(operation.anchor, replacement, 1)
                action = "inserted"
            else:
                text = text.replace(operation.anchor, "", 1)
                action = "deleted"
            applied.append({**outcome_base, "action": action})
            continue

        if isinstance(operation, InsertAfter) and operation.new_text.strip():
            separator = "" if not text or text.endswith("\n") else "\n"
            text += separator + operation.new_text
            appended.append(
                {
                    **outcome_base,
                    "action": "appended",
                    "reason": f"anchor_count={anchor_count}",
                    "separator_inserted": bool(separator),
                }
            )
            continue

        skipped.append(
            {
                **outcome_base,
                "reason": f"anchor_count={anchor_count}",
            }
        )

    warnings: list[str] = []
    if skipped:
        warnings.append(f"{len(skipped)} patch operation(s) skipped")
    if appended:
        warnings.append(f"{len(appended)} InsertAfter operation(s) appended after anchor fallback")

    if skipped or appended:
        status = ReconstructionStatus.COMPLETED_WITH_FALLBACKS
    elif applied:
        status = ReconstructionStatus.APPLIED
    else:
        status = ReconstructionStatus.NO_OP

    return PatchApplicationResult(
        instruction=text,
        status=status,
        input_instruction_hash=_sha256(previous_instruction),
        output_instruction_hash=_sha256(text),
        proposed_operations=tuple(proposed),
        applied_operations=tuple(applied),
        skipped_operations=tuple(skipped),
        appended_operations=tuple(appended),
        warnings=tuple(warnings),
    )


__all__ = [
    "DeleteSpan",
    "InsertAfter",
    "PatchApplicationResult",
    "PatchOperation",
    "ReplaceSpan",
    "apply_patch_operations",
]
