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

"""Canonical hashing and artifact-safety helpers.

The public artifact layer accepts only JSON-compatible values.  It never
serializes provider SDK objects, credentials, or hidden reasoning fields.  The
same canonical encoder is used for prompt provenance, file hashes, and
manifest inputs so hashes remain reproducible in a clean process.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, JsonValue

from grace.artifacts.models import JsonObject, PromptProvenance
from grace.errors import ArtifactError


_FORBIDDEN_KEYS = frozenset(
    {
        "__thought__",
        "access_token",
        "api_key",
        "authorization",
        "chain_of_thought",
        "credential",
        "credential_path",
        "credentials",
        "gcp_project",
        "hidden_reasoning",
        "openai_api_key",
        "password",
        "principal",
        "private_key",
        "project_id",
        "reasoning_content",
        "refresh_token",
        "secret",
        "service_account",
        "vertex_project",
    }
)
_FORBIDDEN_KEY_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_credential",
    "_credential_path",
    "_password",
    "_private_key",
    "_refresh_token",
    "_secret",
)
_RAW_OUTPUT_KEYS = frozenset(
    {
        "raw_model_output",
        "raw_model_outputs",
        "raw_provider_payload",
        "raw_provider_payloads",
        "raw_response",
        "raw_responses",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}", re.IGNORECASE),
)


def to_json_value(value: Any, *, path: str = "$") -> JsonValue:
    """Convert a supported value to strict JSON data or raise ``ArtifactError``."""

    if isinstance(value, BaseModel):
        return to_json_value(value.model_dump(mode="json"), path=path)
    if isinstance(value, Enum):
        return to_json_value(value.value, path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ArtifactError(f"artifact datetime at {path} must include a timezone")
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError(f"artifact value at {path} is not finite")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactError(f"artifact mapping key at {path} is not a string")
            converted[key] = to_json_value(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ArtifactError(f"artifact value at {path} has unsupported type {type(value).__name__}")


def to_json_object(value: Mapping[str, Any] | BaseModel, *, path: str = "$") -> JsonObject:
    """Convert ``value`` to a strict JSON object."""

    converted = to_json_value(value, path=path)
    if not isinstance(converted, dict):
        raise ArtifactError(f"artifact value at {path} must be a mapping")
    return converted


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one strict, UTF-8 canonical JSON representation."""

    converted = to_json_value(value)
    try:
        return json.dumps(
            converted,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # defensive: conversion should catch these
        raise ArtifactError(f"artifact value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``value``."""

    return hashlib.sha256(value).hexdigest()


def hash_json(value: Any) -> str:
    """Hash canonical JSON data."""

    return sha256_bytes(canonical_json_bytes(value))


def hash_text(value: str) -> str:
    """Hash exact UTF-8 text without newline normalization."""

    return sha256_bytes(value.encode("utf-8"))


def capture_prompt_provenance(
    snapshot: Mapping[str, Any] | None,
) -> tuple[PromptProvenance, JsonObject | None]:
    """Derive hashes from the exact supplied prompt snapshot.

    Entry order is persisted separately because call/stage order may be
    operationally meaningful even though canonical JSON sorts mapping keys.
    """

    if snapshot is None:
        return PromptProvenance(captured=False), None

    normalized = to_json_object(snapshot, path="$.prompt_snapshot")
    if not all(key for key in normalized):
        raise ArtifactError("prompt snapshot entry names cannot be empty")
    if "__snapshot__" in normalized:
        raise ArtifactError("prompt snapshot entry name '__snapshot__' is reserved")
    entry_order = tuple(normalized)
    entry_hashes = {key: hash_json(normalized[key]) for key in entry_order}
    snapshot_hash = hash_json(
        {
            "namespace": "grace.prompt-snapshot.v1",
            "entry_order": entry_order,
            "entries": normalized,
        }
    )
    return (
        PromptProvenance(
            captured=True,
            snapshot_hash=snapshot_hash,
            entry_order=entry_order,
            entry_hashes=entry_hashes,
        ),
        normalized,
    )


def manifest_prompt_hashes(provenance: PromptProvenance) -> dict[str, str]:
    """Render prompt provenance into the compact manifest hash mapping."""

    if not provenance.captured:
        return {}
    assert provenance.snapshot_hash is not None
    return {
        "__snapshot__": provenance.snapshot_hash,
        **provenance.entry_hashes,
    }


def assert_artifact_safe(
    value: Any,
    *,
    sensitive_values: Sequence[str] = (),
    allow_raw_model_outputs: bool = False,
    path: str = "$",
) -> None:
    """Reject credential-shaped values and hidden reasoning recursively.

    The function deliberately reports only the location and category, never
    the rejected value.  Callers may supply runtime secrets/canaries through
    ``sensitive_values`` so even non-standard credential formats are caught.
    """

    normalized_sensitive = tuple(item for item in sensitive_values if item)

    def walk(item: Any, location: str) -> None:
        converted = to_json_value(item, path=location)
        if isinstance(converted, dict):
            for key, child in converted.items():
                normalized_key = key.strip().lower().replace("-", "_").replace(" ", "_")
                if normalized_key in _FORBIDDEN_KEYS or normalized_key.endswith(
                    _FORBIDDEN_KEY_SUFFIXES
                ):
                    raise ArtifactError(
                        f"artifact contains forbidden credential/reasoning field at "
                        f"{location}.{key}"
                    )
                if not allow_raw_model_outputs and normalized_key in _RAW_OUTPUT_KEYS:
                    raise ArtifactError(
                        f"raw model output at {location}.{key} requires explicit audit policy"
                    )
                walk(child, f"{location}.{key}")
            return
        if isinstance(converted, list):
            for index, child in enumerate(converted):
                walk(child, f"{location}[{index}]")
            return
        if isinstance(converted, str):
            if any(secret in converted for secret in normalized_sensitive):
                raise ArtifactError(f"artifact contains a configured sensitive value at {location}")
            if any(pattern.search(converted) for pattern in _SECRET_PATTERNS):
                raise ArtifactError(f"artifact contains credential-shaped text at {location}")

    walk(value, path)


__all__ = [
    "assert_artifact_safe",
    "canonical_json_bytes",
    "capture_prompt_provenance",
    "hash_json",
    "hash_text",
    "manifest_prompt_hashes",
    "sha256_bytes",
    "to_json_object",
    "to_json_value",
]
