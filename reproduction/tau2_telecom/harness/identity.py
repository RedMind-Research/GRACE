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

"""Canonical, collision-resistant identifiers for the telecom reproduction.

The reproduction never places tau2's long semantic task IDs in paths.  Stable
content identifiers are SHA-256 digests of a single canonical-JSON encoding.
The functions in this module are dependency-free and therefore usable by data
validators before tau2 or a model provider is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


SHA256_HEX_LENGTH = 64
TAU2_BENCHMARK = "tau2"
TELECOM_DOMAIN = "telecom"

EVAL20_TASK_ID = "[mms_issue]break_apn_mms_setting[PERSONA:Hard]"
EVAL20_TASK_UID = "8de4337b1af0459549fd46388cc2bd380049a3fc63c9f3e5aa33741719c1a58e"
PINNED_TAU2_REVISION = "c5b2d228d850c59b749b93cf32c4745d3aa53967"


class IdentityError(ValueError):
    """Raised when an identifier preimage cannot be encoded canonically."""


def _json_value(value: Any) -> Any:
    """Convert supported Python values into an unambiguous JSON value."""

    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityError("canonical JSON forbids NaN and infinite numbers")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityError("canonical JSON object keys must be strings")
            converted[key] = _json_value(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise IdentityError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the release's canonical JSON representation.

    ``ensure_ascii=False`` is deliberate: identifiers are hashes of UTF-8 text,
    not implementation-dependent escaped Unicode spellings.
    """

    try:
        return json.dumps(
            _json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, IdentityError):
            raise
        raise IdentityError(f"cannot encode canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Hash the UTF-8 bytes of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    """Hash exact UTF-8 text, preserving whitespace and newline choices."""

    if not isinstance(value, str):
        raise IdentityError("text hash input must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_sha256(value: str, *, field: str = "hash") -> str:
    """Validate and normalize a full lowercase SHA-256 digest."""

    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise IdentityError(f"{field} must be a full lowercase SHA-256 hex digest")
    return value


def make_task_uid(
    benchmark_task_id: str,
    *,
    benchmark_revision: str,
    domain: str = TELECOM_DOMAIN,
    benchmark: str = TAU2_BENCHMARK,
) -> str:
    """Return the stable task UID defined by the v1 public contract."""

    if not benchmark_task_id:
        raise IdentityError("benchmark_task_id must not be empty")
    if not benchmark_revision:
        raise IdentityError("benchmark_revision must not be empty")
    if not benchmark or not domain:
        raise IdentityError("benchmark and domain must not be empty")
    return canonical_sha256(
        {
            "benchmark": benchmark,
            "benchmark_revision": benchmark_revision,
            "domain": domain,
            "namespace": "grace.task.v1",
            "task_id": benchmark_task_id,
        }
    )


def make_split_slot(
    split: Literal["experience", "eval"],
    *,
    slot: int,
    batch: int | None = None,
) -> str:
    """Format a zero-based, human-readable locator that is never an identity."""

    if slot < 0:
        raise IdentityError("slot must be zero or greater")
    if split == "experience":
        if batch is None or batch < 0:
            raise IdentityError("experience split requires a non-negative batch")
        return f"exp-b{batch:02d}-s{slot:03d}"
    if split == "eval":
        if batch is not None:
            raise IdentityError("eval split must not specify a batch")
        return f"eval-s{slot:03d}"
    raise IdentityError(f"unsupported split: {split}")


def stable_seed_offset(benchmark_task_id: str) -> int:
    """Reproduce the project-wide deterministic tau2 seed offset."""

    digest = hashlib.sha256(benchmark_task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 10_000


def make_episode_id(
    *,
    task_uid: str,
    task_definition_hash: str,
    split_slot: str,
    trial: int,
    protocol_seed: int,
    checkpoint_state_id: str,
    benchmark_revision: str,
    protocol_hash: str,
    config_hash: str,
    model_roles_hash: str,
    evaluator_hash: str,
) -> str:
    """Bind one expected task/trial cell independently of physical retries."""

    validate_sha256(task_uid, field="task_uid")
    validate_sha256(task_definition_hash, field="task_definition_hash")
    if trial < 0:
        raise IdentityError("trial must be zero or greater")
    for name, value in {
        "split_slot": split_slot,
        "checkpoint_state_id": checkpoint_state_id,
        "benchmark_revision": benchmark_revision,
    }.items():
        if not value:
            raise IdentityError(f"{name} must not be empty")
    for name, value in {
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "model_roles_hash": model_roles_hash,
        "evaluator_hash": evaluator_hash,
    }.items():
        validate_sha256(value, field=name)
    return canonical_sha256(
        {
            "benchmark_revision": benchmark_revision,
            "checkpoint_state_id": checkpoint_state_id,
            "config_hash": config_hash,
            "evaluator_hash": evaluator_hash,
            "model_roles_hash": model_roles_hash,
            "namespace": "grace.episode.v1",
            "protocol_hash": protocol_hash,
            "protocol_seed": protocol_seed,
            "split_slot": split_slot,
            "task_definition_hash": task_definition_hash,
            "task_uid": task_uid,
            "trial": trial,
        }
    )


def make_reflection_id(
    *,
    episode_id: str,
    task_uid: str,
    trajectory_hash: str,
    policy_hash: str,
    diagnosis_prompt_hash: str,
    model: str,
    config_hash: str,
) -> str:
    """Return the cache key for one logical per-failure reflection."""

    for name, value in {
        "episode_id": episode_id,
        "task_uid": task_uid,
        "trajectory_hash": trajectory_hash,
        "policy_hash": policy_hash,
        "diagnosis_prompt_hash": diagnosis_prompt_hash,
        "config_hash": config_hash,
    }.items():
        validate_sha256(value, field=name)
    if not model:
        raise IdentityError("model must not be empty")
    return canonical_sha256(
        {
            "config_hash": config_hash,
            "diagnosis_prompt_hash": diagnosis_prompt_hash,
            "episode_id": episode_id,
            "model": model,
            "namespace": "grace.reflection.v1",
            "policy_hash": policy_hash,
            "task_uid": task_uid,
            "trajectory_hash": trajectory_hash,
        }
    )


def make_matrix_hash(ordered_episode_ids: Sequence[str]) -> str:
    """Hash the exact ordered expected episode set."""

    ids = list(ordered_episode_ids)
    if len(ids) != len(set(ids)):
        raise IdentityError("expected matrix contains duplicate episode IDs")
    for episode_id in ids:
        validate_sha256(episode_id, field="episode_id")
    return canonical_sha256({"episode_ids": ids, "namespace": "grace.expected-matrix.v1"})


def make_diagnosis_fingerprint(
    *,
    expected_matrix_hash: str,
    policy_hash: str,
    diagnosis_prompt_hash: str,
    model: str,
    config_hash: str,
    ordered_reflection_ids: Sequence[str],
    classification_hash: str,
) -> str:
    """Bind all inputs whose change invalidates diagnosis resume artifacts."""

    for name, value in {
        "expected_matrix_hash": expected_matrix_hash,
        "policy_hash": policy_hash,
        "diagnosis_prompt_hash": diagnosis_prompt_hash,
        "config_hash": config_hash,
        "classification_hash": classification_hash,
    }.items():
        validate_sha256(value, field=name)
    reflection_ids = list(ordered_reflection_ids)
    if len(reflection_ids) != len(set(reflection_ids)):
        raise IdentityError("diagnosis contains duplicate reflection IDs")
    for reflection_id in reflection_ids:
        validate_sha256(reflection_id, field="reflection_id")
    return canonical_sha256(
        {
            "classification_hash": classification_hash,
            "config_hash": config_hash,
            "diagnosis_prompt_hash": diagnosis_prompt_hash,
            "expected_matrix_hash": expected_matrix_hash,
            "model": model,
            "namespace": "grace.diagnosis.v1",
            "ordered_reflection_ids": reflection_ids,
            "policy_hash": policy_hash,
        }
    )


def verify_release_test_vectors() -> None:
    """Fail fast if canonicalization drifts from the reviewed release contract."""

    actual = make_task_uid(
        EVAL20_TASK_ID,
        benchmark_revision=PINNED_TAU2_REVISION,
    )
    if actual != EVAL20_TASK_UID:
        raise IdentityError(f"EVAL20 task UID drifted: expected {EVAL20_TASK_UID}, got {actual}")


__all__ = [
    "EVAL20_TASK_ID",
    "EVAL20_TASK_UID",
    "IdentityError",
    "PINNED_TAU2_REVISION",
    "TELECOM_DOMAIN",
    "canonical_json",
    "canonical_sha256",
    "make_diagnosis_fingerprint",
    "make_episode_id",
    "make_matrix_hash",
    "make_reflection_id",
    "make_split_slot",
    "make_task_uid",
    "stable_seed_offset",
    "text_sha256",
    "validate_sha256",
    "verify_release_test_vectors",
]
