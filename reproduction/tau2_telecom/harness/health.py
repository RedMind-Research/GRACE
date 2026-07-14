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

"""Read-only health views derived from immutable episode authority records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .identity import canonical_sha256
from .tau2_interface import EpisodeMatrixExecution


class EpisodeHealthSummary(BaseModel):
    """Derived execution health; never consulted to decide resume work."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["grace.episode-health-summary.v1"] = "grace.episode-health-summary.v1"
    source: Literal["derived_from_immutable_authority"] = "derived_from_immutable_authority"
    matrix_hash: str
    status: Literal["complete", "incomplete"]
    expected_episodes: int = Field(ge=0)
    accepted_episodes: int = Field(ge=0)
    behavioral_successes: int = Field(ge=0)
    behavioral_failures: int = Field(ge=0)
    infrastructure_attempts: int = Field(ge=0)
    attempts_total: int = Field(ge=0)
    missing_episode_ids: tuple[str, ...] = ()
    infrastructure_error_counts: dict[str, int] = Field(default_factory=dict)
    authority_snapshot_hash: str

    @model_validator(mode="after")
    def validate_counts(self) -> EpisodeHealthSummary:
        if self.accepted_episodes != self.behavioral_successes + self.behavioral_failures:
            raise ValueError("accepted health count must equal behavioral outcomes")
        if self.attempts_total < self.accepted_episodes + self.infrastructure_attempts:
            raise ValueError("attempt health counts are inconsistent")
        return self


def derive_health_summary(execution: EpisodeMatrixExecution) -> EpisodeHealthSummary:
    """Rebuild health entirely from immutable accepted/attempt outcome content."""

    successes = sum(outcome.disposition == "behavioral_success" for outcome in execution.accepted)
    failures = sum(outcome.disposition == "behavioral_failure" for outcome in execution.accepted)
    infrastructure = [
        outcome for outcome in execution.attempts if outcome.disposition == "infrastructure_error"
    ]
    error_counts: dict[str, int] = {}
    for outcome in infrastructure:
        error_type = outcome.infrastructure_error_type or "unknown"
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
    snapshot_hash = canonical_sha256(
        {
            "accepted_outcome_ids": [item.outcome_id for item in execution.accepted],
            "attempt_outcome_ids": [item.outcome_id for item in execution.attempts],
            "matrix_hash": execution.matrix_hash,
            "namespace": "grace.episode-authority-snapshot.v1",
        }
    )
    return EpisodeHealthSummary(
        matrix_hash=execution.matrix_hash,
        status=execution.status,
        expected_episodes=execution.completeness.expected_count,
        accepted_episodes=len(execution.accepted),
        behavioral_successes=successes,
        behavioral_failures=failures,
        infrastructure_attempts=len(infrastructure),
        attempts_total=len(execution.attempts),
        missing_episode_ids=execution.completeness.missing_episode_ids,
        infrastructure_error_counts=error_counts,
        authority_snapshot_hash=snapshot_hash,
    )


__all__ = ["EpisodeHealthSummary", "derive_health_summary"]
