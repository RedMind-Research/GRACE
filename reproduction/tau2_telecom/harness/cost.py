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

"""Cost/usage summaries derived only from immutable episode attempt outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .identity import canonical_sha256
from .tau2_interface import EpisodeAttemptOutcome


NonEmpty = Annotated[str, Field(min_length=1)]


class RoleCostSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: NonEmpty
    model_turns: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    usage_complete: bool
    cost_complete: bool


class EpisodeCostSummary(BaseModel):
    """A rebuildable view, explicitly unsuitable as resume authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["grace.episode-cost-summary.v1"] = "grace.episode-cost-summary.v1"
    source: Literal["derived_from_immutable_attempts"] = "derived_from_immutable_attempts"
    attempt_count: int = Field(ge=0)
    infrastructure_attempt_count: int = Field(ge=0)
    agent: RoleCostSummary
    user_simulator: RoleCostSummary
    total_cost_usd: float | None = Field(default=None, ge=0.0)
    duration_seconds: float = Field(ge=0.0)
    derived_hash: str


def _optional_int_sum(values: Sequence[int | None]) -> int | None:
    if not values:
        return None
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _optional_float_sum(values: Sequence[float | None]) -> float | None:
    if not values:
        return None
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def derive_cost_summary(attempts: Sequence[EpisodeAttemptOutcome]) -> EpisodeCostSummary:
    """Aggregate all attempts, including infrastructure attempts, without zero filling."""

    records = tuple(attempts)
    if records:
        agent_models = {item.contract.agent_model for item in records}
        user_models = {item.contract.user_model for item in records}
        if len(agent_models) != 1 or len(user_models) != 1:
            raise ValueError("cost summary cannot mix model-role configurations")
        agent_model = next(iter(agent_models))
        user_model = next(iter(user_models))
    else:
        agent_model = "not-run"
        user_model = "not-run"

    agent_input = _optional_int_sum([item.accounting.agent_input_tokens for item in records])
    agent_output = _optional_int_sum([item.accounting.agent_output_tokens for item in records])
    user_input = _optional_int_sum([item.accounting.user_input_tokens for item in records])
    user_output = _optional_int_sum([item.accounting.user_output_tokens for item in records])
    agent_cost = _optional_float_sum([item.accounting.agent_cost_usd for item in records])
    user_cost = _optional_float_sum([item.accounting.user_cost_usd for item in records])
    total_cost = (
        float(agent_cost + user_cost) if agent_cost is not None and user_cost is not None else None
    )
    agent_turns = sum(item.accounting.agent_model_turns for item in records)
    user_turns = sum(item.accounting.user_model_turns for item in records)
    payload = {
        "agent": {
            "cost_usd": agent_cost,
            "input_tokens": agent_input,
            "model": agent_model,
            "model_turns": agent_turns,
            "output_tokens": agent_output,
        },
        "attempt_outcome_ids": [item.outcome_id for item in records],
        "namespace": "grace.episode-cost-derived.v1",
        "user": {
            "cost_usd": user_cost,
            "input_tokens": user_input,
            "model": user_model,
            "model_turns": user_turns,
            "output_tokens": user_output,
        },
    }
    return EpisodeCostSummary(
        attempt_count=len(records),
        infrastructure_attempt_count=sum(
            item.disposition == "infrastructure_error" for item in records
        ),
        agent=RoleCostSummary(
            model=agent_model,
            model_turns=agent_turns,
            input_tokens=agent_input,
            output_tokens=agent_output,
            cost_usd=agent_cost,
            usage_complete=agent_input is not None and agent_output is not None,
            cost_complete=agent_cost is not None,
        ),
        user_simulator=RoleCostSummary(
            model=user_model,
            model_turns=user_turns,
            input_tokens=user_input,
            output_tokens=user_output,
            cost_usd=user_cost,
            usage_complete=user_input is not None and user_output is not None,
            cost_complete=user_cost is not None,
        ),
        total_cost_usd=total_cost,
        duration_seconds=sum(item.accounting.duration_seconds for item in records),
        derived_hash=canonical_sha256(payload),
    )


__all__ = [
    "EpisodeCostSummary",
    "RoleCostSummary",
    "derive_cost_summary",
]
