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

"""Provider-injected HCE reference baseline for telecom reproduction."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grace.artifacts.provenance import hash_text
from grace.errors import ProviderResponseError
from grace.providers.base import (
    LLMProvider,
    PromptRequest,
    ProviderAttemptRecord,
    UsageRecord,
    make_logical_call_id,
)

from .prompts import HCE_SYSTEM, HCE_USER


_OUTER_FENCE = re.compile(r"^```(?:\w*)\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class HCEResult(BaseModel):
    """One deterministic-contract HCE update result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evolved_prompt: str = Field(min_length=1)
    input_prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnosis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rendered_user_prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_prompt_length: int = Field(ge=1)
    output_prompt_length: int = Field(ge=1)
    length_delta: int
    model: str = Field(min_length=1)
    usage: UsageRecord
    attempts: tuple[ProviderAttemptRecord, ...] = ()

    @model_validator(mode="after")
    def validate_content_telemetry(self) -> "HCEResult":
        if self.output_prompt_hash != hash_text(self.evolved_prompt):
            raise ValueError("output_prompt_hash does not match evolved_prompt")
        if self.output_prompt_length != len(self.evolved_prompt):
            raise ValueError("output_prompt_length does not match evolved_prompt")
        if self.length_delta != self.output_prompt_length - self.input_prompt_length:
            raise ValueError("length_delta does not match prompt lengths")
        return self


def _strip_outer_fence(value: str) -> str:
    stripped = value.strip()
    match = _OUTER_FENCE.fullmatch(stripped)
    return match.group(1).strip() if match else stripped


def evolve_hce(
    *,
    current_prompt: str,
    diagnosis_report: str,
    provider: LLMProvider,
) -> HCEResult:
    """Run the frozen one-call HCE baseline.

    HCE is intentionally not imported by the public ``grace`` package and is
    never used as a product Evolution implementation.
    """

    if not current_prompt.strip():
        raise ValueError("current_prompt must not be empty")
    if not diagnosis_report.strip():
        raise ValueError("diagnosis_report must not be empty")
    rendered_user = HCE_USER.format(
        current_prompt=current_prompt,
        diagnosis_report=diagnosis_report,
    )
    logical_call_id = make_logical_call_id(
        stage="hce_evolution",
        system_prompt=HCE_SYSTEM,
        user_prompt=rendered_user,
        model=provider.model,
        binding_id=hash_text(f"{hash_text(current_prompt)}:{hash_text(diagnosis_report)}"),
    )
    response = provider.complete(
        PromptRequest(
            system_prompt=HCE_SYSTEM,
            user_prompt=rendered_user,
            stage="hce_evolution",
            logical_call_id=logical_call_id,
            expect_json=False,
            temperature=0.0,
            max_tokens=65536,
        )
    )
    raw = response.raw_content
    if raw is None and isinstance(response.parsed_content, str):
        raw = response.parsed_content
    if raw is None:
        raise ProviderResponseError("HCE provider response did not contain text")
    evolved = _strip_outer_fence(raw)
    if not evolved:
        raise ProviderResponseError("HCE provider response was empty after normalization")
    return HCEResult(
        evolved_prompt=evolved,
        input_prompt_hash=hash_text(current_prompt),
        output_prompt_hash=hash_text(evolved),
        diagnosis_hash=hash_text(diagnosis_report),
        system_prompt_hash=hash_text(HCE_SYSTEM),
        rendered_user_prompt_hash=hash_text(rendered_user),
        input_prompt_length=len(current_prompt),
        output_prompt_length=len(evolved),
        length_delta=len(evolved) - len(current_prompt),
        model=provider.model,
        usage=response.usage,
        attempts=response.attempts,
    )


__all__ = ["HCEResult", "evolve_hce"]
