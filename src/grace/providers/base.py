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

"""Provider-neutral request, response, attempt, and usage contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from grace.providers.config import ProviderDescriptor


def make_logical_call_id(
    *,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    binding_id: str | None = None,
) -> str:
    """Hash one logical provider call without retaining prompt content.

    The exact preimage is canonical JSON under namespace
    ``grace.logical-provider-call.v1``.  Prompt hashes preserve every UTF-8
    whitespace byte; ``model`` must be the route-qualified provider model.
    ``binding_id`` can additionally bind a state, reflection, or diagnosis
    fingerprint whose own semantics are owned by the caller.
    """

    if not stage.strip() or not model.strip():
        raise ValueError("stage and model must not be blank")
    if binding_id is not None and not binding_id.strip():
        raise ValueError("binding_id must not be blank when supplied")
    preimage = {
        "binding_id": binding_id,
        "model": model,
        "namespace": "grace.logical-provider-call.v1",
        "stage": stage,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
    }
    canonical = json.dumps(
        preimage,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PromptRequest(BaseModel):
    """One model completion requested by a named GRACE pipeline stage.

    Reliability controls intentionally do not live here.  The provider runtime
    policy owns the bounded attempt and timeout envelope for every stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prompt: str
    user_prompt: str
    stage: str = Field(default="unspecified", min_length=1)
    logical_call_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    expect_json: bool = True
    response_schema: dict[str, JsonValue] | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def schema_requires_json(self) -> "PromptRequest":
        if self.response_schema is not None and not self.expect_json:
            raise ValueError("response_schema requires expect_json=True")
        return self


class ProviderAttemptRecord(BaseModel):
    """Credential-free telemetry for one actual provider dispatch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_seconds: float = Field(gt=0.0)
    overall_timeout_seconds: float | None = Field(default=None, gt=0.0)
    elapsed_since_call_start_seconds: float | None = Field(default=None, ge=0.0)
    latency_seconds: float = Field(ge=0.0)
    outcome: Literal["succeeded", "retry", "failed"]
    category: str = Field(min_length=1)
    retryable: bool = False
    status_code: int | None = Field(default=None, ge=100, le=599)
    error_code: str | None = None


class ProviderAttemptContext(BaseModel):
    """Credential- and prompt-free context supplied before an actual dispatch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    descriptor: ProviderDescriptor
    stage: str = Field(min_length=1)
    logical_call_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    attempt: int = Field(ge=1)
    input_token_bound: int = Field(ge=0)
    input_token_bound_method: Literal["canonical_json_utf8_bytes_plus_512_v1"] = (
        "canonical_json_utf8_bytes_plus_512_v1"
    )
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_seconds: float = Field(gt=0.0)
    overall_timeout_seconds: float = Field(gt=0.0)
    elapsed_since_call_start_seconds: float = Field(ge=0.0)
    max_tokens: int | None = Field(default=None, ge=1)


class UsageRecord(BaseModel):
    """Provider call accounting with an explicit missing-data state.

    Token counts and provider model identifiers are authoritative when the SDK
    returns them.  Monetary cost remains optional and requires a recorded
    pricing source/version when present.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    route: str = Field(default="custom", min_length=1)
    provider_response_model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens_reported: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    usage_source: Literal["provider", "estimated", "missing"] = "provider"
    latency_seconds: float = Field(ge=0.0)
    attempts: int = Field(default=1, ge=1)
    retries: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    request_id: str | None = None
    cost_usd: float | None = Field(default=None, ge=0.0)
    pricing_source: str | None = None
    pricing_version: str | None = None

    @model_validator(mode="after")
    def validate_accounting(self) -> "UsageRecord":
        if self.cost_usd is not None and (not self.pricing_source or not self.pricing_version):
            raise ValueError(
                "pricing_source and pricing_version are required when cost_usd is recorded"
            )
        if self.attempts != self.retries + 1:
            raise ValueError("attempts must equal retries + 1")
        token_values = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens_reported,
            self.cached_input_tokens,
            self.reasoning_tokens,
        )
        if self.usage_source == "missing" and any(value is not None for value in token_values):
            raise ValueError("missing usage cannot claim provider token counts")
        if self.usage_source != "missing" and (
            self.input_tokens is None or self.output_tokens is None
        ):
            raise ValueError("provider/estimated usage requires input and output tokens")
        return self

    @property
    def total_tokens(self) -> int | None:
        """Return a derived total only when both counters are available."""

        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class ProviderCallResult(BaseModel):
    """JSON-safe content, usage, and attempt telemetry from one logical call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parsed_content: JsonValue | None = None
    raw_content: str | None = None
    descriptor: ProviderDescriptor | None = None
    usage: UsageRecord
    attempts: tuple[ProviderAttemptRecord, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> "ProviderCallResult":
        if self.parsed_content is None and self.raw_content is None:
            raise ValueError("a provider result must contain parsed_content or raw_content")
        if self.attempts:
            if len(self.attempts) != self.usage.attempts:
                raise ValueError("attempt history length must match usage.attempts")
            if self.attempts[-1].outcome != "succeeded":
                raise ValueError("a returned provider result must end in a successful attempt")
        return self


@runtime_checkable
class ProviderAttemptHook(Protocol):
    """Persistence/budget boundary around every actual provider dispatch.

    ``before_attempt`` can atomically reserve a call budget and returns an
    opaque reservation owned by the hook.  ``after_attempt`` receives the
    typed, safe attempt record for both success and failure.  A missing usage
    value means the reservation must remain conservatively accounted for.
    Hook failures propagate and stop dispatch/retry; the adapter never retries
    around a failed persistence boundary.
    """

    def before_attempt(self, context: ProviderAttemptContext) -> object:
        """Persist/reserve before the provider can receive the request."""

        ...

    def after_attempt(
        self,
        reservation: object,
        *,
        attempt: ProviderAttemptRecord,
        usage: UsageRecord | None,
    ) -> None:
        """Settle or retain a reservation after one dispatch."""

        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal synchronous provider interface used by the GRACE engine."""

    @property
    def model(self) -> str:
        """Return the configured provider-qualified model identifier."""

        ...

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        """Complete ``request`` or raise a typed public provider error."""

        ...


Provider = LLMProvider
ProviderProtocol = LLMProvider


__all__ = [
    "LLMProvider",
    "make_logical_call_id",
    "PromptRequest",
    "Provider",
    "ProviderAttemptContext",
    "ProviderAttemptHook",
    "ProviderAttemptRecord",
    "ProviderCallResult",
    "ProviderProtocol",
    "UsageRecord",
]
