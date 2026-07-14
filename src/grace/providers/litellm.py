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

"""Explicit, bounded LiteLLM adapter for GRACE's supported Gemini routes.

LiteLLM is imported lazily and only in production mode.  This matters because
LiteLLM's development-mode import calls ``dotenv.load_dotenv()``, which would
otherwise make ambient ``.env`` files part of GRACE's credential selection.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NoReturn

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from grace.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTransportError,
)
from grace.providers.auth import ADCLoader, AuthReadiness, SecretResolver, discover_auth
from grace.providers.base import (
    PromptRequest,
    ProviderAttemptContext,
    ProviderAttemptHook,
    ProviderAttemptRecord,
    ProviderCallResult,
    UsageRecord,
)
from grace.providers.config import (
    GoogleAIStudioConfig,
    ProviderDescriptor,
    ProviderRuntimeConfig,
    VertexAIADCConfig,
)
from grace.providers.parsing import parse_json_response, require_nonempty_content


CompletionFn = Callable[..., Any]
SleepFn = Callable[[float], None]
MonotonicFn = Callable[[], float]
RandomFn = Callable[[], float]

_LITELLM_IMPORT_LOCK = threading.Lock()
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
INPUT_TOKEN_BOUND_METHOD: Literal["canonical_json_utf8_bytes_plus_512_v1"] = (
    "canonical_json_utf8_bytes_plus_512_v1"
)
INPUT_TOKEN_BOUND_OVERHEAD_BYTES = 512


def _load_litellm() -> Any:
    """Import LiteLLM without permitting its implicit dotenv side effect.

    If another library already imported LiteLLM in development mode, the
    dotenv side effect may already have happened.  Continuing would make the
    credential boundary unverifiable, so GRACE fails closed.
    """

    with _LITELLM_IMPORT_LOCK:
        imported = sys.modules.get("litellm")
        if imported is not None:
            mode = str(getattr(imported, "litellm_mode", "DEV")).upper()
            if mode != "PRODUCTION":
                raise ProviderConfigurationError(
                    "LiteLLM was already imported outside PRODUCTION mode; "
                    "start a fresh process with LITELLM_MODE=PRODUCTION"
                )
            return imported

        configured_mode = os.environ.get("LITELLM_MODE")
        if configured_mode is not None and configured_mode.upper() != "PRODUCTION":
            raise ProviderConfigurationError(
                "LITELLM_MODE must be PRODUCTION before GRACE imports LiteLLM"
            )

        # LiteLLM reads this value during import.  Setting it before the import
        # is the supported way to prevent development-mode dotenv loading.
        os.environ["LITELLM_MODE"] = "PRODUCTION"
        try:
            return importlib.import_module("litellm")
        except ImportError:
            raise ProviderConfigurationError(
                "LiteLLM is not installed; install 'redmind-grace[gemini]'"
            ) from None
        except Exception:
            raise ProviderConfigurationError(
                "LiteLLM import failed; verify the supported dependency installation"
            ) from None


def ensure_litellm_production() -> Any:
    """Safely load LiteLLM before an optional integration imports it eagerly.

    Integrations with eager imports can use this boundary to prevent a dependency
    from reintroducing development-mode dotenv loading. Product callers normally
    do not need to call it because :class:`LiteLLMProvider` remains lazy.
    """

    return _load_litellm()


def _read(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _status_code(exception: Exception) -> int | None:
    value = getattr(exception, "status_code", None)
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    response = getattr(exception, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    return None


def _error_code(exception: Exception) -> str | None:
    value = getattr(exception, "code", None)
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    candidate = str(value)
    return candidate if _SAFE_CODE.fullmatch(candidate) else None


def _classify_exception(exception: Exception) -> tuple[str, bool, int | None, str | None]:
    """Classify an SDK exception without serializing its potentially sensitive text."""

    status = _status_code(exception)
    code = _error_code(exception)
    class_name = type(exception).__name__
    lowered = class_name.lower()

    if status in (401, 403) or any(
        token in lowered
        for token in ("authentication", "permissiondenied", "unauthorized", "forbidden")
    ):
        return "authentication", False, status, code

    if status in (400, 404, 422) or any(
        token in lowered for token in ("badrequest", "notfound", "unprocessable")
    ):
        return "request_configuration", False, status, code

    retryable = (
        status in (408, 409, 425, 429)
        or (status is not None and 500 <= status <= 599)
        or any(
            token in lowered
            for token in (
                "timeout",
                "ratelimit",
                "apiconnection",
                "serviceunavailable",
                "internalserver",
                "temporar",
            )
        )
    )
    return ("transient_transport" if retryable else "transport"), retryable, status, code


def _gemini_transport_schema(value: Any) -> Any:
    """Return a provider-safe schema while preserving strict local validation.

    Gemini structured output does not consistently enforce every JSON Schema
    keyword accepted by Draft 2020-12.  In particular, Vertex can return an
    empty response for tiny schemas containing ``const`` and
    ``additionalProperties``.  Remove those transport-only constraints and
    translate string constants to supported one-value enums.  The original
    caller schema is still compiled and enforced locally after parsing.
    """

    if isinstance(value, list):
        return [_gemini_transport_schema(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    normalized: dict[str, Any] = {}
    const_value = value.get("const")
    for key, item in value.items():
        if key in {"const", "additionalProperties"}:
            continue
        normalized[key] = _gemini_transport_schema(item)
    if isinstance(const_value, str):
        existing = normalized.get("enum")
        if not isinstance(existing, list):
            normalized["enum"] = [const_value]
    return normalized


def _response_format(
    request: PromptRequest,
    *,
    normalize_for_gemini: bool,
) -> dict[str, Any] | None:
    if not request.expect_json:
        return None
    if request.response_schema is None:
        return {"type": "json_object"}
    schema: dict[str, Any] = request.response_schema
    if normalize_for_gemini:
        normalized = _gemini_transport_schema(schema)
        if not isinstance(normalized, dict):  # defensive: root schema is typed as a mapping
            raise ProviderConfigurationError("normalized response schema must be an object")
        schema = normalized
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "grace_response",
            "schema": schema,
            "strict": True,
        },
    }


def _response_validator(request: PromptRequest) -> Draft202012Validator | None:
    """Compile caller-owned JSON Schema before any provider dispatch."""

    if request.response_schema is None:
        return None
    try:
        Draft202012Validator.check_schema(request.response_schema)
        return Draft202012Validator(request.response_schema)
    except JSONSchemaError:
        raise ProviderConfigurationError("response_schema is not valid JSON Schema") from None


def _input_token_bound(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None,
    temperature: float,
    max_tokens: int | None,
) -> int:
    """Conservative input-token bound without exposing request content.

    Every tokenizer token consumes at least one byte of its UTF-8 input.  The
    canonical payload byte length is therefore an upper bound for visible
    input tokens; a fixed, versioned overhead covers provider-side message
    framing that is not represented in the caller's content.
    """

    payload: dict[str, Any] = {
        "max_tokens": max_tokens,
        "messages": messages,
        "model": model,
        "num_retries": 0,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return len(encoded) + INPUT_TOKEN_BOUND_OVERHEAD_BYTES


def _extract_response(response: object) -> tuple[str, str | None, str | None, str | None, object]:
    choices = _read(response, "choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes, bytearray))
        or not choices
    ):
        raise ProviderResponseError("provider response did not contain a completion choice")

    choice = choices[0]
    message = _read(choice, "message")
    content = require_nonempty_content(_read(message, "content"))

    finish_reason = _read(choice, "finish_reason")
    provider_model = _read(response, "model")
    request_id = _read(response, "id")
    return (
        content,
        finish_reason if isinstance(finish_reason, str) else None,
        provider_model if isinstance(provider_model, str) else None,
        request_id if isinstance(request_id, str) else None,
        _read(response, "usage"),
    )


def _usage_counts(
    usage: object, *, content: str
) -> tuple[
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    Literal["provider", "missing"],
]:
    if usage is None:
        return None, None, None, None, None, "missing"

    input_tokens = _integer_or_none(_read(usage, "prompt_tokens", _read(usage, "input_tokens")))
    output_tokens = _integer_or_none(
        _read(usage, "completion_tokens", _read(usage, "output_tokens"))
    )
    total_tokens = _integer_or_none(_read(usage, "total_tokens"))

    prompt_details = _read(usage, "prompt_tokens_details")
    completion_details = _read(usage, "completion_tokens_details")
    cached_tokens = _integer_or_none(_read(prompt_details, "cached_tokens"))
    reasoning_tokens = _integer_or_none(_read(completion_details, "reasoning_tokens"))

    # LiteLLM sometimes materializes a zero-filled usage object when the
    # upstream response did not report usage.  Non-empty generated text cannot
    # truthfully have an all-zero completion, so preserve the missing state.
    all_zero = input_tokens == 0 and output_tokens == 0 and total_tokens in (None, 0)
    if input_tokens is None or output_tokens is None or (content and all_zero):
        return None, None, None, None, None, "missing"
    return (
        input_tokens,
        output_tokens,
        total_tokens,
        cached_tokens,
        reasoning_tokens,
        "provider",
    )


class LiteLLMProvider:
    """Synchronous Gemini provider with an explicit route and one retry owner."""

    def __init__(
        self,
        route: VertexAIADCConfig | GoogleAIStudioConfig,
        *,
        runtime: ProviderRuntimeConfig | None = None,
        secret_resolver: SecretResolver | None = None,
        adc_loader: ADCLoader | None = None,
        completion_fn: CompletionFn | None = None,
        attempt_hook: ProviderAttemptHook | None = None,
        sleep_fn: SleepFn = time.sleep,
        monotonic_fn: MonotonicFn = time.monotonic,
        random_fn: RandomFn = random.random,
    ) -> None:
        if not isinstance(route, (VertexAIADCConfig, GoogleAIStudioConfig)):
            raise ProviderConfigurationError("unsupported LiteLLM provider route")
        self._route = route
        self._runtime = runtime or ProviderRuntimeConfig()
        self._secret_resolver = secret_resolver
        self._adc_loader = adc_loader
        self._completion_fn = completion_fn
        self._attempt_hook = attempt_hook
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._random = random_fn

    @property
    def model(self) -> str:
        return self._route.litellm_model

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._route.safe_descriptor()

    def validate_auth(self) -> AuthReadiness:
        """Discover the configured credential source without a provider request."""

        return discover_auth(
            self._route,
            secret_resolver=self._secret_resolver,
            adc_loader=self._adc_loader,
        )

    def _completion(self) -> CompletionFn:
        if self._completion_fn is not None:
            return self._completion_fn
        module = _load_litellm()
        completion = getattr(module, "completion", None)
        if not callable(completion):
            raise ProviderConfigurationError("installed LiteLLM does not expose completion()")
        return completion

    def _route_kwargs(self) -> dict[str, Any]:
        if isinstance(self._route, VertexAIADCConfig):
            return {
                "vertex_project": self._route.project,
                "vertex_location": self._route.location,
            }
        from grace.providers.auth import resolve_ai_studio_api_key

        return {
            "api_key": resolve_ai_studio_api_key(self._route, self._secret_resolver),
        }

    def _backoff(self, failed_attempt: int, deadline: float) -> None:
        base = min(
            self._runtime.initial_backoff_seconds * (2 ** (failed_attempt - 1)),
            self._runtime.max_backoff_seconds,
        )
        centered_jitter = (2.0 * min(max(self._random(), 0.0), 1.0)) - 1.0
        delay = max(0.0, base * (1.0 + self._runtime.jitter_ratio * centered_jitter))
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            return
        self._sleep(min(delay, remaining))

    @staticmethod
    def _raise_failure(category: str, attempts: list[ProviderAttemptRecord]) -> NoReturn:
        history = tuple(attempts)
        if category == "authentication":
            raise ProviderAuthenticationError(
                f"provider authentication failed after {len(attempts)} attempt(s)",
                attempts=history,
            ) from None
        if category == "request_configuration":
            raise ProviderConfigurationError(
                f"provider rejected request configuration after {len(attempts)} attempt(s)",
                attempts=history,
            ) from None
        raise ProviderTransportError(
            f"provider transport failed after {len(attempts)} attempt(s)",
            attempts=history,
        ) from None

    def _before_attempt(
        self,
        request: PromptRequest,
        *,
        attempt: int,
        temperature: float,
        timeout: float,
        elapsed: float,
        input_token_bound: int,
    ) -> object | None:
        if self._attempt_hook is None:
            return None
        context = ProviderAttemptContext(
            descriptor=self.descriptor,
            stage=request.stage,
            logical_call_id=request.logical_call_id,
            attempt=attempt,
            input_token_bound=input_token_bound,
            input_token_bound_method=INPUT_TOKEN_BOUND_METHOD,
            temperature=temperature,
            timeout_seconds=timeout,
            overall_timeout_seconds=self._runtime.overall_timeout_seconds,
            elapsed_since_call_start_seconds=elapsed,
            max_tokens=request.max_tokens,
        )
        return self._attempt_hook.before_attempt(context)

    def _after_attempt(
        self,
        reservation: object | None,
        *,
        attempt: ProviderAttemptRecord,
        usage: UsageRecord | None,
    ) -> None:
        if self._attempt_hook is None:
            return
        self._attempt_hook.after_attempt(reservation, attempt=attempt, usage=usage)

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        """Complete a request under bounded GRACE-owned retry and timeout policy."""

        response_validator = _response_validator(request)
        completion = self._completion()
        route_kwargs = self._route_kwargs()
        started = self._monotonic()
        deadline = started + self._runtime.overall_timeout_seconds
        attempts: list[ProviderAttemptRecord] = []
        temperature = request.temperature
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        response_format = _response_format(request, normalize_for_gemini=True)

        for attempt_number in range(1, self._runtime.max_attempts + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise ProviderTransportError(
                    f"provider overall timeout exhausted after {len(attempts)} attempt(s)",
                    attempts=tuple(attempts),
                ) from None
            timeout = min(self._runtime.attempt_timeout_seconds, remaining)
            input_token_bound = _input_token_bound(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=request.max_tokens,
            )
            reservation = self._before_attempt(
                request,
                attempt=attempt_number,
                temperature=temperature,
                timeout=timeout,
                elapsed=max(0.0, self._monotonic() - started),
                input_token_bound=input_token_bound,
            )
            dispatch_started = self._monotonic()
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
                "num_retries": 0,
                **route_kwargs,
            }
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens
            if response_format is not None:
                kwargs["response_format"] = response_format

            try:
                response = completion(**kwargs)
            except Exception as exception:
                latency = max(0.0, self._monotonic() - dispatch_started)
                category, retryable, status, code = _classify_exception(exception)
                will_retry = retryable and attempt_number < self._runtime.max_attempts
                attempt_record = ProviderAttemptRecord(
                    attempt=attempt_number,
                    temperature=temperature,
                    timeout_seconds=timeout,
                    overall_timeout_seconds=self._runtime.overall_timeout_seconds,
                    elapsed_since_call_start_seconds=max(0.0, self._monotonic() - started),
                    latency_seconds=latency,
                    outcome="retry" if will_retry else "failed",
                    category=category,
                    retryable=retryable,
                    status_code=status,
                    error_code=code,
                )
                attempts.append(attempt_record)
                self._after_attempt(reservation, attempt=attempt_record, usage=None)
                if not will_retry:
                    self._raise_failure(category, attempts)
                self._backoff(attempt_number, deadline)
                continue

            if self._monotonic() > deadline:
                latency = max(0.0, self._monotonic() - dispatch_started)
                attempt_record = ProviderAttemptRecord(
                    attempt=attempt_number,
                    temperature=temperature,
                    timeout_seconds=timeout,
                    overall_timeout_seconds=self._runtime.overall_timeout_seconds,
                    elapsed_since_call_start_seconds=max(0.0, self._monotonic() - started),
                    latency_seconds=latency,
                    outcome="failed",
                    category="overall_timeout",
                    retryable=False,
                )
                attempts.append(attempt_record)
                self._after_attempt(reservation, attempt=attempt_record, usage=None)
                raise ProviderTransportError(
                    f"provider overall timeout exceeded after {len(attempts)} attempt(s)",
                    attempts=tuple(attempts),
                ) from None

            try:
                raw_content, finish_reason, provider_model, request_id, raw_usage = (
                    _extract_response(response)
                )
                parsed_content = parse_json_response(raw_content) if request.expect_json else None
                if response_validator is not None:
                    try:
                        response_validator.validate(parsed_content)
                    except JSONSchemaValidationError:
                        raise ProviderResponseError(
                            "provider JSON did not match response_schema"
                        ) from None
            except ProviderResponseError as exception:
                latency = max(0.0, self._monotonic() - dispatch_started)
                category = (
                    "empty_response"
                    if "empty" in str(exception).lower()
                    else "invalid_structured_response"
                )
                will_retry = attempt_number < self._runtime.max_attempts
                attempt_record = ProviderAttemptRecord(
                    attempt=attempt_number,
                    temperature=temperature,
                    timeout_seconds=timeout,
                    overall_timeout_seconds=self._runtime.overall_timeout_seconds,
                    elapsed_since_call_start_seconds=max(0.0, self._monotonic() - started),
                    latency_seconds=latency,
                    outcome="retry" if will_retry else "failed",
                    category=category,
                    retryable=True,
                )
                attempts.append(attempt_record)
                self._after_attempt(reservation, attempt=attempt_record, usage=None)
                if not will_retry:
                    raise ProviderResponseError(
                        f"provider response validation failed after {len(attempts)} attempt(s)",
                        attempts=tuple(attempts),
                    ) from None
                temperature = self._runtime.structured_response_retry_temperature
                self._backoff(attempt_number, deadline)
                continue
            except Exception:
                latency = max(0.0, self._monotonic() - dispatch_started)
                will_retry = attempt_number < self._runtime.max_attempts
                attempt_record = ProviderAttemptRecord(
                    attempt=attempt_number,
                    temperature=temperature,
                    timeout_seconds=timeout,
                    overall_timeout_seconds=self._runtime.overall_timeout_seconds,
                    elapsed_since_call_start_seconds=max(0.0, self._monotonic() - started),
                    latency_seconds=latency,
                    outcome="retry" if will_retry else "failed",
                    category="unsupported_response",
                    retryable=True,
                )
                attempts.append(attempt_record)
                self._after_attempt(reservation, attempt=attempt_record, usage=None)
                if not will_retry:
                    raise ProviderResponseError(
                        f"provider response normalization failed after {len(attempts)} attempt(s)",
                        attempts=tuple(attempts),
                    ) from None
                temperature = self._runtime.structured_response_retry_temperature
                self._backoff(attempt_number, deadline)
                continue

            latency = max(0.0, self._monotonic() - dispatch_started)
            attempt_record = ProviderAttemptRecord(
                attempt=attempt_number,
                temperature=temperature,
                timeout_seconds=timeout,
                overall_timeout_seconds=self._runtime.overall_timeout_seconds,
                elapsed_since_call_start_seconds=max(0.0, self._monotonic() - started),
                latency_seconds=latency,
                outcome="succeeded",
                category="structured_response" if request.expect_json else "text_response",
            )
            attempts.append(attempt_record)
            (
                input_tokens,
                output_tokens,
                total_tokens,
                cached_tokens,
                reasoning_tokens,
                usage_source,
            ) = _usage_counts(raw_usage, content=raw_content)
            usage = UsageRecord(
                model=self.model,
                route=self._route.route,
                provider_response_model=provider_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens_reported=total_tokens,
                cached_input_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                usage_source=usage_source,
                latency_seconds=max(0.0, self._monotonic() - started),
                attempts=len(attempts),
                retries=len(attempts) - 1,
                finish_reason=finish_reason,
                request_id=request_id,
            )
            self._after_attempt(reservation, attempt=attempt_record, usage=usage)
            return ProviderCallResult(
                parsed_content=parsed_content,
                raw_content=raw_content,
                descriptor=self.descriptor,
                usage=usage,
                attempts=tuple(attempts),
            )

        raise ProviderTransportError("provider attempt budget exhausted", attempts=tuple(attempts))


__all__ = [
    "INPUT_TOKEN_BOUND_METHOD",
    "INPUT_TOKEN_BOUND_OVERHEAD_BYTES",
    "LiteLLMProvider",
    "ensure_litellm_production",
]
