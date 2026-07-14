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

"""Offline tests for retry ownership, route isolation, and provider telemetry."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from grace.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTransportError,
)
from grace.providers.base import (
    PromptRequest,
    ProviderAttemptContext,
    ProviderAttemptRecord,
    UsageRecord,
)
from grace.providers.config import (
    GoogleAIStudioConfig,
    ProviderRuntimeConfig,
    VertexAIADCConfig,
)
from grace.providers.litellm import (
    INPUT_TOKEN_BOUND_METHOD,
    INPUT_TOKEN_BOUND_OVERHEAD_BYTES,
    LiteLLMProvider,
    _load_litellm,
)


def _response(
    content: str = '{"ok": true}',
    *,
    usage: object = ...,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": "request-123",
        "model": "gemini-2.5-flash-001",
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is ...:
        result["usage"] = {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    elif usage is not None:
        result["usage"] = usage
    return result


def _runtime(**overrides: object) -> ProviderRuntimeConfig:
    values: dict[str, object] = {
        "max_attempts": 3,
        "attempt_timeout_seconds": 5,
        "overall_timeout_seconds": 30,
        "initial_backoff_seconds": 0,
        "max_backoff_seconds": 0,
        "jitter_ratio": 0,
    }
    values.update(overrides)
    return ProviderRuntimeConfig(**values)


def test_vertex_route_and_structured_response_contract() -> None:
    calls: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _response()

    provider = LiteLLMProvider(
        VertexAIADCConfig(project="private-project", location="us-central1"),
        completion_fn=completion,
    )
    result = provider.complete(
        PromptRequest(
            system_prompt="system",
            user_prompt="user",
            stage="p2g",
            response_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
            },
            max_tokens=100,
        )
    )

    assert result.parsed_content == {"ok": True}
    assert result.usage.model == "vertex_ai/gemini-2.5-flash"
    assert result.usage.provider_response_model == "gemini-2.5-flash-001"
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.cached_input_tokens == 3
    assert result.usage.reasoning_tokens == 2
    assert result.usage.attempts == 1
    assert result.descriptor is not None
    assert "private-project" not in result.descriptor.model_dump_json()

    call = calls[0]
    assert call["model"] == "vertex_ai/gemini-2.5-flash"
    assert call["vertex_project"] == "private-project"
    assert call["vertex_location"] == "us-central1"
    assert "api_key" not in call
    assert call["num_retries"] == 0
    assert call["timeout"] == 300.0
    assert call["max_tokens"] == 100
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "grace_response",
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
            },
            "strict": True,
        },
    }


def test_gemini_transport_schema_is_compatible_but_local_validation_stays_strict() -> None:
    calls: list[dict[str, Any]] = []
    provider = LiteLLMProvider(
        VertexAIADCConfig(project="private-project", location="us-central1"),
        runtime=_runtime(max_attempts=1),
        completion_fn=lambda **kwargs: calls.append(kwargs) or _response('{"op":"AddNode"}'),
    )
    caller_schema = {
        "type": "object",
        "properties": {"op": {"type": "string", "const": "AddNode"}},
        "required": ["op"],
        "additionalProperties": False,
    }

    result = provider.complete(
        PromptRequest(
            system_prompt="system",
            user_prompt="user",
            response_schema=caller_schema,
        )
    )

    assert result.parsed_content == {"op": "AddNode"}
    transported = calls[0]["response_format"]["json_schema"]["schema"]
    assert transported == {
        "type": "object",
        "properties": {"op": {"type": "string", "enum": ["AddNode"]}},
        "required": ["op"],
    }

    rejecting = LiteLLMProvider(
        VertexAIADCConfig(project="private-project", location="us-central1"),
        runtime=_runtime(max_attempts=1),
        completion_fn=lambda **_kwargs: _response('{"op":"Wrong"}'),
    )
    with pytest.raises(ProviderResponseError, match="validation failed"):
        rejecting.complete(
            PromptRequest(
                system_prompt="system",
                user_prompt="user",
                response_schema=caller_schema,
            )
        )


def test_ai_studio_route_passes_only_explicit_key() -> None:
    calls: list[dict[str, Any]] = []

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(api_key_env="SELECTED_KEY"),
        secret_resolver=lambda name: "selected-secret" if name == "SELECTED_KEY" else None,
        completion_fn=lambda **kwargs: calls.append(kwargs) or _response(),
    )
    result = provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))

    assert result.usage.route == "google_ai_studio"
    assert calls[0]["model"] == "gemini/gemini-2.5-flash"
    assert calls[0]["api_key"] == "selected-secret"
    assert "vertex_project" not in calls[0]
    assert "vertex_location" not in calls[0]


def test_missing_usage_remains_nullable_instead_of_zero() -> None:
    for usage in (None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}):
        provider = LiteLLMProvider(
            GoogleAIStudioConfig(),
            secret_resolver=lambda _name: "key",
            completion_fn=lambda **_kwargs: _response(usage=usage),
        )

        result = provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))

        assert result.usage.usage_source == "missing"
        assert result.usage.input_tokens is None
        assert result.usage.output_tokens is None
        assert result.usage.total_tokens is None


def test_structured_response_retry_uses_repair_temperature() -> None:
    responses = iter([_response("not json"), _response('{"repaired": true}')])
    calls: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return next(responses)

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
    )
    result = provider.complete(
        PromptRequest(system_prompt="system", user_prompt="user", temperature=0.1)
    )

    assert [call["temperature"] for call in calls] == [0.1, 0.5]
    assert all(call["num_retries"] == 0 for call in calls)
    assert [attempt.outcome for attempt in result.attempts] == ["retry", "succeeded"]
    assert result.attempts[0].category == "invalid_structured_response"
    assert result.usage.attempts == 2
    assert result.usage.retries == 1


def test_response_schema_mismatch_retries_before_success_settlement() -> None:
    class RecordingHook:
        def __init__(self) -> None:
            self.contexts: list[ProviderAttemptContext] = []
            self.settlements: list[tuple[object, ProviderAttemptRecord, UsageRecord | None]] = []

        def before_attempt(self, context: ProviderAttemptContext) -> object:
            self.contexts.append(context)
            return f"reservation-{context.attempt}"

        def after_attempt(
            self,
            reservation: object,
            *,
            attempt: ProviderAttemptRecord,
            usage: UsageRecord | None,
        ) -> None:
            self.settlements.append((reservation, attempt, usage))

    responses = iter([_response('{"ok":"wrong-type"}'), _response('{"ok":true}')])
    hook = RecordingHook()
    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(),
        secret_resolver=lambda _name: "key",
        completion_fn=lambda **_kwargs: next(responses),
        attempt_hook=hook,
    )
    result = provider.complete(
        PromptRequest(
            system_prompt="system",
            user_prompt="user",
            logical_call_id="a" * 64,
            response_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
    )

    assert result.parsed_content == {"ok": True}
    assert [attempt.outcome for attempt in result.attempts] == ["retry", "succeeded"]
    assert result.attempts[0].category == "invalid_structured_response"
    assert hook.settlements[0][2] is None
    assert hook.settlements[1][2] is result.usage


def test_invalid_caller_response_schema_fails_before_dispatch() -> None:
    calls = 0

    def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _response()

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
    )
    with pytest.raises(ProviderConfigurationError, match="response_schema"):
        provider.complete(
            PromptRequest(
                system_prompt="system",
                user_prompt="user",
                response_schema={"type": "not-a-json-schema-type"},
            )
        )
    assert calls == 0


def test_transport_retry_preserves_temperature_and_is_bounded() -> None:
    class RateLimitError(Exception):
        status_code = 429
        code = "rate_limit"

    temperatures: list[float] = []

    def completion(**kwargs: Any) -> dict[str, Any]:
        temperatures.append(kwargs["temperature"])
        if len(temperatures) < 3:
            raise RateLimitError("sensitive upstream body")
        return _response()

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
    )
    result = provider.complete(
        PromptRequest(system_prompt="system", user_prompt="user", temperature=0.2)
    )

    assert temperatures == [0.2, 0.2, 0.2]
    assert [attempt.status_code for attempt in result.attempts] == [429, 429, None]
    assert [attempt.error_code for attempt in result.attempts] == [
        "rate_limit",
        "rate_limit",
        None,
    ]
    assert result.usage.attempts == 3


def test_retry_exhaustion_raises_safe_transport_error() -> None:
    class APIConnectionError(Exception):
        pass

    calls = 0

    def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise APIConnectionError("token=super-secret")

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
    )

    with pytest.raises(ProviderTransportError) as captured:
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))

    assert calls == 3
    assert "super-secret" not in str(captured.value)
    assert len(captured.value.attempts) == 3
    assert all(isinstance(item, ProviderAttemptRecord) for item in captured.value.attempts)


def test_overall_deadline_stops_after_a_timed_out_attempt() -> None:
    class Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    clock = Clock()
    calls = 0

    def completion(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert kwargs["timeout"] == 5.0
        clock.now += 5.0
        raise TimeoutError("private timeout body")

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(
            attempt_timeout_seconds=5,
            overall_timeout_seconds=5,
        ),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
    )

    with pytest.raises(ProviderTransportError, match="overall timeout") as captured:
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))

    assert calls == 1
    assert len(captured.value.attempts) == 1
    attempt = captured.value.attempts[0]
    assert isinstance(attempt, ProviderAttemptRecord)
    assert attempt.category == "transient_transport"
    assert attempt.timeout_seconds == 5.0
    assert attempt.overall_timeout_seconds == 5.0
    assert attempt.latency_seconds == 5.0


def test_late_success_is_rejected_when_provider_ignores_timeout() -> None:
    class Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

    clock = Clock()

    def completion(**_kwargs: Any) -> dict[str, Any]:
        clock.now = 6.0
        return _response()

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(attempt_timeout_seconds=5, overall_timeout_seconds=5),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
        monotonic_fn=clock.monotonic,
    )

    with pytest.raises(ProviderTransportError, match="timeout exceeded") as captured:
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))

    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].category == "overall_timeout"


def test_authentication_failure_is_not_retried() -> None:
    class AuthenticationError(Exception):
        status_code = 401

    calls = 0

    def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AuthenticationError("secret credential")

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
    )

    with pytest.raises(ProviderAuthenticationError):
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))
    assert calls == 1


def test_invalid_json_is_bounded_to_three_total_attempts() -> None:
    calls = 0

    def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _response("invalid")

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
    )

    with pytest.raises(ProviderResponseError, match="3 attempt"):
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))
    assert calls == 3


def test_unexpected_response_object_failure_is_safe_and_bounded() -> None:
    class BrokenMessage:
        @property
        def content(self) -> str:
            raise RuntimeError("credential=super-secret")

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=BrokenMessage(), finish_reason=None)]
    )
    calls = 0

    def completion(**_kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        return response

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        runtime=_runtime(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
    )

    with pytest.raises(ProviderResponseError) as captured:
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))

    assert calls == 3
    assert "super-secret" not in str(captured.value)
    assert [attempt.category for attempt in captured.value.attempts] == [
        "unsupported_response",
        "unsupported_response",
        "unsupported_response",
    ]


def test_text_response_does_not_send_a_response_format() -> None:
    calls: list[dict[str, Any]] = []
    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        secret_resolver=lambda _name: "key",
        completion_fn=lambda **kwargs: calls.append(kwargs) or _response("plain text"),
    )

    result = provider.complete(
        PromptRequest(system_prompt="system", user_prompt="user", expect_json=False)
    )

    assert result.parsed_content is None
    assert result.raw_content == "plain text"
    assert "response_format" not in calls[0]


def test_already_imported_development_mode_litellm_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(litellm_mode="DEV"))

    with pytest.raises(ProviderConfigurationError, match="fresh process"):
        _load_litellm()


def test_attempt_hook_wraps_each_dispatch_and_receives_safe_records() -> None:
    class APIConnectionError(Exception):
        pass

    class RecordingHook:
        def __init__(self) -> None:
            self.contexts: list[ProviderAttemptContext] = []
            self.settlements: list[tuple[object, ProviderAttemptRecord, UsageRecord | None]] = []

        def before_attempt(self, context: ProviderAttemptContext) -> object:
            self.contexts.append(context)
            return f"reservation-{context.attempt}"

        def after_attempt(
            self,
            reservation: object,
            *,
            attempt: ProviderAttemptRecord,
            usage: UsageRecord | None,
        ) -> None:
            self.settlements.append((reservation, attempt, usage))

    calls = 0

    def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APIConnectionError("private provider response")
        return _response()

    hook = RecordingHook()
    provider = LiteLLMProvider(
        VertexAIADCConfig(project="private-project", location="us-central1"),
        runtime=_runtime(),
        completion_fn=completion,
        attempt_hook=hook,
    )
    result = provider.complete(
        PromptRequest(
            system_prompt="private system prompt",
            user_prompt="private trajectory",
            stage="evolution.reconstruction",
            logical_call_id="a" * 64,
            max_tokens=321,
        )
    )

    assert result.usage.attempts == 2
    assert [context.attempt for context in hook.contexts] == [1, 2]
    assert all(context.logical_call_id == "a" * 64 for context in hook.contexts)
    visible_content_bytes = len("private system prompt".encode()) + len(
        "private trajectory".encode()
    )
    assert all(
        context.input_token_bound > visible_content_bytes + INPUT_TOKEN_BOUND_OVERHEAD_BYTES
        for context in hook.contexts
    )
    assert all(
        context.input_token_bound_method == INPUT_TOKEN_BOUND_METHOD for context in hook.contexts
    )
    assert all(context.max_tokens == 321 for context in hook.contexts)
    serialized_context = "".join(context.model_dump_json() for context in hook.contexts)
    assert "private system prompt" not in serialized_context
    assert "private trajectory" not in serialized_context
    assert "private-project" not in serialized_context
    assert [reservation for reservation, _attempt, _usage in hook.settlements] == [
        "reservation-1",
        "reservation-2",
    ]
    assert hook.settlements[0][1].outcome == "retry"
    assert hook.settlements[0][2] is None
    assert hook.settlements[1][1].outcome == "succeeded"
    assert hook.settlements[1][2] is result.usage


def test_failed_budget_reservation_stops_before_provider_dispatch() -> None:
    class BudgetDenied(RuntimeError):
        pass

    class DenyingHook:
        def before_attempt(self, _context: ProviderAttemptContext) -> object:
            raise BudgetDenied("hard budget reached")

        def after_attempt(
            self,
            _reservation: object,
            *,
            attempt: ProviderAttemptRecord,
            usage: UsageRecord | None,
        ) -> None:
            raise AssertionError((attempt, usage))

    calls = 0

    def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _response()

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
        attempt_hook=DenyingHook(),
    )

    with pytest.raises(BudgetDenied, match="hard budget"):
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))
    assert calls == 0


def test_failed_attempt_persistence_stops_without_an_outer_retry() -> None:
    class PersistenceFailure(RuntimeError):
        pass

    class FailingSink:
        def before_attempt(self, context: ProviderAttemptContext) -> object:
            return context.attempt

        def after_attempt(
            self,
            _reservation: object,
            *,
            attempt: ProviderAttemptRecord,
            usage: UsageRecord | None,
        ) -> None:
            assert attempt.outcome == "succeeded"
            assert usage is not None
            raise PersistenceFailure("cannot finalize attempt")

    calls = 0

    def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _response()

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        secret_resolver=lambda _name: "key",
        completion_fn=completion,
        attempt_hook=FailingSink(),
    )

    with pytest.raises(PersistenceFailure, match="cannot finalize"):
        provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))
    assert calls == 1
