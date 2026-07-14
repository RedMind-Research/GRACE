# Providers

GRACE is provider-neutral at the engine boundary. The first-party Gemini adapter is the default documented live integration and supports two explicit routes, while teams using another model or an internal inference platform can inject any object that implements the two-member [`LLMProvider` contract](#custom-providers). GRACE does not ship Anthropic, OpenAI, or other named adapters in this release.

## First-party Gemini adapter

Install the Gemini extra when using the first-party adapter:

```bash
python -m pip install 'redmind-grace[gemini]'
```

The adapter calls LiteLLM as an in-process library; it does not require a separate LiteLLM proxy server. The two supported routes never silently fall back to one another.

| Route | Model string | Authentication |
| --- | --- | --- |
| Vertex AI | `vertex_ai/gemini-2.5-flash` | ADC plus explicit project and location |
| Google AI Studio | `gemini/gemini-2.5-flash` | Explicit API-key reference |

### Vertex AI

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
export GRACE_VERTEX_PROJECT=YOUR_PROJECT_ID
export GRACE_VERTEX_LOCATION=us-central1
```

```python
import os

from grace.providers import LiteLLMProvider, VertexAIADCConfig

provider = LiteLLMProvider(
    VertexAIADCConfig(
        project=os.environ["GRACE_VERTEX_PROJECT"],
        location=os.getenv("GRACE_VERTEX_LOCATION", "us-central1"),
    )
)
provider.validate_auth()  # Credential discovery only; no model call.
```

Vertex AI uses Application Default Credentials and does not require `GEMINI_API_KEY`. GRACE passes the configured project and location on every request.

### Google AI Studio

```bash
export GEMINI_API_KEY=YOUR_KEY
```

```python
from grace.providers import GoogleAIStudioConfig, LiteLLMProvider

provider = LiteLLMProvider(
    GoogleAIStudioConfig(api_key_env="GEMINI_API_KEY")
)
```

Applications may inject a secret resolver instead of reading an environment variable. Only the configured reference is resolved.

## Output-token limit

GRACE forwards `GraceConfig.max_output_tokens` unchanged on every P2G and Evolution model request. The public default is `65536`, matching the released experiment configuration; integrations can select a lower positive integer for a model with a smaller output limit.

```python
from grace import GraceConfig, GraceEngine

engine = GraceEngine(
    provider=provider,
    config=GraceConfig(max_output_tokens=8192),
)
```

The engine does not silently clamp this value or switch models when a route rejects it. A route-specific incompatibility must fail explicitly so that the caller can change the configuration and start or resume under intentional provenance.

## Reliability boundary

The first-party adapter owns a bounded retry and timeout policy while disabling LiteLLM retries, preventing retry multiplication. Every successful logical call returns typed usage and attempt records; unavailable usage remains explicitly missing rather than becoming a fabricated zero.

Provider-facing JSON Schemas are normalized only for Gemini's supported structured-output subset. The original stricter JSON Schema is always enforced locally after parsing, so transport compatibility never weakens GRACE validation.

Importing `grace` does not import LiteLLM, resolve credentials, or load `.env`. Credentials are resolved only when the configured Gemini provider is used.

### Host processes that already use LiteLLM

Set `LITELLM_MODE=PRODUCTION` before any library imports LiteLLM. GRACE imports LiteLLM lazily in production mode to prevent LiteLLM's development-mode dotenv loading from changing the credential boundary. If a host process has already imported LiteLLM outside production mode, the first-party adapter fails closed and asks for a fresh process; it does not mutate an already-loaded global module. Hosts that intentionally own a different LiteLLM lifecycle should connect through the [custom provider contract](#custom-providers).

## Model lifecycle and provenance

For Vertex AI, Google currently lists October 16, 2026 as the retirement date for `gemini-2.5-flash` in its official [model versions and lifecycle documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions); Google notes that the timeline may be extended but not moved earlier. For Google AI Studio, the official [Gemini API deprecation schedule](https://ai.google.dev/gemini-api/docs/deprecations) lists the same date as the earliest shutdown date and states that the exact date will be communicated in advance. Verify the route-specific page before a new run.

Treat any model or route migration as a deliberate experiment change. `LLMProvider.model` participates in logical-call identity and recorded model provenance, and the reproduction layer binds model routes into its run fingerprint. Do not add a silent fallback: configure the replacement explicitly, run qualification, and create artifacts under the new provenance.

## Custom providers

Use a custom provider when your organization already has an inference gateway, needs a model other than the first-party Gemini routes, or must enforce its own authentication and reliability controls. This extension point does not imply that GRACE ships a first-party adapter for any particular external provider.

Install the base distribution; the custom-provider path does not require LiteLLM:

```bash
python -m pip install redmind-grace
```

### Two-member protocol

The `LLMProvider` protocol requires two members; an implementation may expose additional application-specific methods:

```python
from grace.providers import PromptRequest, ProviderCallResult


class LLMProvider:
    @property
    def model(self) -> str: ...

    def complete(self, request: PromptRequest) -> ProviderCallResult: ...
```

`model` must be a stable, provider-qualified route identifier such as `internal_gateway/model-version`. GRACE records it in artifact provenance and includes it in logical-call identity, so changing the returned value is an intentional provenance change rather than an implementation detail.

`complete` is synchronous. It must return a valid `ProviderCallResult` or raise a typed public provider error; the engine does not wrap this method in another retry or timeout loop.

### Request contract

`PromptRequest` is frozen and rejects unknown fields. A provider should preserve these semantics:

| Field | Adapter responsibility |
| --- | --- |
| `system_prompt`, `user_prompt` | Deliver the two prompt roles without rewriting their contents. |
| `stage` | Preserve for safe telemetry and error attribution. |
| `logical_call_id` | Preserve in adapter-owned telemetry or attempt-hook context; do not recompute it with a different model identifier. |
| `expect_json` | When `True`, parse the response as JSON and return it through `parsed_content`. Current GRACE stages require a JSON object, not a scalar or array. |
| `response_schema` | When present, validate the schema itself before dispatch, then validate the parsed JSON against the original schema before returning. Provider-specific schema simplification may be used for transport only; it must not weaken local validation. |
| `temperature` | Use the requested sampling temperature for the first attempt. If a documented retry policy changes it, record the actual value in each attempt record. |
| `max_tokens` | Forward as the output-token ceiling. Do not silently clamp it; reject unsupported values explicitly. |

`response_schema` is only valid when `expect_json=True`. An invalid caller-owned schema should raise `ProviderConfigurationError` before dispatch. A malformed or empty provider response or a response-schema mismatch should raise `ProviderResponseError`; authentication, request configuration, and exhausted transport failures should use the corresponding public error class from `grace.errors`.

### Result and usage invariants

Constructing `ProviderCallResult` and `UsageRecord` validates the public accounting contract:

- A result must contain `parsed_content` or `raw_content`.
- If attempt records are supplied, their count must equal `usage.attempts`, and the final attempt must have outcome `succeeded`.
- `usage.attempts` must equal `usage.retries + 1`.
- `usage_source="provider"` or `"estimated"` requires both input and output token counts.
- `usage_source="missing"` requires every token-count field to remain `None`; never replace unavailable usage with fabricated zeroes.
- Recording `cost_usd` also requires `pricing_source` and `pricing_version`.

Use the same stable route identifier for `LLMProvider.model` and `UsageRecord.model`. If the upstream service returns a more specific resolved model, preserve it separately as `provider_response_model`.

### Retry and timeout ownership

The adapter owns one bounded reliability envelope for each `complete` call: per-attempt timeout, overall timeout, retry classification, maximum attempts, and backoff. Disable retries in the underlying SDK or gateway client when the adapter retries, otherwise nested retry policies can multiply calls and invalidate accounting.

Return one `ProviderAttemptRecord` for every actual dispatch when production telemetry is available, including failed attempts that preceded success. If all attempts fail, raise a `ProviderError` subclass with the completed attempt history instead of returning a partial success. Credentials, raw prompts, and response bodies must not appear in attempt records or exception messages.

### Minimal executable skeleton

The following adapter is a zero-network contract smoke test. It intentionally performs one attempt and reports missing token usage; replace `transport` with your bounded production transport and add attempt records before deployment.

```python
import json
from collections.abc import Callable
from time import monotonic

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError

from grace.errors import (
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTransportError,
)
from grace.providers import PromptRequest, ProviderCallResult, UsageRecord

Transport = Callable[[PromptRequest], str]


class CustomProvider:
    def __init__(self, *, model: str, transport: Transport) -> None:
        self._model = model
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        validator = None
        if request.response_schema is not None:
            try:
                Draft202012Validator.check_schema(request.response_schema)
                validator = Draft202012Validator(request.response_schema)
            except JSONSchemaError:
                raise ProviderConfigurationError(
                    "response_schema is not valid JSON Schema"
                ) from None

        started = monotonic()
        try:
            raw = self._transport(request)
        except ProviderError:
            raise
        except Exception:
            raise ProviderTransportError("custom provider transport failed") from None

        if not isinstance(raw, str) or not raw.strip():
            raise ProviderResponseError("custom provider returned an empty response")

        parsed = None
        if request.expect_json:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                raise ProviderResponseError("custom provider returned invalid JSON") from None
            if not isinstance(parsed, dict):
                raise ProviderResponseError("GRACE requires a JSON object response")
            if validator is not None:
                try:
                    validator.validate(parsed)
                except JSONSchemaValidationError:
                    raise ProviderResponseError(
                        "custom provider response did not match response_schema"
                    ) from None

        return ProviderCallResult(
            parsed_content=parsed,
            raw_content=raw,
            usage=UsageRecord(
                model=self.model,
                route="custom",
                usage_source="missing",
                latency_seconds=monotonic() - started,
            ),
        )


provider = CustomProvider(
    model="custom/scripted-v1",
    transport=lambda request: '{"operations": []}',
)
result = provider.complete(
    PromptRequest(
        system_prompt="Return a JSON object.",
        user_prompt="Return an empty operation list.",
        stage="contract_smoke",
        expect_json=True,
        response_schema={
            "type": "object",
            "properties": {"operations": {"type": "array"}},
            "required": ["operations"],
            "additionalProperties": False,
        },
        max_tokens=256,
    )
)
assert result.parsed_content == {"operations": []}
```

Run the repository's complete P2G-to-Evolution example with a scripted provider and no network access:

```bash
python examples/offline_quickstart.py --artifact-dir /tmp/grace-provider-contract
```

Before using a production adapter, test malformed and empty responses, an invalid caller-owned schema, response-schema mismatch, authentication failure, retry exhaustion, timeout exhaustion, missing usage, preservation of typed provider errors, and the exact forwarding of `temperature` and `max_tokens`. The executable [`examples/offline_quickstart.py`](../examples/offline_quickstart.py) demonstrates dependency injection through the public `GraceEngine` API.
