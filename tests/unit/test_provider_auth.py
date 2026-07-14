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

"""Authentication discovery must never cross provider-route boundaries."""

from __future__ import annotations

import pytest

from grace.errors import ProviderAuthenticationError
from grace.providers.auth import GOOGLE_CLOUD_SCOPE, discover_auth, resolve_ai_studio_api_key
from grace.providers.config import GoogleAIStudioConfig, VertexAIADCConfig


def test_ai_studio_resolves_exactly_the_configured_secret_name() -> None:
    requested: list[str] = []

    def resolver(name: str) -> str | None:
        requested.append(name)
        return {"SELECTED_KEY": "secret-value", "GOOGLE_API_KEY": "wrong"}.get(name)

    config = GoogleAIStudioConfig(api_key_env="SELECTED_KEY")

    assert resolve_ai_studio_api_key(config, resolver) == "secret-value"
    assert requested == ["SELECTED_KEY"]


def test_ai_studio_has_no_google_api_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-be-used")

    with pytest.raises(ProviderAuthenticationError, match="configured AI Studio"):
        resolve_ai_studio_api_key(GoogleAIStudioConfig())


def test_secret_resolver_failure_does_not_expose_exception_text() -> None:
    def resolver(_name: str) -> str | None:
        raise RuntimeError("credential=super-secret")

    with pytest.raises(ProviderAuthenticationError) as captured:
        resolve_ai_studio_api_key(GoogleAIStudioConfig(), resolver)

    assert "super-secret" not in str(captured.value)


def test_ai_studio_discovery_never_calls_adc() -> None:
    def forbidden_adc(**_kwargs: object) -> tuple[object, str | None]:
        raise AssertionError("AI Studio route called ADC")

    readiness = discover_auth(
        GoogleAIStudioConfig(),
        secret_resolver=lambda _name: "key",
        adc_loader=forbidden_adc,
    )

    assert readiness.credential_kind == "api_key"
    assert readiness.live_verified is False


def test_vertex_discovery_uses_cloud_scope_and_never_resolves_api_key() -> None:
    calls: list[tuple[str, ...]] = []

    def loader(*, scopes: tuple[str, ...]) -> tuple[object, str | None]:
        calls.append(scopes)
        return object(), "ambient-project-is-ignored"

    def forbidden_resolver(_name: str) -> str | None:
        raise AssertionError("Vertex route resolved an API key")

    readiness = discover_auth(
        VertexAIADCConfig(project="explicit-project", location="us-central1"),
        secret_resolver=forbidden_resolver,
        adc_loader=loader,
    )

    assert calls == [(GOOGLE_CLOUD_SCOPE,)]
    assert readiness.credential_kind == "adc"
    assert "explicit-project" not in readiness.model_dump_json()


def test_vertex_discovery_failure_is_safe() -> None:
    def loader(**_kwargs: object) -> tuple[object, str | None]:
        raise RuntimeError("/home/user/private-service-account.json")

    with pytest.raises(ProviderAuthenticationError) as captured:
        discover_auth(
            VertexAIADCConfig(project="project", location="us-central1"),
            adc_loader=loader,
        )

    assert "private-service-account" not in str(captured.value)
