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

"""Credential discovery without implicit dotenv loading or provider calls."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from grace.errors import ProviderAuthenticationError, ProviderConfigurationError
from grace.providers.config import (
    GoogleAIStudioConfig,
    ProviderDescriptor,
    ProviderRouteConfig,
    VertexAIADCConfig,
)


GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class SecretResolver(Protocol):
    """Resolve a named secret at runtime without making it serializable."""

    def __call__(self, name: str) -> str | None: ...


ADCLoader = Callable[..., tuple[Any, str | None]]


class AuthReadiness(BaseModel):
    """Safe local credential-discovery result; not a live authorization claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["credentials_discovered"] = "credentials_discovered"
    descriptor: ProviderDescriptor
    credential_kind: Literal["adc", "api_key"]
    live_verified: bool = False
    warnings: tuple[str, ...] = ()


def environment_secret_resolver(name: str) -> str | None:
    """Read exactly ``name`` from the existing process environment."""

    return os.environ.get(name)


def resolve_ai_studio_api_key(
    config: GoogleAIStudioConfig,
    resolver: SecretResolver | None = None,
) -> str:
    """Resolve only the explicitly configured key reference.

    There is deliberately no fallback to ``GOOGLE_API_KEY`` or any other
    environment variable, even when one exists.
    """

    resolver = resolver or environment_secret_resolver
    try:
        value = resolver(config.api_key_env)
    except Exception:
        raise ProviderAuthenticationError("AI Studio credential resolver failed") from None
    if value is None or not value.strip():
        raise ProviderAuthenticationError("configured AI Studio credential is missing or blank")
    return value


def _default_adc_loader(*, scopes: tuple[str, ...]) -> tuple[Any, str | None]:
    try:
        import google.auth
    except ImportError:
        raise ProviderConfigurationError(
            "Vertex ADC support is not installed; install 'redmind-grace[gemini]'"
        ) from None
    return google.auth.default(scopes=scopes)


def discover_auth(
    config: ProviderRouteConfig,
    *,
    secret_resolver: SecretResolver | None = None,
    adc_loader: ADCLoader | None = None,
) -> AuthReadiness:
    """Discover configured credentials without making a model request.

    Discovery proves only that a credential source exists.  IAM permissions,
    API enablement, billing, quota, and model access require a protected live
    provider canary.
    """

    if isinstance(config, GoogleAIStudioConfig):
        resolve_ai_studio_api_key(config, secret_resolver)
        return AuthReadiness(
            descriptor=config.safe_descriptor(),
            credential_kind="api_key",
        )

    if not isinstance(config, VertexAIADCConfig):
        raise ProviderConfigurationError("unsupported provider route")

    loader = adc_loader or _default_adc_loader
    try:
        credentials, _discovered_project = loader(scopes=(GOOGLE_CLOUD_SCOPE,))
    except ProviderConfigurationError:
        raise
    except Exception:
        raise ProviderAuthenticationError("Vertex ADC credentials were not discovered") from None
    if credentials is None:
        raise ProviderAuthenticationError("Vertex ADC credentials were not discovered")
    return AuthReadiness(
        descriptor=config.safe_descriptor(),
        credential_kind="adc",
    )


__all__ = [
    "ADCLoader",
    "AuthReadiness",
    "GOOGLE_CLOUD_SCOPE",
    "SecretResolver",
    "discover_auth",
    "environment_secret_resolver",
    "resolve_ai_studio_api_key",
]
