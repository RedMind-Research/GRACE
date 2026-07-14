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

"""Serializable configuration for the public LiteLLM provider adapter.

The semantic model identifier is deliberately kept separate from LiteLLM's
provider-qualified route.  This prevents ambient credentials or LiteLLM
globals from silently selecting a different provider.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ProviderRoute: TypeAlias = str


def _nonempty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be blank")
    return stripped


def _semantic_model_id(value: str) -> str:
    value = _nonempty(value)
    if "/" in value:
        raise ValueError("model_id must be provider-neutral and must not contain '/'")
    return value


class ProviderDescriptor(BaseModel):
    """Credential-free provider provenance safe for reports and artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: ProviderRoute = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    location: str | None = None
    project_hash: str | None = None

    _validate_route = field_validator("route")(_nonempty)


class VertexAIADCConfig(BaseModel):
    """Vertex AI route using Google Application Default Credentials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: Literal["vertex_ai_adc"] = "vertex_ai_adc"
    model_id: str = "gemini-2.5-flash"
    project: str = Field(min_length=1, repr=False)
    location: str = Field(min_length=1)

    _validate_model_id = field_validator("model_id")(_semantic_model_id)
    _validate_project = field_validator("project")(_nonempty)
    _validate_location = field_validator("location")(_nonempty)

    @property
    def litellm_model(self) -> str:
        return f"vertex_ai/{self.model_id}"

    def safe_descriptor(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            route=self.route,
            model_id=self.model_id,
            model=self.litellm_model,
            location=self.location,
            project_hash=hashlib.sha256(self.project.encode("utf-8")).hexdigest(),
        )


class GoogleAIStudioConfig(BaseModel):
    """Google AI Studio route with a caller-selected secret reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: Literal["google_ai_studio"] = "google_ai_studio"
    model_id: str = "gemini-2.5-flash"
    api_key_env: str = Field(default="GEMINI_API_KEY", min_length=1)

    _validate_model_id = field_validator("model_id")(_semantic_model_id)
    _validate_api_key_env = field_validator("api_key_env")(_nonempty)

    @property
    def litellm_model(self) -> str:
        return f"gemini/{self.model_id}"

    def safe_descriptor(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            route=self.route,
            model_id=self.model_id,
            model=self.litellm_model,
        )


ProviderRouteConfig: TypeAlias = Annotated[
    VertexAIADCConfig | GoogleAIStudioConfig,
    Field(discriminator="route"),
]


class ProviderRuntimeConfig(BaseModel):
    """Bounded reliability policy owned by the GRACE adapter.

    ``max_attempts`` is the total number of dispatches, including the initial
    attempt.  LiteLLM's own retries are disabled by the adapter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=3, ge=1)
    attempt_timeout_seconds: float = Field(default=300.0, gt=0.0)
    overall_timeout_seconds: float = Field(default=900.0, gt=0.0)
    initial_backoff_seconds: float = Field(default=1.0, ge=0.0)
    max_backoff_seconds: float = Field(default=60.0, ge=0.0)
    jitter_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    structured_response_retry_temperature: float = Field(default=0.5, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ProviderRuntimeConfig":
        if self.overall_timeout_seconds < self.attempt_timeout_seconds:
            raise ValueError("overall_timeout_seconds must cover at least one attempt")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        return self


__all__ = [
    "GoogleAIStudioConfig",
    "ProviderDescriptor",
    "ProviderRoute",
    "ProviderRouteConfig",
    "ProviderRuntimeConfig",
    "VertexAIADCConfig",
]
