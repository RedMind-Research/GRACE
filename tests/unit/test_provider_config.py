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

"""Provider route configuration is explicit and safe to serialize."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import TypeAdapter, ValidationError

from grace.providers.base import PromptRequest, make_logical_call_id
from grace.providers.config import (
    GoogleAIStudioConfig,
    ProviderDescriptor,
    ProviderRouteConfig,
    ProviderRuntimeConfig,
    VertexAIADCConfig,
)


def test_descriptor_accepts_a_custom_provider_route() -> None:
    descriptor = ProviderDescriptor(
        route="internal_gateway",
        model_id="agent-model-v7",
        model="internal_gateway/agent-model-v7",
    )

    assert descriptor.route == "internal_gateway"


def test_descriptor_rejects_a_blank_provider_route() -> None:
    with pytest.raises(ValidationError, match="blank"):
        ProviderDescriptor(route="  ", model_id="model", model="custom/model")


def test_vertex_route_qualifies_model_and_hashes_project() -> None:
    config = VertexAIADCConfig(project="private-project", location="us-central1")

    assert config.model_id == "gemini-2.5-flash"
    assert config.litellm_model == "vertex_ai/gemini-2.5-flash"
    descriptor = config.safe_descriptor()
    assert descriptor.route == "vertex_ai_adc"
    assert descriptor.location == "us-central1"
    assert descriptor.project_hash == hashlib.sha256(b"private-project").hexdigest()
    assert "private-project" not in descriptor.model_dump_json()


def test_ai_studio_route_is_provider_qualified_without_a_secret() -> None:
    config = GoogleAIStudioConfig(api_key_env="MY_GEMINI_KEY")

    assert config.litellm_model == "gemini/gemini-2.5-flash"
    assert config.safe_descriptor().model_dump() == {
        "route": "google_ai_studio",
        "model_id": "gemini-2.5-flash",
        "model": "gemini/gemini-2.5-flash",
        "location": None,
        "project_hash": None,
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: VertexAIADCConfig(
            project="project", location="us-central1", model_id="vertex_ai/gemini"
        ),
        lambda: GoogleAIStudioConfig(model_id="gemini/gemini-2.5-flash"),
    ],
)
def test_semantic_model_id_rejects_provider_prefix(factory: object) -> None:
    with pytest.raises(ValidationError, match="provider-neutral"):
        factory()  # type: ignore[operator]


def test_route_union_requires_matching_fields() -> None:
    adapter = TypeAdapter(ProviderRouteConfig)

    route = adapter.validate_python(
        {
            "route": "vertex_ai_adc",
            "project": "project",
            "location": "us-central1",
        }
    )
    assert isinstance(route, VertexAIADCConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({"route": "vertex_ai_adc", "api_key_env": "KEY"})


def test_runtime_defaults_encode_the_release_retry_envelope() -> None:
    runtime = ProviderRuntimeConfig()

    assert runtime.max_attempts == 3
    assert runtime.attempt_timeout_seconds == 300.0
    assert runtime.overall_timeout_seconds == 900.0
    assert runtime.structured_response_retry_temperature == 0.5


def test_runtime_rejects_an_impossible_timeout_envelope() -> None:
    with pytest.raises(ValidationError, match="cover at least one attempt"):
        ProviderRuntimeConfig(attempt_timeout_seconds=10, overall_timeout_seconds=9)


def test_logical_call_id_is_optional_but_must_be_a_full_lowercase_sha256() -> None:
    request = PromptRequest(
        system_prompt="system",
        user_prompt="user",
        logical_call_id="a" * 64,
    )
    assert request.logical_call_id == "a" * 64

    with pytest.raises(ValidationError, match="logical_call_id"):
        PromptRequest(
            system_prompt="system",
            user_prompt="user",
            logical_call_id="A" * 64,
        )


def test_logical_call_id_helper_binds_exact_prompts_model_stage_and_binding() -> None:
    values = {
        "stage": "p2g_initial",
        "system_prompt": "system",
        "user_prompt": "user",
        "model": "vertex_ai/gemini-2.5-flash",
        "binding_id": "state-001",
    }
    call_id = make_logical_call_id(**values)

    assert len(call_id) == 64
    assert call_id == make_logical_call_id(**values)
    assert call_id != make_logical_call_id(**(values | {"user_prompt": "user\n"}))
    assert call_id != make_logical_call_id(**(values | {"model": "gemini/gemini-2.5-flash"}))
