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

"""LLM-provider contracts and implementations."""

from grace.providers.auth import AuthReadiness, discover_auth
from grace.providers.base import (
    LLMProvider,
    PromptRequest,
    Provider,
    ProviderAttemptContext,
    ProviderAttemptHook,
    ProviderAttemptRecord,
    ProviderCallResult,
    ProviderProtocol,
    UsageRecord,
    make_logical_call_id,
)
from grace.providers.config import (
    GoogleAIStudioConfig,
    ProviderDescriptor,
    ProviderRuntimeConfig,
    VertexAIADCConfig,
)
from grace.providers.litellm import LiteLLMProvider, ensure_litellm_production

__all__ = [
    "AuthReadiness",
    "GoogleAIStudioConfig",
    "LLMProvider",
    "LiteLLMProvider",
    "make_logical_call_id",
    "PromptRequest",
    "Provider",
    "ProviderAttemptContext",
    "ProviderAttemptHook",
    "ProviderAttemptRecord",
    "ProviderCallResult",
    "ProviderDescriptor",
    "ProviderProtocol",
    "ProviderRuntimeConfig",
    "UsageRecord",
    "VertexAIADCConfig",
    "discover_auth",
    "ensure_litellm_production",
]
