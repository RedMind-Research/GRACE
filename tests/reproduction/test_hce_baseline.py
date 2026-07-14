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

from __future__ import annotations

from grace.artifacts.provenance import hash_text
from grace.providers.base import PromptRequest, ProviderCallResult, UsageRecord
from reproduction.tau2_telecom.baselines.hce import (
    HCE_SYSTEM,
    HCE_USER,
    evolve_hce,
)


class TextProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[PromptRequest] = []

    @property
    def model(self) -> str:
        return "scripted/hce"

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        self.requests.append(request)
        return ProviderCallResult(
            raw_content=self.text,
            usage=UsageRecord(
                model=self.model,
                input_tokens=10,
                output_tokens=4,
                latency_seconds=0.01,
            ),
        )


def test_frozen_hce_prompt_hashes_match_release_contract() -> None:
    assert hash_text(HCE_SYSTEM) == (
        "fa3424ad6459caa8ef2f4ca2b7186db34b882eed9201ca1d85d7bdc849ec2326"
    )
    assert hash_text(HCE_USER) == (
        "e1ed464b8b256eac031c68aea1e8850ba17542449ad0998b588dcb441cc4e9da"
    )


def test_hce_is_one_provider_call_with_typed_provenance() -> None:
    provider = TextProvider("```xml\nKeep the policy.\nAdd account verification.\n```")
    result = evolve_hce(
        current_prompt="Keep the policy.",
        diagnosis_report="Account verification was skipped.",
        provider=provider,
    )

    assert result.evolved_prompt == "Keep the policy.\nAdd account verification."
    assert result.output_prompt_hash == hash_text(result.evolved_prompt)
    assert result.length_delta == result.output_prompt_length - result.input_prompt_length
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.stage == "hce_evolution"
    assert request.logical_call_id is not None
    assert request.expect_json is False
    assert request.temperature == 0.0
    assert request.max_tokens == 65536
