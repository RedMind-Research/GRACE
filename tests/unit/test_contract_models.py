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

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from grace.artifacts.models import (
    ArtifactManifest,
    EvolutionResult,
    InitializationResult,
    ReconstructionReport,
    ReconstructionStatus,
    ResultStatus,
    ValidationReport,
    ValidationStatus,
)
from grace.graph.models import GraceState, GraphState
from grace.providers.base import LLMProvider, PromptRequest, ProviderCallResult, UsageRecord


def _state() -> GraceState:
    return GraceState.from_parts(
        graph=GraphState(),
        instruction="Keep the instruction stable.",
        schema_id="test-schema",
        schema_hash="schema-hash",
    )


def _usage() -> UsageRecord:
    return UsageRecord(
        model="test-model",
        input_tokens=12,
        output_tokens=5,
        latency_seconds=0.25,
    )


def test_usage_keeps_tokens_authoritative_and_cost_optional() -> None:
    usage = _usage()

    assert usage.total_tokens == 17
    assert usage.cost_usd is None
    assert usage.model_dump()["input_tokens"] == 12


def test_usage_requires_pricing_provenance_when_cost_is_saved() -> None:
    with pytest.raises(ValidationError, match="pricing_source"):
        UsageRecord(
            model="test-model",
            input_tokens=1,
            output_tokens=1,
            latency_seconds=0.1,
            cost_usd=0.01,
        )


def test_provider_protocol_is_small_and_provider_neutral() -> None:
    class FakeProvider:
        @property
        def model(self) -> str:
            return "test-model"

        def complete(self, request: PromptRequest) -> ProviderCallResult:
            return ProviderCallResult(parsed_content={"ok": True}, usage=_usage())

    provider = FakeProvider()
    result = provider.complete(PromptRequest(system_prompt="System", user_prompt="User"))

    assert isinstance(provider, LLMProvider)
    assert result.parsed_content == {"ok": True}


def test_public_evolution_result_serializes_without_sdk_objects() -> None:
    state = _state()
    validation = ValidationReport(
        status=ValidationStatus.PASSED,
        schema_valid=True,
    )
    reconstruction = ReconstructionReport(
        status=ReconstructionStatus.NO_OP,
        input_instruction_hash=state.instruction_hash,
        output_instruction_hash=state.instruction_hash,
    )
    result = EvolutionResult(
        state=state,
        change_log=(),
        validation_report=validation,
        reconstruction_report=reconstruction,
        usage=(_usage(),),
    )

    payload = result.model_dump(mode="json")
    assert payload["state"]["state_id"] == state.state_id
    assert payload["status"] == "completed"
    assert payload["usage"][0]["model"] == "test-model"


def test_public_results_cannot_claim_success_for_invalid_final_state() -> None:
    state = _state()
    invalid = ValidationReport(
        status=ValidationStatus.FAILED,
        schema_valid=False,
    )

    with pytest.raises(ValidationError, match="must be schema-valid"):
        InitializationResult(state=state, validation_report=invalid)


def test_evolution_result_binds_reconstruction_output_to_state() -> None:
    state = _state()
    validation = ValidationReport(
        status=ValidationStatus.PASSED,
        schema_valid=True,
    )
    mismatched = ReconstructionReport(
        status=ReconstructionStatus.APPLIED,
        input_instruction_hash=state.instruction_hash,
        output_instruction_hash="different-hash",
    )

    with pytest.raises(ValidationError, match="output hash"):
        EvolutionResult(
            state=state,
            change_log=(),
            validation_report=validation,
            reconstruction_report=mismatched,
        )


def test_warning_conditions_require_warning_result_status() -> None:
    state = _state()
    capped = ValidationReport(
        status=ValidationStatus.CAP_REACHED,
        schema_valid=True,
    )

    with pytest.raises(ValidationError, match="completed_with_warnings"):
        InitializationResult(state=state, validation_report=capped)

    result = InitializationResult(
        state=state,
        validation_report=capped,
        status=ResultStatus.COMPLETED_WITH_WARNINGS,
    )
    assert result.status == ResultStatus.COMPLETED_WITH_WARNINGS


def test_manifest_distinguishes_artifact_and_package_versions() -> None:
    state = _state()
    manifest = ArtifactManifest(
        artifact_format_version="1",
        grace_version="0.1.0.dev0",
        state_format_version=state.format_version,
        mode="checkpoint",
        run_id="run-test",
        state_id=state.state_id,
        parent_state_id=state.parent_state_id,
        step=state.step,
        schema_id=state.schema_id,
        schema_hash=state.schema_hash,
        instruction_hash=state.instruction_hash,
        graph_hash=state.graph_hash,
        status=ResultStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
    )

    payload = manifest.model_dump(mode="json")
    assert payload["artifact_format_version"] == "1"
    assert payload["grace_version"] == "0.1.0.dev0"
    assert payload["state_format_version"] == "1"
    assert payload["schema_file"] == "schema.json"


def test_manifest_rejects_broken_lineage_and_naive_timestamp() -> None:
    state = _state()
    fields = {
        "artifact_format_version": "1",
        "grace_version": "0.1.0.dev0",
        "state_format_version": state.format_version,
        "mode": "audit",
        "run_id": "run-test",
        "state_id": state.state_id,
        "step": 1,
        "schema_id": state.schema_id,
        "schema_hash": state.schema_hash,
        "instruction_hash": state.instruction_hash,
        "graph_hash": state.graph_hash,
        "status": ResultStatus.COMPLETED,
        "created_at": datetime.now(timezone.utc),
    }

    with pytest.raises(ValidationError, match="requires parent_state_id"):
        ArtifactManifest(**fields)

    fields["step"] = 0
    fields["created_at"] = datetime.now()
    with pytest.raises(ValidationError, match="timezone"):
        ArtifactManifest(**fields)
