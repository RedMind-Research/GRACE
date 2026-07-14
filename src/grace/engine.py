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

"""High-level, persistence-first public GRACE API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from grace.artifacts.models import (
    EvolutionResult,
    InitializationResult,
    ResultStatus,
    ValidationReport,
    ValidationStatus,
)
from grace.artifacts.provenance import hash_json, hash_text
from grace.artifacts.store import ArtifactMode, ArtifactStore
from grace.config import GraceConfig
from grace.errors import ArtifactError, SchemaValidationError
from grace.evolution.engine import advance_no_update as _advance_no_update
from grace.evolution.engine import evolve_graph
from grace.graph.models import GraceState, GraphState
from grace.graph.validation import validate_graph
from grace.initialization.p2g import initialize_graph
from grace.initialization.prompts import PromptPair
from grace.providers.base import LLMProvider
from grace.schemas.default import DefaultGraceSchema
from grace.schemas.models import NetworkSchema


def _method_config_hash(config: GraceConfig) -> str:
    """Hash only behavior-affecting configuration, not local storage paths."""

    payload = config.model_dump(
        mode="json",
        exclude={"artifact_dir", "artifact_mode"},
    )
    return hash_json(
        {
            "namespace": "grace.method-config.v1",
            "config": payload,
        }
    )


class _PromptCapture:
    """Capture exact logical-call prompts in deterministic invocation order."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, str]] = {}

    def __call__(self, stage: str, pair: PromptPair) -> None:
        index = len(self._entries) + 1
        key = f"call_{index:03d}"
        self._entries[key] = {
            "stage": stage,
            "system_prompt": pair.system,
            "user_prompt": pair.user,
        }

    @property
    def snapshot(self) -> Mapping[str, Any]:
        return self._entries


class GraceEngine:
    """Initialize and evolve a GRACE state with immutable artifacts by default.

    Provider routing and authentication stay inside the injected provider.  The
    engine is consequently usable with GRACE's LiteLLM adapter or a caller's
    own implementation of :class:`~grace.providers.base.LLMProvider`.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        schema: NetworkSchema | None = None,
        config: GraceConfig | None = None,
        artifact_dir: str | Path | None = None,
        artifact_mode: ArtifactMode | None = None,
        run_id: str | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        base_config = config or GraceConfig()
        updates: dict[str, Any] = {}
        if artifact_dir is not None:
            updates["artifact_dir"] = Path(artifact_dir)
        if artifact_mode is not None:
            updates["artifact_mode"] = artifact_mode
        config_payload = base_config.model_dump(mode="python")
        config_payload.update(updates)
        resolved_config = GraceConfig.model_validate(config_payload)

        if artifact_store is not None and (
            artifact_dir is not None or artifact_mode is not None or run_id is not None
        ):
            raise ValueError(
                "artifact_store cannot be combined with artifact_dir, artifact_mode, or run_id"
            )

        self.provider = provider
        self.schema = schema or DefaultGraceSchema()
        self.config = resolved_config
        self.artifacts = artifact_store or ArtifactStore(
            resolved_config.artifact_dir,
            mode=resolved_config.artifact_mode,
            run_id=run_id,
        )
        self._config_hash = _method_config_hash(resolved_config)

    @property
    def run_id(self) -> str:
        """Return the artifact namespace required to resume in a new process."""

        return self.artifacts.run_id

    def initialize(self, *, instruction: str) -> InitializationResult:
        """Create state zero and persist its checkpoint or audit record."""

        prompts = _PromptCapture()
        result = initialize_graph(
            instruction,
            provider=self.provider,
            schema=self.schema,
            max_rounds=self.config.p2g_max_rounds,
            max_output_tokens=self.config.max_output_tokens,
            prompt_sink=prompts,
        )
        location = self.artifacts.write_state(
            state=result.state,
            schema=self.schema,
            result_kind="initialization",
            result_payload={
                "validation_report": result.validation_report,
            },
            prompt_snapshot=prompts.snapshot,
            input_hashes={"instruction": hash_text(instruction)},
            config_hash=self._config_hash,
            models=(self.provider.model,),
            usage=result.usage,
            attempts=result.attempts,
            status=result.status,
            warnings=result.warnings,
        )
        return result.model_copy(update={"artifact_location": location})

    def initialize_from_graph(
        self,
        *,
        graph: GraphState | Mapping[str, Any],
        instruction: str,
    ) -> InitializationResult:
        """Adopt a caller-owned graph as a validated, persisted state zero.

        This is the zero-provider counterpart to :meth:`initialize`.  It is
        intended for teams that already maintain an initial graph and only
        want GRACE's Evolution component.  The graph is validated against the
        active network schema before any run artifact is created.
        """

        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        try:
            typed_graph = (
                graph if isinstance(graph, GraphState) else GraphState.model_validate(graph)
            )
        except Exception as exc:
            raise SchemaValidationError("provided graph is not a valid graph payload") from exc

        deterministic = validate_graph(typed_graph, self.schema)
        if not deterministic.valid:
            messages = "; ".join(issue.message for issue in deterministic.errors[:5])
            raise SchemaValidationError(f"provided graph is invalid: {messages}")

        warnings = tuple(issue.message for issue in deterministic.warnings)
        status = ResultStatus.COMPLETED_WITH_WARNINGS if warnings else ResultStatus.COMPLETED
        state = GraceState.from_parts(
            graph=typed_graph,
            instruction=instruction,
            schema_id=self.schema.id,
            schema_hash=self.schema.schema_hash,
            step=0,
        )
        validation_report = ValidationReport(
            status=ValidationStatus.PASSED,
            schema_valid=True,
            fidelity_valid=None,
            warnings=warnings,
            history=(
                {
                    "stage": "provided_graph_validation",
                    "node_count": len(typed_graph.nodes),
                    "edge_count": len(typed_graph.edges),
                },
            ),
        )
        result = InitializationResult(
            state=state,
            validation_report=validation_report,
            status=status,
            warnings=warnings,
        )
        location = self.artifacts.write_state(
            state=state,
            schema=self.schema,
            result_kind="initialization",
            result_payload={"validation_report": validation_report},
            prompt_snapshot=None,
            input_hashes={
                "instruction": state.instruction_hash,
                "provided_graph": state.graph_hash,
            },
            config_hash=self._config_hash,
            models=(),
            usage=(),
            attempts=(),
            status=status,
            warnings=warnings,
        )
        return result.model_copy(update={"artifact_location": location})

    def evolve(
        self,
        *,
        state: GraceState,
        diagnosis_report: str,
    ) -> EvolutionResult:
        """Apply one diagnosis-driven update and persist the accepted child state."""

        self._require_persisted_parent(state)
        prompts = _PromptCapture()
        result = evolve_graph(
            state,
            diagnosis_report,
            provider=self.provider,
            schema=self.schema,
            config=self.config,
            prompt_sink=prompts,
        )
        location = self.artifacts.write_state(
            state=result.state,
            schema=self.schema,
            result_kind="evolution",
            result_payload={
                "change_log": result.change_log,
                "validation_report": result.validation_report,
                "reconstruction_report": result.reconstruction_report,
                "provenance": result.provenance,
            },
            prompt_snapshot=prompts.snapshot,
            input_hashes={
                "parent_state": state.state_id,
                "diagnosis_report": hash_text(diagnosis_report),
            },
            config_hash=self._config_hash,
            models=(self.provider.model,),
            usage=result.usage,
            attempts=result.attempts,
            status=result.status,
            warnings=result.warnings,
        )
        return result.model_copy(update={"artifact_location": location})

    def advance_no_update(
        self,
        *,
        state: GraceState,
        diagnosis_report: str | None = None,
    ) -> EvolutionResult:
        """Persist an explicit zero-provider child for a fixed checkpoint sequence."""

        if diagnosis_report is not None and not diagnosis_report.strip():
            raise ValueError("diagnosis_report must not be empty when supplied")
        self._require_persisted_parent(state)
        result = _advance_no_update(state, schema=self.schema)
        input_hashes = {"parent_state": state.state_id}
        if diagnosis_report is not None:
            diagnosis_hash = hash_text(diagnosis_report)
            input_hashes["diagnosis_report"] = diagnosis_hash
            result = result.model_copy(
                update={
                    "provenance": {
                        **result.provenance,
                        "diagnosis_sha256": diagnosis_hash,
                    }
                }
            )
        location = self.artifacts.write_state(
            state=result.state,
            schema=self.schema,
            result_kind="evolution",
            result_payload={
                "change_log": result.change_log,
                "validation_report": result.validation_report,
                "reconstruction_report": result.reconstruction_report,
                "provenance": result.provenance,
            },
            prompt_snapshot=None,
            input_hashes=input_hashes,
            config_hash=self._config_hash,
            models=(),
            usage=(),
            attempts=(),
            status=result.status,
            warnings=result.warnings,
        )
        return result.model_copy(update={"artifact_location": location})

    def _require_persisted_parent(self, state: GraceState) -> None:
        """Fail before provider use when a persistent run cannot prove lineage."""

        if self.artifacts.mode == "none":
            return
        path = self.artifacts.find_state(state.state_id)
        if path is None:
            raise ArtifactError(
                "input state is not present in this artifact run; initialize it in this "
                "run or resume with the originating run_id before evolving"
            )
        persisted = self.artifacts.load_state(path, expected_schema=self.schema)
        if persisted.canonical_json() != state.canonical_json():
            raise ArtifactError("input state differs from its persisted checkpoint")


__all__ = ["GraceEngine"]
