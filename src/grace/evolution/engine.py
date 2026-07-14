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

"""One provider-injected GRACE Evolution step."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Callable, Iterable, cast

from pydantic import JsonValue

from grace.artifacts.models import (
    EvolutionResult,
    ResultStatus,
    ValidationReport,
    ValidationStatus,
)
from grace.config import GraceConfig
from grace.errors import ProviderError, SchemaValidationError, StateIntegrityError
from grace.evolution.prompts import (
    operation_planning,
    patch_reconstruction,
    relation_maintenance,
    structural_analysis,
    structural_repair,
)
from grace.evolution.reconstruction import apply_patch_operations
from grace.graph.diff import diff_graphs
from grace.graph.grounding import apply_grounding_policy
from grace.graph.models import GraceState, GraphState
from grace.graph.neighborhood import local_subgraph
from grace.graph.operations import assemble
from grace.graph.repair import repair_graph
from grace.graph.validation import validate_graph
from grace.initialization.prompts import PromptPair
from grace.providers.base import (
    LLMProvider,
    PromptRequest,
    ProviderAttemptRecord,
    UsageRecord,
    make_logical_call_id,
)
from grace.schemas.models import NetworkSchema


PromptSink = Callable[[str, PromptPair], None]


_STRING: dict[str, JsonValue] = {"type": "string", "minLength": 1}
_OPERATION_FIELDS: dict[str, tuple[str, ...]] = {
    "AddNode": ("type", "content"),
    "ModifyNode": ("node_id", "new_content"),
    "RemoveNode": ("node_id",),
    "AddEdge": ("source", "target", "relation"),
    "RemoveEdge": ("source", "target", "relation"),
    "Merge": ("u_id", "v_id", "new_content"),
    "ReplaceSpan": ("anchor", "new_text"),
    "InsertAfter": ("anchor", "new_text"),
    "DeleteSpan": ("anchor",),
}


def _operation_variant(name: str) -> dict[str, JsonValue]:
    fields = _OPERATION_FIELDS[name]
    properties: dict[str, JsonValue] = {
        "op": {"type": "string", "enum": [name]},
        **{field: dict(_STRING) for field in fields},
    }
    if name == "AddNode":
        properties["attach_edges"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": dict(_STRING),
                    "target": dict(_STRING),
                    "relation": dict(_STRING),
                },
                "required": ["source", "target", "relation"],
                "additionalProperties": False,
            },
        }
    return {
        "type": "object",
        "properties": properties,
        "required": ["op", *fields],
        "additionalProperties": False,
    }


def _operations_response_schema(allowed: Iterable[str]) -> dict[str, JsonValue]:
    """Require the public ``op`` discriminator at the provider boundary.

    Operation-specific fields remain validated by the deterministic typed
    assembler.  The transport schema prevents a model from silently emitting
    legacy discriminators such as ``operation`` and lets the provider adapter
    retry malformed structured output before Evolution sees it.
    """

    variants = cast(
        list[JsonValue],
        [_operation_variant(name) for name in sorted(set(allowed))],
    )
    return {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {"anyOf": variants},
            }
        },
        "required": ["operations"],
        "additionalProperties": False,
    }


_STRUCTURAL_ANALYSIS_RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "contradictions": {"type": "array"},
        "redundancies": {"type": "array"},
    },
    "required": ["contradictions", "redundancies"],
    "additionalProperties": False,
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> JsonValue:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str))


def _json_object(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    normalized = _json_value(dict(value))
    if not isinstance(normalized, dict):  # pragma: no cover - dict input guarantees this
        raise TypeError("JSON object normalization returned a non-object")
    return normalized


def _prompt_record(stage: str, pair: PromptPair) -> dict[str, JsonValue]:
    return {
        "stage": stage,
        "system_prompt_sha256": _sha256(pair.system),
        "user_prompt_sha256": _sha256(pair.user),
        "system_prompt_chars": len(pair.system),
        "user_prompt_chars": len(pair.user),
    }


def _complete_json(
    provider: LLMProvider,
    pair: PromptPair,
    *,
    stage: str,
    usage: list[UsageRecord],
    attempts: list[ProviderAttemptRecord],
    history: list[dict[str, JsonValue]],
    prompt_sink: PromptSink | None,
    binding_id: str,
    response_schema: dict[str, JsonValue],
    max_output_tokens: int,
) -> Mapping[str, Any]:
    if prompt_sink is not None:
        prompt_sink(stage, pair)
    result = provider.complete(
        PromptRequest(
            system_prompt=pair.system,
            user_prompt=pair.user,
            stage=stage,
            logical_call_id=make_logical_call_id(
                stage=stage,
                system_prompt=pair.system,
                user_prompt=pair.user,
                model=provider.model,
                binding_id=binding_id,
            ),
            expect_json=True,
            response_schema=response_schema,
            temperature=0.0,
            max_tokens=max_output_tokens,
        )
    )
    usage.append(result.usage)
    attempts.extend(result.attempts)
    history.append(_prompt_record(stage, pair))
    if not isinstance(result.parsed_content, Mapping):
        raise ProviderError(f"{stage} must return a JSON object")
    return result.parsed_content


def _operations(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = value.get("operations", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _partition_operations(
    operations: Iterable[Mapping[str, Any]],
    allowed: set[str],
    *,
    stage: str,
) -> tuple[list[Mapping[str, Any]], list[dict[str, JsonValue]]]:
    accepted: list[Mapping[str, Any]] = []
    rejected: list[dict[str, JsonValue]] = []
    for index, operation in enumerate(operations):
        name = operation.get("op")
        if name in allowed:
            accepted.append(operation)
        else:
            rejected.append(
                {
                    "stage": stage,
                    "operation_index": index,
                    "code": "operation_not_allowed_in_stage",
                    "operation": _json_value(operation),
                }
            )
    return accepted, rejected


def _validate_or_repair(
    graph: GraphState,
    schema: NetworkSchema,
    *,
    max_rounds: int,
) -> tuple[GraphState, list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    history: list[dict[str, JsonValue]] = []
    drops: list[dict[str, JsonValue]] = []
    candidate = graph
    for round_index in range(max_rounds + 1):
        report = validate_graph(candidate, schema)
        history.append(
            {
                "round": round_index,
                "schema_valid": report.valid,
                "errors": _json_value([error.model_dump(mode="json") for error in report.errors]),
            }
        )
        if report.valid:
            return candidate, history, drops
        if round_index == max_rounds:
            break
        repaired = repair_graph(candidate, schema)
        candidate = repaired.graph
        drops.extend(drop.model_dump(mode="json") for drop in repaired.forced_drops)
    messages = "; ".join(error.message for error in report.errors[:5])
    raise SchemaValidationError(f"graph cannot satisfy configured schema: {messages}")


def _node_patch_changes(previous: GraphState, current: GraphState) -> list[dict[str, JsonValue]]:
    out: list[dict[str, JsonValue]] = []
    for change in diff_graphs(previous, current):
        payload = change.model_dump(mode="json")
        op = payload.get("op")
        if op == "AddNode":
            out.append({"edit": "add", "content": payload.get("content", "")})
        elif op == "ModifyNode":
            out.append(
                {
                    "edit": "revise",
                    "old_content": payload.get("old_content", ""),
                    "new_content": payload.get("new_content", ""),
                }
            )
        elif op == "RemoveNode":
            out.append({"edit": "remove", "content": payload.get("content", "")})
    return out


def _state_preflight(state: GraceState, schema: NetworkSchema) -> None:
    state.validate_integrity(schema_id=schema.id, schema_hash=schema.schema_hash)
    report = validate_graph(state.graph, schema)
    if not report.valid:
        messages = "; ".join(issue.message for issue in report.errors[:5])
        raise StateIntegrityError(f"input state graph is invalid: {messages}")


def evolve_graph(
    state: GraceState,
    diagnosis_report: str,
    *,
    provider: LLMProvider,
    schema: NetworkSchema,
    config: GraceConfig,
    prompt_sink: PromptSink | None = None,
) -> EvolutionResult:
    """Apply one full diagnosis-driven GRACE Evolution update."""

    _state_preflight(state, schema)
    if not diagnosis_report.strip():
        raise ValueError("diagnosis_report must not be empty; use advance_no_update")

    usage: list[UsageRecord] = []
    attempts: list[ProviderAttemptRecord] = []
    history: list[dict[str, JsonValue]] = []
    rejected: list[dict[str, JsonValue]] = []
    forced_drops: list[dict[str, JsonValue]] = []

    planning_pair = operation_planning(
        schema,
        state.graph.model_dump(mode="json"),
        diagnosis_report,
    )
    raw_plan = _complete_json(
        provider,
        planning_pair,
        stage="operation_planning",
        usage=usage,
        attempts=attempts,
        history=history,
        prompt_sink=prompt_sink,
        binding_id=state.state_id,
        response_schema=_operations_response_schema({"AddNode", "ModifyNode", "RemoveNode"}),
        max_output_tokens=config.max_output_tokens,
    )
    node_operations, stage_rejected = _partition_operations(
        _operations(raw_plan),
        {"AddNode", "ModifyNode", "RemoveNode"},
        stage="operation_planning",
    )
    rejected.extend(stage_rejected)
    node_assembly = assemble(state.graph, node_operations, schema)
    intermediate = node_assembly.graph
    rejected.extend(item.model_dump(mode="json") for item in node_assembly.rejected_operations)

    relation_pair = relation_maintenance(
        schema,
        intermediate.model_dump(mode="json"),
        list(node_assembly.touched_node_ids),
    )
    raw_relations = _complete_json(
        provider,
        relation_pair,
        stage="relation_maintenance",
        usage=usage,
        attempts=attempts,
        history=history,
        prompt_sink=prompt_sink,
        binding_id=state.state_id,
        response_schema=_operations_response_schema({"AddEdge", "RemoveEdge"}),
        max_output_tokens=config.max_output_tokens,
    )
    relation_operations, stage_rejected = _partition_operations(
        _operations(raw_relations),
        {"AddEdge", "RemoveEdge"},
        stage="relation_maintenance",
    )
    rejected.extend(stage_rejected)
    relation_assembly = assemble(intermediate, relation_operations, schema)
    candidate = relation_assembly.graph
    rejected.extend(item.model_dump(mode="json") for item in relation_assembly.rejected_operations)
    anchors = set(node_assembly.touched_node_ids) | set(relation_assembly.touched_node_ids)

    candidate, schema_history, drops = _validate_or_repair(
        candidate,
        schema,
        max_rounds=config.schema_repair_max_rounds,
    )
    forced_drops.extend(drops)
    history.append({"stage": "candidate_schema", "rounds": _json_value(schema_history)})

    sa_history: list[dict[str, JsonValue]] = []
    examined_node_ids: set[str] = set()
    sa_cap_reached = False
    if config.structural_analysis:
        for round_index in range(1, config.sa_max_rounds + 1):
            radius = round_index * config.sa_radius_step
            local = local_subgraph(candidate, anchors, radius, schema)
            examined_node_ids.update(node.id for node in local.nodes)
            analysis_pair = structural_analysis(
                schema,
                local.model_dump(mode="json"),
            )
            raw_analysis = _complete_json(
                provider,
                analysis_pair,
                stage=f"structural_analysis_{round_index}",
                usage=usage,
                attempts=attempts,
                history=history,
                prompt_sink=prompt_sink,
                binding_id=state.state_id,
                response_schema=_STRUCTURAL_ANALYSIS_RESPONSE_SCHEMA,
                max_output_tokens=config.max_output_tokens,
            )
            contradictions = raw_analysis.get("contradictions", [])
            redundancies = raw_analysis.get("redundancies", [])
            contradictions = contradictions if isinstance(contradictions, list) else []
            redundancies = redundancies if isinstance(redundancies, list) else []
            has_issues = bool(contradictions or redundancies)
            round_record: dict[str, JsonValue] = {
                "round": round_index,
                "radius": radius,
                "subgraph_node_ids": _json_value(sorted(node.id for node in local.nodes)),
                "graph_nodes_total": len(candidate.nodes),
                "contradictions": _json_value(contradictions),
                "redundancies": _json_value(redundancies),
                "passed": not has_issues,
            }
            sa_history.append(round_record)
            if not has_issues:
                break

            repair_pair = structural_repair(
                schema,
                local.model_dump(mode="json"),
                {
                    "contradictions": contradictions,
                    "redundancies": redundancies,
                },
            )
            raw_repair = _complete_json(
                provider,
                repair_pair,
                stage=f"structural_repair_{round_index}",
                usage=usage,
                attempts=attempts,
                history=history,
                prompt_sink=prompt_sink,
                binding_id=state.state_id,
                response_schema=_operations_response_schema(
                    {"ModifyNode", "Merge", "AddEdge", "RemoveEdge", "RemoveNode"}
                ),
                max_output_tokens=config.max_output_tokens,
            )
            repair_assembly = assemble(candidate, _operations(raw_repair), schema)
            candidate = repair_assembly.graph
            rejected.extend(
                item.model_dump(mode="json") for item in repair_assembly.rejected_operations
            )
            candidate, post_history, post_drops = _validate_or_repair(
                candidate,
                schema,
                max_rounds=config.schema_repair_max_rounds,
            )
            forced_drops.extend(post_drops)
            round_record["repair_schema_rounds"] = _json_value(post_history)
            round_record["repair_operations"] = _json_value(_operations(raw_repair))
            if round_index == config.sa_max_rounds:
                sa_cap_reached = True

    grounding = apply_grounding_policy(candidate, schema)
    evolved_graph = grounding.graph
    final_schema = validate_graph(evolved_graph, schema)
    if not final_schema.valid:
        messages = "; ".join(issue.message for issue in final_schema.errors[:5])
        raise SchemaValidationError(f"evolved graph is invalid: {messages}")

    raw_changes = diff_graphs(state.graph, evolved_graph)
    change_log = tuple(_json_object(change.model_dump(mode="json")) for change in raw_changes)
    node_changes = _node_patch_changes(state.graph, evolved_graph)
    reconstruction_pair = patch_reconstruction(state.instruction, node_changes)
    raw_reconstruction = _complete_json(
        provider,
        reconstruction_pair,
        stage="patch_reconstruction",
        usage=usage,
        attempts=attempts,
        history=history,
        prompt_sink=prompt_sink,
        binding_id=state.state_id,
        response_schema=_operations_response_schema({"ReplaceSpan", "InsertAfter", "DeleteSpan"}),
        max_output_tokens=config.max_output_tokens,
    )
    patch = apply_patch_operations(
        state.instruction,
        _operations(raw_reconstruction),
    )

    metadata = dict(evolved_graph.metadata)
    metadata.update(
        {
            "version": state.step + 1,
            "total_nodes": len(evolved_graph.nodes),
            "total_edges": len(evolved_graph.edges),
        }
    )
    evolved_graph = GraphState(
        nodes=evolved_graph.nodes,
        edges=evolved_graph.edges,
        metadata=metadata,
    )
    child = GraceState.from_parts(
        graph=evolved_graph,
        instruction=patch.instruction,
        schema_id=schema.id,
        schema_hash=schema.schema_hash,
        step=state.step + 1,
        parent_state_id=state.state_id,
    )
    child.validate_integrity(parent_state=state)

    warnings: list[str] = []
    if rejected:
        warnings.append(f"{len(rejected)} graph operation(s) rejected")
    if forced_drops:
        warnings.append(f"{len(forced_drops)} graph item(s) force-dropped")
    if sa_cap_reached:
        warnings.append("structural analysis reached its configured cap")
    warnings.extend(patch.warnings)
    validation_status = ValidationStatus.CAP_REACHED if sa_cap_reached else ValidationStatus.PASSED
    validation_report = ValidationReport(
        status=validation_status,
        schema_valid=True,
        rounds=len(sa_history),
        warnings=tuple(warnings),
        forced_drops=tuple(forced_drops),
        rejected_operations=tuple(rejected),
        examined_node_ids=tuple(sorted(examined_node_ids)),
        history=tuple([*history, {"stage": "sa", "rounds": _json_value(sa_history)}]),
    )
    result_status = ResultStatus.COMPLETED_WITH_WARNINGS if warnings else ResultStatus.COMPLETED
    provenance: dict[str, JsonValue] = {
        "input_state_id": state.state_id,
        "diagnosis_sha256": _sha256(diagnosis_report),
        "schema_id": schema.id,
        "schema_hash": schema.schema_hash,
        "provider_model": provider.model,
        "structural_analysis": config.structural_analysis,
        "u_node": _json_value(list(node_assembly.touched_node_ids)),
        "u_relation": _json_value(list(relation_assembly.touched_node_ids)),
        "anchor_set": _json_value(sorted(anchors)),
        "identity_grounding_edges_added": grounding.added_count,
        "intermediate_graph_hash": intermediate.content_hash(),
        "candidate_graph_hash": candidate.content_hash(),
    }
    return EvolutionResult(
        state=child,
        change_log=change_log,
        validation_report=validation_report,
        reconstruction_report=patch.to_reconstruction_report(),
        provenance=provenance,
        usage=tuple(usage),
        attempts=tuple(attempts),
        status=result_status,
        warnings=tuple(warnings),
    )


def advance_no_update(
    state: GraceState,
    *,
    schema: NetworkSchema,
) -> EvolutionResult:
    """Create an explicit zero-call no-op child for a no-failure batch."""

    _state_preflight(state, schema)
    patch = apply_patch_operations(state.instruction, ())
    child = GraceState.from_parts(
        graph=state.graph,
        instruction=state.instruction,
        schema_id=schema.id,
        schema_hash=schema.schema_hash,
        step=state.step + 1,
        parent_state_id=state.state_id,
    )
    child.validate_integrity(parent_state=state)
    return EvolutionResult(
        state=child,
        change_log=(),
        validation_report=ValidationReport(
            status=ValidationStatus.PASSED,
            schema_valid=True,
        ),
        reconstruction_report=patch.to_reconstruction_report(),
        provenance={
            "input_state_id": state.state_id,
            "schema_id": schema.id,
            "schema_hash": schema.schema_hash,
            "no_update": True,
        },
        usage=(),
        status=ResultStatus.COMPLETED,
    )


__all__ = ["advance_no_update", "evolve_graph"]
