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

"""Provider-injected Prompt-to-Graph initialization for GRACE."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Callable

from pydantic import JsonValue, ValidationError

from grace.artifacts.models import (
    InitializationResult,
    ResultStatus,
    ValidationReport,
    ValidationStatus,
)
from grace.config import DEFAULT_MAX_OUTPUT_TOKENS
from grace.errors import ProviderError, SchemaValidationError
from grace.graph.grounding import apply_grounding_policy
from grace.graph.models import GraceState, GraphState
from grace.graph.operations import assemble
from grace.graph.repair import repair_graph
from grace.graph.validation import validate_graph
from grace.initialization.prompts import (
    PromptPair,
    fidelity_analysis,
    fidelity_repair,
    prompt_to_graph,
)
from grace.providers.base import (
    LLMProvider,
    PromptRequest,
    ProviderAttemptRecord,
    UsageRecord,
    make_logical_call_id,
)
from grace.schemas.models import NetworkSchema


_FIDELITY_KEYS = (
    "missing_node",
    "unfaithful",
    "mistyped",
    "wrong_relation",
    "missing_relation",
)

PromptSink = Callable[[str, PromptPair], None]

_OPERATIONS_RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"operations": {"type": "array"}},
    "required": ["operations"],
    "additionalProperties": True,
}
_FIDELITY_RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {key: {"type": "array"} for key in _FIDELITY_KEYS},
    "required": list(_FIDELITY_KEYS),
    "additionalProperties": False,
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_object(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False, default=str))


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


def _operation_list(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    operations = value.get("operations", [])
    if not isinstance(operations, list):
        return []
    return [item for item in operations if isinstance(item, Mapping)]


def _normalize_fidelity(
    value: Mapping[str, Any],
    graph: GraphState,
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Normalize fidelity lists and apply the frozen anti-false-positive filters."""

    normalized: dict[str, list[Any]] = {}
    for key in _FIDELITY_KEYS:
        raw = value.get(key, [])
        normalized[key] = list(raw) if isinstance(raw, list) else []

    existing_edges = {(edge.source, edge.target, edge.relation) for edge in graph.edges}
    wrong_raw = normalized["wrong_relation"]
    normalized["wrong_relation"] = [
        item
        for item in wrong_raw
        if not (
            isinstance(item, Mapping)
            and (
                item.get("source"),
                item.get("target"),
                item.get("should_be"),
            )
            in existing_edges
        )
    ]

    degree = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
    isolated = {node_id for node_id, count in degree.items() if count == 0}
    missing_raw = normalized["missing_relation"]
    normalized["missing_relation"] = [
        item
        for item in missing_raw
        if isinstance(item, Mapping)
        and (item.get("source") in isolated or item.get("target") in isolated)
    ]

    filtered = {
        "wrong_relation_false_positives": len(wrong_raw) - len(normalized["wrong_relation"]),
        "missing_relation_filtered": len(missing_raw) - len(normalized["missing_relation"]),
    }
    return normalized, filtered


def _has_fidelity_issues(issues: Mapping[str, list[Any]]) -> bool:
    return any(issues.get(key) for key in _FIDELITY_KEYS)


def initialize_graph(
    instruction: str,
    *,
    provider: LLMProvider,
    schema: NetworkSchema,
    max_rounds: int = 10,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    prompt_sink: PromptSink | None = None,
) -> InitializationResult:
    """Create integrity-bound state zero from an instruction.

    The loop is schema detect-only plus provider-assisted fidelity repair.  A
    single destructive deterministic repair is reserved for non-converged
    closeout, and every rewrite or drop remains visible in the returned report.
    """

    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least one")
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")

    usage: list[UsageRecord] = []
    attempts: list[ProviderAttemptRecord] = []
    history: list[dict[str, JsonValue]] = []
    initialization_binding = _sha256(instruction)
    initial_pair = prompt_to_graph(schema, instruction)
    raw_graph = _complete_json(
        provider,
        initial_pair,
        stage="p2g_initial",
        usage=usage,
        attempts=attempts,
        history=history,
        prompt_sink=prompt_sink,
        binding_id=initialization_binding,
        response_schema=GraphState.model_json_schema(),
        max_output_tokens=max_output_tokens,
    )
    try:
        graph = GraphState.model_validate(raw_graph)
    except ValidationError as error:
        raise ProviderError(f"P2G returned an invalid graph payload: {error}") from error

    grounding = apply_grounding_policy(graph, schema)
    graph = grounding.graph
    identity_edges_added = grounding.added_count
    rejected_records: list[dict[str, JsonValue]] = []
    forced_drop_records: list[dict[str, JsonValue]] = []
    rejected_feedback: list[dict[str, JsonValue]] = []
    seen_rejected: set[str] = set()
    converged = False
    stable_stop = False
    cap_reached = False
    unchanged_rounds = 0
    last_graph_hash: str | None = None
    final_issues: dict[str, list[Any]] = {key: [] for key in _FIDELITY_KEYS}

    for round_index in range(1, max_rounds + 1):
        schema_report = validate_graph(graph, schema)
        analysis_pair = fidelity_analysis(
            schema,
            instruction,
            graph.model_dump(mode="json"),
        )
        raw_analysis = _complete_json(
            provider,
            analysis_pair,
            stage=f"p2g_fidelity_{round_index}",
            usage=usage,
            attempts=attempts,
            history=history,
            prompt_sink=prompt_sink,
            binding_id=initialization_binding,
            response_schema=_FIDELITY_RESPONSE_SCHEMA,
            max_output_tokens=max_output_tokens,
        )
        issues, filtered = _normalize_fidelity(raw_analysis, graph)
        final_issues = issues
        round_record: dict[str, JsonValue] = {
            "stage": "p2g_round",
            "round": round_index,
            "schema_valid": schema_report.valid,
            "schema_errors": [issue.model_dump(mode="json") for issue in schema_report.errors],
            "fidelity_counts": _json_object({key: len(value) for key, value in issues.items()}),
            "filtered_counts": _json_object(filtered),
            "graph_hash": graph.content_hash(),
        }
        history.append(round_record)

        if schema_report.valid and not _has_fidelity_issues(issues):
            converged = True
            break
        if round_index == max_rounds:
            cap_reached = True
            break

        current_hash = graph.content_hash()
        if current_hash == last_graph_hash:
            unchanged_rounds += 1
            if unchanged_rounds >= 2:
                stable_stop = True
                break
        else:
            unchanged_rounds = 0
        last_graph_hash = current_hash

        repair_pair = fidelity_repair(
            schema,
            instruction,
            graph.model_dump(mode="json"),
            {
                "schema_violations": [
                    issue.model_dump(mode="json") for issue in schema_report.errors
                ],
                **issues,
            },
            rejected_feedback,
        )
        raw_repair = _complete_json(
            provider,
            repair_pair,
            stage=f"p2g_repair_{round_index}",
            usage=usage,
            attempts=attempts,
            history=history,
            prompt_sink=prompt_sink,
            binding_id=initialization_binding,
            response_schema=_OPERATIONS_RESPONSE_SCHEMA,
            max_output_tokens=max_output_tokens,
        )
        assembly = assemble(graph, _operation_list(raw_repair), schema)
        graph = assembly.graph
        for rejected in assembly.rejected_operations:
            record = rejected.model_dump(mode="json")
            rejected_records.append(record)
            key = json.dumps(record.get("operation"), sort_keys=True, default=str)
            if key not in seen_rejected:
                seen_rejected.add(key)
                rejected_feedback.append(record)
        grounding = apply_grounding_policy(graph, schema)
        graph = grounding.graph
        identity_edges_added += grounding.added_count

    if not converged:
        repaired = repair_graph(graph, schema)
        graph = repaired.graph
        forced_drop_records.extend(drop.model_dump(mode="json") for drop in repaired.forced_drops)
        grounding = apply_grounding_policy(graph, schema)
        graph = grounding.graph
        identity_edges_added += grounding.added_count

    metadata = dict(graph.metadata)
    metadata.update(
        {
            "version": 0,
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
        }
    )
    graph = GraphState(nodes=graph.nodes, edges=graph.edges, metadata=metadata)
    final_schema = validate_graph(graph, schema)
    if not final_schema.valid:
        messages = "; ".join(issue.message for issue in final_schema.errors[:5])
        raise SchemaValidationError(f"P2G final graph is invalid: {messages}")

    fidelity_valid = not _has_fidelity_issues(final_issues)
    if converged:
        validation_status = ValidationStatus.CONVERGED
    elif stable_stop:
        validation_status = ValidationStatus.STABLE_STOP
    else:
        validation_status = ValidationStatus.CAP_REACHED

    warnings: list[str] = []
    if stable_stop:
        warnings.append("P2G reached a schema-valid stable stop before fidelity convergence")
    if cap_reached:
        warnings.append("P2G reached its configured fidelity-validation cap")
    if not fidelity_valid:
        warnings.append("P2G retains reported fidelity issues")
    if forced_drop_records:
        warnings.append(
            f"deterministic closeout changed or dropped {len(forced_drop_records)} item(s)"
        )
    if rejected_records:
        warnings.append(f"deterministic assembly rejected {len(rejected_records)} operation(s)")

    state = GraceState.from_parts(
        graph=graph,
        instruction=instruction,
        schema_id=schema.id,
        schema_hash=schema.schema_hash,
        step=0,
    )
    fidelity_error_records: list[dict[str, JsonValue]] = []
    for kind, items in final_issues.items():
        for item in items:
            detail: JsonValue
            if isinstance(item, Mapping):
                detail = _json_object(item)
            elif item is None or isinstance(item, (str, int, float, bool)):
                detail = item
            else:
                detail = str(item)
            fidelity_error_records.append({"kind": kind, "detail": detail})

    validation_report = ValidationReport(
        status=validation_status,
        schema_valid=True,
        fidelity_valid=fidelity_valid,
        rounds=sum(1 for item in history if item.get("stage") == "p2g_round"),
        errors=tuple(fidelity_error_records),
        warnings=tuple(warnings),
        forced_drops=tuple(forced_drop_records),
        rejected_operations=tuple(rejected_records),
        history=tuple(history),
    )
    result_status = ResultStatus.COMPLETED_WITH_WARNINGS if warnings else ResultStatus.COMPLETED
    # Keep deterministic grounding telemetry in the final history without
    # creating a default-ontology field in the public state contract.
    if identity_edges_added:
        history.append({"stage": "grounding", "added_edges": identity_edges_added})
        validation_report = validation_report.model_copy(update={"history": tuple(history)})

    return InitializationResult(
        state=state,
        validation_report=validation_report,
        usage=tuple(usage),
        attempts=tuple(attempts),
        status=result_status,
        warnings=tuple(warnings),
    )


__all__ = ["initialize_graph"]
