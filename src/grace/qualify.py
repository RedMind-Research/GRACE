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

"""Installed-package, zero-provider offline qualification CLI.

Run after installing the wheel::

    python -m grace.qualify offline --report qualification.json

The command imports no tests or reproduction modules, reads no credential
environment variables, and wraps all checks in a deny-network/deny-provider
guard.  Its JSON report contains only fixed configuration facts, hashes, and
safe route descriptors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from grace import __version__ as GRACE_VERSION
from grace.artifacts.provenance import assert_artifact_safe
from grace.config import GraceConfig
from grace.errors import StateIntegrityError
from grace.graph.models import Edge, GraceState, GraphState, Node
from grace.graph.validation import validate_graph
from grace.initialization.prompts import prompt_to_graph
from grace.providers.config import GoogleAIStudioConfig, VertexAIADCConfig
from grace.schemas.default import DefaultGraceSchema
from grace.schemas.models import NetworkSchema, ObjectType, RelationType


REPORT_SCHEMA: Final = "grace.offline-qualification-report"
REPORT_SCHEMA_VERSION: Final = "1"
EXPECTED_DEFAULT_SCHEMA_HASH: Final = (
    "bd8d0e49a6f6362dd8e2a479001f0767d298ae80afd915372b38a239631acd2b"
)

EXIT_PASSED: Final = 0
EXIT_QUALIFICATION_FAILED: Final = 1
EXIT_USAGE_OR_REPORT_ERROR: Final = 2

_SAFE_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,79}$")


class _QualificationFailure(RuntimeError):
    """Internal assertion failure whose message is never serialized."""


class _OfflineBoundaryViolation(RuntimeError):
    """Raised when qualification code attempts provider or network activity."""


CheckFunction = Callable[[], dict[str, Any]]
CheckSpec = tuple[str, str, CheckFunction]


def _require(condition: bool) -> None:
    if not condition:
        raise _QualificationFailure


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check_import_boundary() -> dict[str, Any]:
    # The caller may legitimately have used the reproduction integration before
    # requesting an offline report.  The property being qualified is a *fresh*
    # ``import grace`` boundary, so test it in an isolated interpreter instead
    # of depending on caller or pytest import order.
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys, grace; "
                "print('clean' if not any(n == 'litellm' or "
                "n.startswith('litellm.') for n in sys.modules) else 'eager')"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    _require(probe.returncode == 0 and probe.stdout.strip() == "clean")
    return {
        "external_litellm_imported": False,
        "package": "grace",
    }


def _check_default_schema() -> dict[str, Any]:
    schema = DefaultGraceSchema()
    _require(schema.id == "grace-default")
    _require(schema.version == "1")
    _require(schema.schema_hash == EXPECTED_DEFAULT_SCHEMA_HASH)
    return {
        "schema_id": schema.id,
        "schema_version": schema.version,
        "schema_hash": schema.schema_hash,
    }


def _check_route_descriptors() -> dict[str, Any]:
    placeholder_project = "grace-offline-qualification-placeholder"
    vertex = VertexAIADCConfig(
        project=placeholder_project,
        location="us-central1",
    )
    vertex_descriptor = vertex.safe_descriptor().model_dump(mode="json")
    vertex_serialized = json.dumps(vertex_descriptor, sort_keys=True)
    _require(vertex_descriptor["route"] == "vertex_ai_adc")
    _require(vertex_descriptor["model"] == "vertex_ai/gemini-2.5-flash")
    _require(vertex_descriptor["location"] == "us-central1")
    _require(vertex_descriptor["project_hash"] == _sha256_text(placeholder_project))
    _require(placeholder_project not in vertex_serialized)

    placeholder_env_name = "GRACE_QUALIFICATION_UNUSED_KEY"
    ai_studio = GoogleAIStudioConfig(api_key_env=placeholder_env_name)
    ai_descriptor = ai_studio.safe_descriptor().model_dump(mode="json")
    ai_serialized = json.dumps(ai_descriptor, sort_keys=True)
    _require(ai_descriptor["route"] == "google_ai_studio")
    _require(ai_descriptor["model"] == "gemini/gemini-2.5-flash")
    _require(ai_descriptor["project_hash"] is None)
    _require(placeholder_env_name not in ai_serialized)

    return {
        "vertex": vertex_descriptor,
        "google_ai_studio": ai_descriptor,
    }


def _check_grace_config() -> dict[str, Any]:
    config = GraceConfig()
    expected = {
        "max_output_tokens": 65536,
        "p2g_max_rounds": 10,
        "structural_analysis": True,
        "sa_max_rounds": 3,
        "sa_radius_step": 3,
        "schema_repair_max_rounds": 3,
        "artifact_dir": "grace_runs",
        "artifact_mode": "checkpoint",
    }
    _require(config.model_dump(mode="json") == expected)
    try:
        config.artifact_mode = "audit"  # type: ignore[misc]
    except ValidationError:
        frozen = True
    else:  # pragma: no cover - protects a future configuration regression
        frozen = False
    _require(frozen)
    return {
        "frozen": True,
        "defaults": expected,
    }


def _custom_schema() -> NetworkSchema:
    return NetworkSchema(
        id="qualification-custom-ontology",
        version="1",
        description="A non-default ontology used only for offline qualification.",
        object_types=(
            ObjectType(id="actor_profile", description="The acting subject and mandate."),
            ObjectType(id="protocol_rule", description="A required operating rule."),
        ),
        relation_types=(
            RelationType(
                id="governs",
                description="The source actor is governed by the target rule.",
                allowed_pairs=(("actor_profile", "protocol_rule"),),
            ),
        ),
    )


def _check_custom_ontology_prompt() -> dict[str, Any]:
    schema = _custom_schema()
    prompt = prompt_to_graph(
        schema,
        "An operator follows the approved protocol before changing a record.",
    )
    combined = f"{prompt.system}\n{prompt.user}"
    lowered = combined.lower()
    for custom_token in ("actor_profile", "protocol_rule", "governs"):
        _require(custom_token in combined)
    for default_token in (
        "identity",
        "norm",
        "knowledge",
        "supports",
        "refines",
        "sequence",
    ):
        _require(re.search(rf"\b{re.escape(default_token)}\b", lowered) is None)
    return {
        "schema_id": schema.id,
        "schema_hash": schema.schema_hash,
        "prompt_hash": _sha256_text(combined),
        "object_types": list(schema.object_type_ids),
        "relation_types": list(schema.relation_type_ids),
        "default_type_strings_absent": True,
    }


def _check_state_integrity() -> dict[str, Any]:
    schema = _custom_schema()
    graph = GraphState(
        nodes=(
            Node(
                id="actor-1",
                type="actor_profile",
                content="The operator acts under an approved mandate.",
            ),
            Node(
                id="rule-1",
                type="protocol_rule",
                content="Validate the record before changing it.",
            ),
        ),
        edges=(Edge(source="actor-1", target="rule-1", relation="governs"),),
    )
    _require(validate_graph(graph, schema).valid)
    state = GraceState.from_parts(
        graph=graph,
        instruction="Validate the record before changing it.",
        schema_id=schema.id,
        schema_hash=schema.schema_hash,
    )
    _require(state.validate_integrity(schema_id=schema.id, schema_hash=schema.schema_hash))
    reloaded = GraceState.model_validate(state.model_dump(mode="json"))
    _require(reloaded.state_id == state.state_id)

    tampered = state.model_dump(mode="json")
    tampered["instruction"] = "Tampered instruction"
    try:
        GraceState.model_validate(tampered)
    except (StateIntegrityError, ValidationError):
        tamper_rejected = True
    else:  # pragma: no cover - protects a future state-model regression
        tamper_rejected = False
    _require(tamper_rejected)
    return {
        "schema_hash": schema.schema_hash,
        "state_id": state.state_id,
        "graph_hash": state.graph_hash,
        "instruction_hash": state.instruction_hash,
        "round_trip": True,
        "tamper_rejected": True,
    }


_OFFLINE_CHECKS: tuple[CheckSpec, ...] = (
    (
        "import_no_eager_litellm",
        "Importing the installed GRACE package does not eagerly import external LiteLLM.",
        _check_import_boundary,
    ),
    (
        "default_schema_hash",
        "The paper's default schema matches the frozen release hash.",
        _check_default_schema,
    ),
    (
        "safe_route_descriptors",
        "Vertex and AI Studio route descriptors are explicit and credential-free.",
        _check_route_descriptors,
    ),
    (
        "grace_config_defaults",
        "GraceConfig is frozen and retains the public method/artifact defaults.",
        _check_grace_config,
    ),
    (
        "custom_ontology_prompt",
        "Prompt-to-Graph renders a custom ontology without default type assumptions.",
        _check_custom_ontology_prompt,
    ),
    (
        "grace_state_integrity",
        "A custom-schema GraceState round trips and rejects content tampering.",
        _check_state_integrity,
    ),
)


@contextmanager
def _offline_boundary() -> Iterator[dict[str, int]]:
    """Deny supported provider dispatch and common socket connection paths."""

    from grace.providers.litellm import LiteLLMProvider

    counters = {"network_attempts": 0, "provider_calls": 0}
    original_create_connection = socket.create_connection
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    original_complete = LiteLLMProvider.complete

    def deny_network(*_args: Any, **_kwargs: Any) -> None:
        counters["network_attempts"] += 1
        raise _OfflineBoundaryViolation

    def deny_connect_ex(*_args: Any, **_kwargs: Any) -> int:
        counters["network_attempts"] += 1
        raise _OfflineBoundaryViolation

    def deny_provider(*_args: Any, **_kwargs: Any) -> None:
        counters["provider_calls"] += 1
        raise _OfflineBoundaryViolation

    socket.create_connection = deny_network  # type: ignore[assignment]
    socket.socket.connect = deny_network  # type: ignore[method-assign]
    socket.socket.connect_ex = deny_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = deny_network  # type: ignore[assignment]
    LiteLLMProvider.complete = deny_provider  # type: ignore[method-assign,assignment]
    try:
        yield counters
    finally:
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
        LiteLLMProvider.complete = original_complete  # type: ignore[method-assign,assignment]


def _safe_error_type(exception: Exception) -> str:
    candidate = type(exception).__name__
    return candidate if _SAFE_ERROR_TYPE.fullmatch(candidate) else "Exception"


def _run_check(check_id: str, description: str, check: CheckFunction) -> dict[str, Any]:
    try:
        evidence = check()
    except Exception as exc:
        return {
            "id": check_id,
            "status": "failed",
            "description": description,
            "evidence": {"error_type": _safe_error_type(exc)},
        }
    return {
        "id": check_id,
        "status": "passed",
        "description": description,
        "evidence": evidence,
    }


def run_offline_qualification(
    checks: Sequence[CheckSpec] | None = None,
) -> dict[str, Any]:
    """Run deterministic installed-package checks and return a safe report."""

    selected_checks = tuple(checks) if checks is not None else _OFFLINE_CHECKS
    results: list[dict[str, Any]] = []
    with _offline_boundary() as counters:
        for check_id, description, check in selected_checks:
            results.append(_run_check(check_id, description, check))

    boundary_passed = counters == {"network_attempts": 0, "provider_calls": 0}
    results.append(
        {
            "id": "offline_execution_boundary",
            "status": "passed" if boundary_passed else "failed",
            "description": (
                "Qualification executed with supported provider calls and socket "
                "connection paths denied."
            ),
            "evidence": counters,
        }
    )
    passed = sum(item["status"] == "passed" for item in results)
    failed = len(results) - passed
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "grace_version": GRACE_VERSION,
        "qualification": "offline",
        "status": "passed" if failed == 0 else "failed",
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
        },
        "checks": results,
    }
    _validate_report(report)
    assert_artifact_safe(report)
    return report


def _validate_report(report: dict[str, Any]) -> None:
    expected_top_level = {
        "schema",
        "schema_version",
        "generated_at",
        "grace_version",
        "qualification",
        "status",
        "summary",
        "checks",
    }
    _require(set(report) == expected_top_level)
    _require(report["schema"] == REPORT_SCHEMA)
    _require(report["schema_version"] == REPORT_SCHEMA_VERSION)
    _require(report["qualification"] == "offline")
    _require(report["status"] in {"passed", "failed"})
    _require(isinstance(report["grace_version"], str) and bool(report["grace_version"]))
    _require(isinstance(report["generated_at"], str))
    generated_at = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
    _require(generated_at.utcoffset() == timezone.utc.utcoffset(generated_at))
    checks = report["checks"]
    _require(isinstance(checks, list) and bool(checks))
    ids = [item["id"] for item in checks]
    _require(len(ids) == len(set(ids)))
    _require(
        all(
            set(item) == {"id", "status", "description", "evidence"}
            and item["status"] in {"passed", "failed"}
            and isinstance(item["id"], str)
            and bool(item["id"])
            and isinstance(item["description"], str)
            and bool(item["description"])
            and isinstance(item["evidence"], dict)
            for item in checks
        )
    )
    summary = report["summary"]
    _require(set(summary) == {"total", "passed", "failed"})
    _require(summary["total"] == len(checks))
    _require(summary["passed"] + summary["failed"] == summary["total"])
    _require(summary["failed"] == sum(item["status"] == "failed" for item in checks))
    _require((report["status"] == "passed") == (summary["failed"] == 0))


def _atomic_write_report(path: Path, report: dict[str, Any]) -> None:
    data = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    parent = path.expanduser().absolute().parent
    target = path.expanduser().absolute()
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short qualification report write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, target)
        try:
            parent_descriptor = os.open(parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(parent_descriptor)
        except OSError:
            pass
        finally:
            os.close(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m grace.qualify")
    subparsers = parser.add_subparsers(dest="command", required=True)
    offline = subparsers.add_parser(
        "offline",
        help="run zero-provider installed-package qualification",
    )
    offline.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "offline":  # pragma: no cover - argparse constrains this
        return EXIT_USAGE_OR_REPORT_ERROR
    try:
        report = run_offline_qualification()
        _atomic_write_report(args.report, report)
    except Exception:
        print("GRACE offline qualification could not produce a report.", file=sys.stderr)
        return EXIT_USAGE_OR_REPORT_ERROR
    if report["status"] != "passed":
        print("GRACE offline qualification failed; inspect the safe JSON report.", file=sys.stderr)
        return EXIT_QUALIFICATION_FAILED
    print("GRACE offline qualification passed.")
    return EXIT_PASSED


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_PASSED",
    "EXIT_QUALIFICATION_FAILED",
    "EXIT_USAGE_OR_REPORT_ERROR",
    "EXPECTED_DEFAULT_SCHEMA_HASH",
    "REPORT_SCHEMA",
    "REPORT_SCHEMA_VERSION",
    "main",
    "run_offline_qualification",
]
