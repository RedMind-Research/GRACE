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

import hashlib
import json
import multiprocessing
from pathlib import Path
from queue import Empty

import pytest

from grace.artifacts.models import (
    ArtifactManifest,
    ReconstructionReport,
    ReconstructionStatus,
    ResultStatus,
    ValidationReport,
    ValidationStatus,
)
from grace.artifacts.provenance import canonical_json_bytes, capture_prompt_provenance
from grace.artifacts.store import ArtifactStore
from grace.errors import ArtifactError
from grace.graph.models import Edge, GraceState, GraphState, Node
from grace.providers.base import ProviderAttemptRecord, UsageRecord
from grace.schemas.default import DefaultGraceSchema


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _graph() -> GraphState:
    return GraphState(
        nodes=(
            Node(id="identity-root", type="identity", content="Act as a telecom agent."),
            Node(id="norm-1", type="norm", content="Verify the customer before changes."),
        ),
        edges=(Edge(source="identity-root", target="norm-1", relation="supports"),),
    )


def _initial_state() -> GraceState:
    schema = DefaultGraceSchema()
    return GraceState.from_parts(
        graph=_graph(),
        instruction="Verify the customer before making account changes.",
        schema_id=schema.id,
        schema_hash=schema.schema_hash,
    )


def test_write_bytes_preserves_lf_payload(tmp_path: Path) -> None:
    target = tmp_path / "instruction.txt"
    payload = b"first line\nsecond line\n"

    ArtifactStore._write_bytes(target, payload)

    assert target.read_bytes() == payload


def _child_state(
    parent: GraceState,
    suffix: str = " Record the verification outcome.",
) -> GraceState:
    return GraceState.from_parts(
        graph=parent.graph,
        instruction=parent.instruction + suffix,
        schema_id=parent.schema_id,
        schema_hash=parent.schema_hash,
        step=parent.step + 1,
        parent_state_id=parent.state_id,
    )


def _validation(*, with_history: bool = True) -> ValidationReport:
    return ValidationReport(
        status=ValidationStatus.PASSED,
        schema_valid=True,
        examined_node_ids=("norm-1",),
        history=({"round": 1, "decision": "accepted"},) if with_history else (),
    )


def _usage() -> UsageRecord:
    return UsageRecord(
        model="vertex_ai/gemini-2.5-flash",
        route="vertex_ai_adc",
        input_tokens=12,
        output_tokens=5,
        latency_seconds=0.25,
        attempts=2,
        retries=1,
    )


def _attempts() -> tuple[ProviderAttemptRecord, ...]:
    return (
        ProviderAttemptRecord(
            attempt=1,
            temperature=0.0,
            timeout_seconds=30.0,
            overall_timeout_seconds=90.0,
            elapsed_since_call_start_seconds=0.0,
            latency_seconds=0.1,
            outcome="retry",
            category="malformed_response",
            retryable=True,
        ),
        ProviderAttemptRecord(
            attempt=2,
            temperature=0.5,
            timeout_seconds=30.0,
            overall_timeout_seconds=90.0,
            elapsed_since_call_start_seconds=0.1,
            latency_seconds=0.15,
            outcome="succeeded",
            category="success",
        ),
    )


def _initial_payload() -> dict[str, object]:
    return {"validation_report": _validation().model_dump(mode="json")}


def _evolution_payload(parent: GraceState, child: GraceState) -> dict[str, object]:
    reconstruction = ReconstructionReport(
        status=ReconstructionStatus.APPLIED,
        input_instruction_hash=parent.instruction_hash,
        output_instruction_hash=child.instruction_hash,
        applied_operations=("append verification outcome",),
    )
    return {
        "change_log": [{"kind": "instruction_patch"}],
        "validation_report": _validation().model_dump(mode="json"),
        "reconstruction_report": reconstruction.model_dump(mode="json"),
        "provenance": {"diagnosis_hash": _digest("diagnosis")},
    }


def _write_initial(
    store: ArtifactStore,
    *,
    state: GraceState | None = None,
    prompt_snapshot: dict[str, object] | None = None,
    audit_payload: dict[str, object] | None = None,
) -> Path | None:
    return store.write_state(
        state=state or _initial_state(),
        schema=DefaultGraceSchema(),
        result_kind="initialization",
        result_payload=_initial_payload(),
        prompt_snapshot=prompt_snapshot,
        input_hashes={"instruction": _digest("instruction")},
        config_hash=_digest("config"),
        models=("vertex_ai/gemini-2.5-flash",),
        usage=(_usage(),),
        attempts=_attempts(),
        status=ResultStatus.COMPLETED,
        audit_payload=audit_payload,
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _concurrent_writer(
    root: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    """Spawn-safe writer used to exercise the filesystem race boundary."""

    store = ArtifactStore(root, mode="checkpoint", run_id="run-concurrent")
    start.wait(timeout=10)
    try:
        location = _write_initial(store)
    except Exception as exc:  # pragma: no cover - surfaced in the parent assertion
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", str(location)))


def _concurrent_child_writer(
    root: str,
    suffix: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    store = ArtifactStore(root, mode="checkpoint", run_id="run-siblings")
    parent = _initial_state()
    child = _child_state(parent, suffix)
    start.wait(timeout=10)
    try:
        location = store.write_state(
            state=child,
            schema=DefaultGraceSchema(),
            result_kind="evolution",
            result_payload=_evolution_payload(parent, child),
            input_hashes={"diagnosis_report": _digest(suffix)},
            config_hash=_digest("config"),
        )
    except Exception as exc:  # pragma: no cover - surfaced in the parent assertion
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", str(location), child.state_id))


def test_checkpoint_round_trip_is_minimal_self_contained_and_hashed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", mode="checkpoint", run_id="run-checkpoint")
    prompt = {
        "p2g.initial": {
            "system_prompt": "Use the configured graph schema.",
            "user_prompt": "Initialize the exact instruction.",
        }
    }

    location = _write_initial(store, prompt_snapshot=prompt)

    assert location is not None
    assert store.schema_path.is_file()
    assert {path.name for path in location.iterdir()} == {
        "graph.json",
        "instruction.txt",
        "manifest.json",
        "result.json",
        "validation_report.json",
    }
    record = store.verify_record(location, expected_schema=DefaultGraceSchema())
    assert record.state == _initial_state()
    assert record.schema_snapshot.schema_hash == DefaultGraceSchema().schema_hash
    assert record.prompt_provenance is None
    assert record.manifest.manifest_hash == record.manifest.computed_manifest_hash()
    assert len(record.manifest.attempts) == 2
    assert "__snapshot__" in record.manifest.prompt_hashes
    assert "history" not in record.result_payload["validation_report"]
    assert "examined_node_ids" not in record.result_payload["validation_report"]
    assert not (location / "prompt_snapshots.json").exists()


def test_audit_is_checkpoint_superset_and_recomputes_actual_prompt_provenance(
    tmp_path: Path,
) -> None:
    prompt = {
        "patch": {"system_prompt": "Patch only accepted anchors", "user_prompt": "Delta"},
        "operation_plan": {"system_prompt": "Plan exact edits", "user_prompt": "Report"},
    }
    audit = {
        "input_graph": _graph().model_dump(mode="json"),
        "input_instruction": "Previous instruction",
        "diagnosis_report": "Observed verification omissions.",
        "operation_plan": [{"op": "UpdateContent"}],
        "sa_history": [{"round": 1, "radius": 3}],
    }
    store = ArtifactStore(tmp_path / "runs", mode="audit", run_id="run-audit")

    location = _write_initial(store, prompt_snapshot=prompt, audit_payload=audit)

    assert location is not None
    assert {path.name for path in location.iterdir()} == {
        "audit.json",
        "graph.json",
        "instruction.txt",
        "manifest.json",
        "prompt_provenance.json",
        "prompt_snapshots.json",
        "result.json",
        "validation_report.json",
    }
    record = store.verify_record(location)
    expected_provenance, normalized = capture_prompt_provenance(prompt)
    assert record.prompt_provenance == expected_provenance
    assert json.loads((location / "prompt_snapshots.json").read_text()) == normalized
    assert record.audit_payload == audit
    assert record.result_payload["validation_report"]["history"]


def test_none_mode_writes_zero_bytes(tmp_path: Path) -> None:
    root = tmp_path / "must-not-exist"
    store = ArtifactStore(root, mode="none", run_id="run-none")

    assert _write_initial(store) is None
    assert not root.exists()
    assert store.find_state(_initial_state().state_id) is None


def test_evolution_round_trip_parent_resume_and_rollback(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", mode="checkpoint", run_id="run-lineage")
    parent = _initial_state()
    child = _child_state(parent)
    parent_path = _write_initial(store, state=parent)
    assert parent_path is not None

    child_path = store.write_state(
        state=child,
        schema=DefaultGraceSchema(),
        result_kind="evolution",
        result_payload=_evolution_payload(parent, child),
        input_hashes={"diagnosis_report": _digest("diagnosis")},
        config_hash=_digest("config"),
        models=("vertex_ai/gemini-2.5-flash",),
        usage=(_usage(),),
        attempts=_attempts(),
    )

    assert child_path is not None
    assert store.resume(child.state_id, parent_state=parent) == child
    assert store.load_parent(child_path) == parent
    assert store.load_parent(parent_path) is None


def test_repeated_write_is_zero_mutation_and_conflicting_write_is_rejected(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "runs", mode="checkpoint", run_id="run-idempotent")
    first = _write_initial(store)
    assert first is not None
    before = _tree_snapshot(store.root)

    second = _write_initial(store)

    assert second == first
    assert _tree_snapshot(store.root) == before
    conflicting_report = _validation(with_history=False).model_copy(update={"rounds": 2})
    conflicting = {"validation_report": conflicting_report.model_dump(mode="json")}
    with pytest.raises(ArtifactError, match="different artifact content"):
        store.write_state(
            state=_initial_state(),
            schema=DefaultGraceSchema(),
            result_kind="initialization",
            result_payload=conflicting,
            input_hashes={"instruction": _digest("instruction")},
            config_hash=_digest("config"),
            models=("vertex_ai/gemini-2.5-flash",),
            usage=(_usage(),),
            attempts=_attempts(),
        )


@pytest.mark.parametrize("filename", ["graph.json", "instruction.txt", "schema.json"])
def test_tampered_content_is_rejected(tmp_path: Path, filename: str) -> None:
    store = ArtifactStore(tmp_path / filename, mode="checkpoint", run_id="run-tamper")
    location = _write_initial(store)
    assert location is not None
    target = store.schema_path if filename == "schema.json" else location / filename
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(ArtifactError, match="hash mismatch|canonical|schema"):
        store.load_state(location)


def test_manifest_tamper_and_rehashed_unknown_format_are_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", mode="checkpoint", run_id="run-manifest")
    location = _write_initial(store)
    assert location is not None
    manifest_path = location / "manifest.json"
    original = json.loads(manifest_path.read_text())
    changed = dict(original)
    changed["grace_version"] = "tampered"
    manifest_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(ArtifactError, match="manifest_hash"):
        store.verify_record(location)

    changed["grace_version"] = original["grace_version"]
    changed["artifact_format_version"] = "999"
    rehashed = ArtifactManifest.model_validate(changed).with_integrity_hash()
    manifest_path.write_bytes(canonical_json_bytes(rehashed))
    with pytest.raises(ArtifactError, match="unsupported artifact format"):
        store.verify_record(location)


def test_extra_file_and_wrong_parent_are_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs", mode="checkpoint", run_id="run-extra")
    parent = _initial_state()
    child = _child_state(parent)
    _write_initial(store, state=parent)
    child_path = store.write_state(
        state=child,
        schema=DefaultGraceSchema(),
        result_kind="evolution",
        result_payload=_evolution_payload(parent, child),
        input_hashes={"diagnosis_report": _digest("diagnosis")},
        config_hash=_digest("config"),
    )
    assert child_path is not None
    wrong_parent = GraceState.from_parts(
        graph=parent.graph,
        instruction="A different valid parent instruction.",
        schema_id=parent.schema_id,
        schema_hash=parent.schema_hash,
    )
    with pytest.raises(ArtifactError, match="parent"):
        store.load_state(child_path, parent_state=wrong_parent)

    (child_path / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactError, match="file set mismatch"):
        store.verify_record(child_path)


def test_artifact_mode_cannot_change_inside_one_lineage(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    checkpoint = ArtifactStore(root, mode="checkpoint", run_id="run-fixed-mode")
    parent = _initial_state()
    child = _child_state(parent)
    _write_initial(checkpoint, state=parent)
    audit = ArtifactStore(root, mode="audit", run_id="run-fixed-mode")

    with pytest.raises(ArtifactError, match="mode cannot change"):
        audit.write_state(
            state=child,
            schema=DefaultGraceSchema(),
            result_kind="evolution",
            result_payload=_evolution_payload(parent, child),
            input_hashes={"diagnosis_report": _digest("diagnosis")},
            config_hash=_digest("config"),
            audit_payload={},
        )


def test_interrupted_temp_record_never_becomes_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "runs", mode="checkpoint", run_id="run-interrupt")

    def fail_finalize(**_: object) -> Path:
        raise OSError("injected finalization failure")

    monkeypatch.setattr(store, "_finalize_record", fail_finalize)
    with pytest.raises(ArtifactError, match="injected finalization failure"):
        _write_initial(store)

    assert not (store.states_path / _initial_state().state_id).exists()
    assert not list(store.states_path.glob(".tmp-*"))


def test_multiprocess_create_if_absent_accepts_one_identical_record(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    root = tmp_path / "runs"
    processes = [
        context.Process(target=_concurrent_writer, args=(str(root), start, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    try:
        outcomes = [results.get(timeout=2) for _ in processes]
    except Empty as exc:  # pragma: no cover - diagnostic for a broken child process
        raise AssertionError("concurrent writer did not return a result") from exc
    assert [item[0] for item in outcomes] == ["ok", "ok"]
    assert outcomes[0][1] == outcomes[1][1]

    store = ArtifactStore(root, mode="checkpoint", run_id="run-concurrent")
    record = store.verify_record(_initial_state().state_id)
    assert record.state == _initial_state()
    assert not list(store.states_path.glob(".tmp-*"))


def test_multiprocess_children_from_one_parent_are_distinct_valid_siblings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    store = ArtifactStore(root, mode="checkpoint", run_id="run-siblings")
    parent = _initial_state()
    _write_initial(store, state=parent)

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    suffixes = (" Record outcome A.", " Record outcome B.")
    processes = [
        context.Process(
            target=_concurrent_child_writer,
            args=(str(root), suffix, start, results),
        )
        for suffix in suffixes
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    outcomes = [results.get(timeout=2) for _ in processes]
    assert [item[0] for item in outcomes] == ["ok", "ok"]
    child_ids = {item[2] for item in outcomes}
    assert len(child_ids) == 2
    for child_id in child_ids:
        child = store.load_state(child_id, parent_state=parent)
        assert child.parent_state_id == parent.state_id
        assert child.step == 1
    assert not list(store.states_path.glob(".tmp-*"))


@pytest.mark.parametrize(
    "audit_payload, message",
    [
        ({"api_key": "ordinary-looking-value"}, "forbidden"),
        ({"trace": {"__thought__": "hidden reasoning"}}, "forbidden"),
        ({"raw_model_outputs": ["opaque provider object"]}, "explicit audit policy"),
    ],
)
def test_audit_rejects_credentials_hidden_reasoning_and_unapproved_raw_outputs(
    tmp_path: Path, audit_payload: dict[str, object], message: str
) -> None:
    store = ArtifactStore(tmp_path / "runs", mode="audit", run_id="run-safety")

    with pytest.raises(ArtifactError, match=message):
        _write_initial(store, audit_payload=audit_payload)
    assert not store.root.exists()


def test_configured_canary_is_rejected_without_leaking_its_value(tmp_path: Path) -> None:
    canary = "NONSTANDARD-CANARY-CREDENTIAL"
    store = ArtifactStore(
        tmp_path / "runs",
        mode="audit",
        run_id="run-canary",
        sensitive_values=(canary,),
    )

    with pytest.raises(ArtifactError) as exc_info:
        _write_initial(store, audit_payload={"note": f"unsafe {canary}"})
    assert canary not in str(exc_info.value)
    assert not store.root.exists()


def test_raw_model_outputs_require_explicit_audit_policy(tmp_path: Path) -> None:
    store = ArtifactStore(
        tmp_path / "runs",
        mode="audit",
        run_id="run-raw-policy",
        allow_raw_model_outputs=True,
    )

    location = _write_initial(store, audit_payload={"raw_model_outputs": ["safe output"]})

    assert location is not None
    assert store.verify_record(location).audit_payload == {"raw_model_outputs": ["safe output"]}
