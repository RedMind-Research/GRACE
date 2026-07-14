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

import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from grace import qualify


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "src"
_CREDENTIAL_CANARIES = {
    "GEMINI_API_KEY": "QUALIFY-GEMINI-CANARY-DO-NOT-SERIALIZE",
    "OPENAI_API_KEY": "QUALIFY-OPENAI-CANARY-DO-NOT-SERIALIZE",
    "GOOGLE_APPLICATION_CREDENTIALS": "QUALIFY-ADC-CANARY-DO-NOT-SERIALIZE",
}


def _minimal_env(tmp_path: Path, *, pythonpath: str | None = None) -> dict[str, str]:
    env = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": str(tmp_path),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        **_CREDENTIAL_CANARIES,
    }
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    return env


def _sitecustomize_network_denial(directory: Path) -> None:
    directory.mkdir()
    (directory / "sitecustomize.py").write_text(
        """
import socket

def deny(*_args, **_kwargs):
    raise AssertionError("network access is forbidden")

socket.create_connection = deny
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.getaddrinfo = deny
""",
        encoding="utf-8",
    )


def test_offline_report_has_versioned_schema_and_required_checks() -> None:
    report = qualify.run_offline_qualification()

    assert report["schema"] == "grace.offline-qualification-report"
    assert report["schema_version"] == "1"
    assert report["qualification"] == "offline"
    assert report["status"] == "passed"
    assert report["summary"] == {"total": 7, "passed": 7, "failed": 0}
    assert {item["id"] for item in report["checks"]} == {
        "import_no_eager_litellm",
        "default_schema_hash",
        "safe_route_descriptors",
        "grace_config_defaults",
        "custom_ontology_prompt",
        "grace_state_integrity",
        "offline_execution_boundary",
    }
    boundary = next(item for item in report["checks"] if item["id"] == "offline_execution_boundary")
    assert boundary["evidence"] == {"network_attempts": 0, "provider_calls": 0}


def test_cli_isolated_from_network_and_credential_environment(tmp_path: Path) -> None:
    guard = tmp_path / "guard"
    _sitecustomize_network_denial(guard)
    report_path = tmp_path / "offline.json"
    env = _minimal_env(
        tmp_path,
        pythonpath=os.pathsep.join((str(guard), str(SOURCE_ROOT))),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-m",
            "grace.qualify",
            "offline",
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "GRACE offline qualification passed."
    serialized = report_path.read_text(encoding="utf-8")
    assert json.loads(serialized)["status"] == "passed"
    for canary in _CREDENTIAL_CANARIES.values():
        assert canary not in serialized
        assert canary not in completed.stdout
        assert canary not in completed.stderr


def test_failed_check_returns_stable_nonzero_and_does_not_serialize_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive_error = "QUALIFY-PRIVATE-ERROR-CANARY"

    def fail() -> dict[str, object]:
        raise RuntimeError(sensitive_error)

    monkeypatch.setattr(
        qualify,
        "_OFFLINE_CHECKS",
        (("forced_failure", "A deliberately failed test check.", fail),),
    )
    report_path = tmp_path / "failed.json"

    exit_code = qualify.main(["offline", "--report", str(report_path)])

    assert exit_code == qualify.EXIT_QUALIFICATION_FAILED
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["status"] == "failed"
    assert report["checks"][0]["evidence"] == {"error_type": "RuntimeError"}
    captured = capsys.readouterr()
    assert sensitive_error not in report_text + captured.out + captured.err


def test_report_write_error_returns_stable_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_target = tmp_path / "directory-target"
    invalid_target.mkdir()

    exit_code = qualify.main(["offline", "--report", str(invalid_target)])

    assert exit_code == qualify.EXIT_USAGE_OR_REPORT_ERROR
    captured = capsys.readouterr()
    assert captured.err.strip() == "GRACE offline qualification could not produce a report."


def test_atomic_report_replacement_preserves_previous_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "offline.json"
    first = qualify.run_offline_qualification()
    qualify._atomic_write_report(target, first)
    previous = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(qualify.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        qualify._atomic_write_report(target, qualify.run_offline_qualification())

    assert target.read_bytes() == previous
    assert not list(tmp_path.glob(".offline.json.tmp-*"))


def test_wheel_fresh_venv_installed_qualification_without_source_or_network(
    tmp_path: Path,
) -> None:
    """Build, install, import, and qualify the public wheel using local deps only."""

    source_copy = tmp_path / "source"
    shutil.copytree(
        PACKAGE_ROOT,
        source_copy,
        ignore=shutil.ignore_patterns(
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(source_copy),
        ],
        cwd=tmp_path,
        env=_minimal_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    wheels = list(wheelhouse.glob("redmind_grace-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    assert "grace/qualify.py" in members
    assert "grace/py.typed" in members
    assert not any(name.startswith("reproduction/") for name in members)
    assert not any(name.startswith("tests/") for name in members)

    fresh = tmp_path / "fresh-venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(fresh)],
        cwd=tmp_path,
        env=_minimal_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    fresh_python = fresh / "bin" / "python"
    fresh_site = subprocess.run(
        [fresh_python, "-c", "import site; print(site.getsitepackages()[0])"],
        cwd=tmp_path,
        env=_minimal_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    parent_site = next(
        path for path in sys.path if path.endswith("site-packages") and Path(path).is_dir()
    )
    Path(fresh_site, "qualification-local-dependencies.pth").write_text(
        f"{parent_site}\n",
        encoding="utf-8",
    )

    installed = subprocess.run(
        [
            fresh_python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        cwd=tmp_path,
        env=_minimal_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr

    guard = tmp_path / "wheel-guard"
    _sitecustomize_network_denial(guard)
    env = _minimal_env(tmp_path, pythonpath=str(guard))
    import_probe = subprocess.run(
        [
            fresh_python,
            "-c",
            (
                "import json, pathlib, sys, grace; "
                "print(json.dumps({'file': str(pathlib.Path(grace.__file__).resolve()), "
                "'litellm': 'litellm' in sys.modules}))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert import_probe.returncode == 0, import_probe.stderr
    probe = json.loads(import_probe.stdout)
    assert Path(probe["file"]).is_relative_to(fresh)
    assert probe["litellm"] is False

    report_path = tmp_path / "installed-offline.json"
    qualified = subprocess.run(
        [
            fresh_python,
            "-W",
            "error",
            "-m",
            "grace.qualify",
            "offline",
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert qualified.returncode == 0, qualified.stderr
    report_text = report_path.read_text(encoding="utf-8")
    assert json.loads(report_text)["status"] == "passed"
    for canary in _CREDENTIAL_CANARIES.values():
        assert canary not in report_text + qualified.stdout + qualified.stderr
