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

"""Privacy, accounting, and optional-import checks for the tau2 boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from reproduction.tau2_telecom.harness import tau2_interface
from reproduction.tau2_telecom.harness.tau2_interface import (
    Tau2UnavailableError,
    _message_usage,
    _optional_tau2_imports,
    _reward_details,
    _safe_provider_attempt_category,
    installed_tau2_revision,
)


HARNESS_DIR = Path(__file__).parents[2] / "reproduction/tau2_telecom/harness"
_SUBPROCESS_NETWORK_GUARD = """
import socket
def _grace_deny_network(*args, **kwargs):
    del args, kwargs
    raise AssertionError('network access is disabled in this subprocess')
socket.create_connection = _grace_deny_network
socket.getaddrinfo = _grace_deny_network
socket.gethostbyname = _grace_deny_network
socket.socket.connect = _grace_deny_network
socket.socket.connect_ex = _grace_deny_network
socket.socket.sendto = _grace_deny_network
"""


def test_missing_usage_keys_are_unknown_not_numeric_zero() -> None:
    messages = (
        SimpleNamespace(
            role="assistant",
            usage={"completion_tokens": 4},
            cost=0.01,
            tool_calls=(),
        ),
        SimpleNamespace(
            role="user",
            usage={"prompt_tokens": 8, "completion_tokens": 2},
            cost=0.02,
            tool_calls=(),
        ),
    )

    accounting = _message_usage(messages)

    assert accounting.agent_input_tokens is None
    assert accounting.agent_output_tokens is None
    assert accounting.user_input_tokens == 8
    assert accounting.user_output_tokens == 2
    assert accounting.agent_model_turns == 1
    assert accounting.user_model_turns == 1
    assert accounting.model_turns == 2


def test_one_missing_role_usage_makes_the_entire_role_incomplete() -> None:
    messages = (
        SimpleNamespace(
            role="assistant",
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            tool_calls=(),
        ),
        SimpleNamespace(role="assistant", usage=None, tool_calls=()),
    )

    accounting = _message_usage(messages)

    assert accounting.agent_input_tokens is None
    assert accounting.agent_output_tokens is None
    assert accounting.agent_model_turns == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("OpenAI quota exceeded"), "transient_transport"),
        (RuntimeError("429 resource_exhausted"), "transient_transport"),
        (TimeoutError("timed out"), "overall_timeout"),
        (RuntimeError("credential denied"), "authentication"),
        (RuntimeError("assistantmessage must have either content or tool calls"), "empty_response"),
        (RuntimeError("opaque unexpected provider failure"), "transport"),
    ],
)
def test_tau2_provider_failures_map_to_bounded_ledger_categories(
    error: Exception,
    expected: str,
) -> None:
    assert _safe_provider_attempt_category(error) == expected


def test_reward_details_are_a_strict_non_pii_allowlist() -> None:
    secret = "synthetic-customer-phone-555-0199"
    reward_info = SimpleNamespace(
        reward=0.5,
        reward_basis=(SimpleNamespace(value="DB"), SimpleNamespace(value="ACTION")),
        db_check=SimpleNamespace(db_match=True, db_reward=1.0, raw_value=secret),
        env_assertions=(
            SimpleNamespace(met=True, env_assertion=secret, reward=1.0),
            SimpleNamespace(met=False, env_assertion=secret, reward=0.0),
        ),
        action_checks=(SimpleNamespace(action_match=True, action={"phone": secret}),),
        nl_assertions=(SimpleNamespace(met=False, justification=secret, nl_assertion=secret),),
        communicate_checks=(SimpleNamespace(met=True, info=secret, justification=secret),),
        reward_breakdown={"private-key": secret},
        info={"customer": secret},
    )

    details = _reward_details(reward_info)
    rendered = str(details)

    assert secret not in rendered
    assert "info" not in details
    assert "reward_breakdown" not in details
    assert details == {
        "reward": 0.5,
        "reward_basis": ["DB", "ACTION"],
        "checks": {
            "database": {"present": True, "passed": True},
            "environment_assertions": {"present": True, "total": 2, "passed": 1},
            "actions": {"present": True, "total": 1, "passed": 1},
            "natural_language_assertions": {
                "present": True,
                "total": 1,
                "passed": 0,
            },
            "communication": {"present": True, "total": 1, "passed": 1},
        },
    }


def test_importing_harness_does_not_import_tau2_or_litellm() -> None:
    script = (
        _SUBPROCESS_NETWORK_GUARD
        + """
import sys
before = set(sys.modules)
import reproduction.tau2_telecom.harness.tau2_interface
loaded = set(sys.modules) - before
bad = sorted(name for name in loaded if name == 'tau2' or name.startswith('tau2.') or name == 'litellm' or name.startswith('litellm.'))
if bad:
    raise SystemExit('unexpected optional imports: ' + repr(bad))
"""
    )
    env = dict(os.environ)
    grace_root = Path(__file__).parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(grace_root / "src"),
            str(grace_root),
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_tau2_import_rejects_litellm_preloaded_in_development_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(litellm_mode="DEV"))

    with pytest.raises(Tau2UnavailableError, match="outside PRODUCTION"):
        _optional_tau2_imports()


def test_editable_direct_url_file_uri_resolves_to_checkout_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``file:///C:/...`` editable-install URL must map back to a ``C:`` drive path.

    ``Path("/C:/...")`` resolves relative to the drive's cwd on Windows, which
    hid the ``.git`` directory and made the pinned revision unverifiable.
    """

    checkout = tmp_path / "src dir" / "tau2"
    checkout.mkdir(parents=True)
    # ``as_uri`` yields ``file:///C:/...`` on Windows and percent-encodes the
    # space, exercising both the drive-letter and the unquote paths.
    direct_url = json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})
    assert direct_url.startswith('{"url": "file:///')

    fake_distribution = SimpleNamespace(
        read_text=lambda name: direct_url if name == "direct_url.json" else None
    )
    monkeypatch.setattr(tau2_interface.metadata, "distribution", lambda name: fake_distribution)

    seen_roots: list[Path] = []

    def fake_git_revision(root: Path) -> str | None:
        seen_roots.append(root)
        return "deadbeef" if root == checkout.resolve() else None

    monkeypatch.setattr(tau2_interface, "_git_revision", fake_git_revision)

    assert installed_tau2_revision() == "deadbeef"
    assert seen_roots == [checkout.resolve()]


def test_installed_tau2_and_litellm_start_offline_without_dotenv(
    tmp_path: Path,
) -> None:
    try:
        metadata.distribution("tau2")
    except metadata.PackageNotFoundError:
        pytest.skip("optional pinned tau2 reproduction dependency is not installed")

    canary_name = "GRACE_TAU2_DOTENV_CANARY"
    canary_value = "must-not-be-loaded"
    (tmp_path / ".env").write_text(f"{canary_name}={canary_value}\n", encoding="utf-8")
    manifest_path = (
        Path(__file__).parents[2]
        / "reproduction"
        / "tau2_telecom"
        / "data"
        / "experience_smoke_v1.json"
    )
    script = (
        _SUBPROCESS_NETWORK_GUARD
        + f"""
import os
from reproduction.tau2_telecom.harness.tasks import load_task_selection_manifest
from reproduction.tau2_telecom.harness.tau2_interface import Tau2EpisodeExecutor, load_tau2_task_selection, get_tau2_initial_policy
manifest = load_task_selection_manifest({str(manifest_path)!r})
resolved = load_tau2_task_selection(manifest)
policy = get_tau2_initial_policy()
prompt_hash = Tau2EpisodeExecutor().prompt_hash(task=resolved.tasks[0], policy=policy, agent_model='vertex_ai/gemini-2.5-flash', user_model='gpt-4.1-2025-04-14')
assert [task.id for task in resolved.tasks] == [slot.benchmark_task_id for slot in manifest.tasks]
assert len(policy) == 23318
assert len(prompt_hash) == 64
assert os.environ.get({canary_name!r}) is None
print('offline tau2/litellm startup qualified')
"""
    )
    grace_root = Path(__file__).parents[2]
    env = dict(os.environ)
    env.pop(canary_name, None)
    env["LITELLM_MODE"] = "PRODUCTION"
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(grace_root / "src"),
            str(grace_root),
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "offline tau2/litellm startup qualified" in result.stdout


def test_formal_harness_has_no_thread_pool_branch() -> None:
    for filename in (
        "tau2_interface.py",
        "experience.py",
        "evaluation.py",
        "health.py",
        "cost.py",
    ):
        source = (HARNESS_DIR / filename).read_text(encoding="utf-8")
        assert "ThreadPoolExecutor" not in source
