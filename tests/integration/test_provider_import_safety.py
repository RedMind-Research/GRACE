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

"""Fresh-process provider canaries that make no network requests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from grace.providers.base import PromptRequest
from grace.providers.config import GoogleAIStudioConfig
from grace.providers.litellm import LiteLLMProvider


PACKAGE_SRC = Path(__file__).resolve().parents[2] / "src"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_SRC)
    env.pop("LITELLM_MODE", None)
    env.pop("GRACE_DOTENV_CANARY", None)
    return env


def test_importing_grace_does_not_import_litellm_or_load_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GRACE_DOTENV_CANARY=loaded\n", encoding="utf-8")
    code = (
        "import os, sys; import grace; "
        "assert 'litellm' not in sys.modules; "
        "assert os.environ.get('GRACE_DOTENV_CANARY') is None"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_lazy_litellm_import_forces_production_and_does_not_load_dotenv(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("GRACE_DOTENV_CANARY=loaded\n", encoding="utf-8")
    code = """
import os
import socket

def deny_network(*_args, **_kwargs):
    raise AssertionError("LiteLLM import attempted network access")

socket.create_connection = deny_network
socket.socket.connect = deny_network

from grace.providers.litellm import _load_litellm

module = _load_litellm()
assert module.litellm_mode == "PRODUCTION"
assert os.environ.get("GRACE_DOTENV_CANARY") is None
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_offline_adapter_path_succeeds_with_network_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden in offline provider tests")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)

    def completion(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"offline": true}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    provider = LiteLLMProvider(
        GoogleAIStudioConfig(),
        secret_resolver=lambda _name: "offline-placeholder",
        completion_fn=completion,
    )

    result = provider.complete(PromptRequest(system_prompt="system", user_prompt="user"))

    assert result.parsed_content == {"offline": True}
