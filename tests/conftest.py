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

"""Global offline safety boundary for the test suite."""

from __future__ import annotations

import os
import socket
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def deny_network_by_default(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Fail deterministically if an offline test attempts network access."""

    live_requested = request.node.get_closest_marker("live") is not None
    live_enabled = os.environ.get("GRACE_RUN_LIVE_TESTS") == "1"
    if live_requested and live_enabled:
        yield
        return

    def denied(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("network access is disabled in GRACE offline tests")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "gethostbyname", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket.socket, "sendto", denied)
    yield
