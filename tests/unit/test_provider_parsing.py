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

"""Structured responses use strict JSON and safe failure messages."""

from __future__ import annotations

import pytest

from grace.errors import ProviderResponseError
from grace.providers.parsing import (
    MAX_STRUCTURED_RESPONSE_CHARS,
    parse_json_response,
    require_nonempty_content,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"ok": true}', {"ok": True}),
        ('```json\n{"ok": true}\n```', {"ok": True}),
        ('Result follows: {"ok": true}', {"ok": True}),
        ("prefix [1, 2, 3] suffix", [1, 2, 3]),
    ],
)
def test_parse_supported_strict_json_forms(text: str, expected: object) -> None:
    assert parse_json_response(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "{'python': True}", "not-json"])
def test_invalid_content_raises_without_echoing_input(text: str) -> None:
    with pytest.raises(ProviderResponseError) as captured:
        parse_json_response(text)

    assert text.strip() not in str(captured.value) if text.strip() else True


def test_failure_never_exposes_response_secret() -> None:
    secret = "credential-super-secret"
    with pytest.raises(ProviderResponseError) as captured:
        parse_json_response(f"invalid output {secret}")

    assert secret not in str(captured.value)


def test_nonempty_content_requires_a_string() -> None:
    with pytest.raises(ProviderResponseError, match="empty content"):
        require_nonempty_content({"not": "text"})


def test_structured_response_size_is_bounded() -> None:
    oversized = '"' + ("x" * MAX_STRUCTURED_RESPONSE_CHARS) + '"'

    with pytest.raises(ProviderResponseError, match="size limit"):
        parse_json_response(oversized)
