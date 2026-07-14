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

"""Strict, bounded parsing for structured provider responses."""

from __future__ import annotations

import json
import re

from pydantic import JsonValue

from grace.errors import ProviderResponseError


_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
MAX_STRUCTURED_RESPONSE_CHARS = 4_000_000


def require_nonempty_content(value: object) -> str:
    """Return stripped text or raise a safe response error."""

    if not isinstance(value, str) or not value.strip():
        raise ProviderResponseError("provider returned empty content")
    return value.strip()


def _loads(candidate: str) -> JsonValue | None:
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        return None


def parse_json_response(text: str) -> JsonValue:
    """Parse strict JSON without exposing response text in failures.

    Direct JSON is preferred.  Markdown fences and one bounded surrounding-text
    extraction are accepted for compatibility with model-generated structured
    output.  JSON5, Python literals, and ``eval`` are intentionally unsupported.
    """

    content = require_nonempty_content(text)
    if len(content) > MAX_STRUCTURED_RESPONSE_CHARS:
        raise ProviderResponseError("provider structured response exceeded the size limit")
    direct = _loads(content)
    if direct is not None:
        return direct

    for match in _FENCED_JSON.finditer(content):
        parsed = _loads(match.group(1).strip())
        if parsed is not None:
            return parsed

    candidates: list[tuple[int, str]] = []
    for opening, closing in (("{", "}"), ("[", "]")):
        start = content.find(opening)
        end = content.rfind(closing)
        if start >= 0 and end > start:
            candidates.append((end - start, content[start : end + 1]))
    for _length, candidate in sorted(candidates, reverse=True):
        parsed = _loads(candidate)
        if parsed is not None:
            return parsed

    raise ProviderResponseError("provider response was not valid JSON")


__all__ = [
    "MAX_STRUCTURED_RESPONSE_CHARS",
    "parse_json_response",
    "require_nonempty_content",
]
