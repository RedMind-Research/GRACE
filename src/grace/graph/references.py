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

"""Context-aware graph node-ID references embedded in node content."""

from __future__ import annotations

import re
from collections.abc import Collection


_PROVIDER_TEMP_ID_RE = re.compile(r"(?<!\w)N_(?:new_?)?\d+(?!\w)")


def _exact_id_pattern(node_ids: Collection[str]) -> re.Pattern[str] | None:
    ordered = sorted(
        {node_id for node_id in node_ids if node_id}, key=lambda item: (-len(item), item)
    )
    if not ordered:
        return None
    alternatives = "|".join(re.escape(node_id) for node_id in ordered)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")


def find_node_id_references(
    content: str,
    node_ids: Collection[str],
) -> tuple[str, ...]:
    """Return real graph IDs and explicit provider-temporary IDs in ``content``.

    Plain, unbound ``N``-number tokens can be legitimate domain language (for
    example, ``N95``).  They are structural references only when they identify
    a node in the current graph.  Underscored forms such as ``N_7`` and
    ``N_new_7`` are reserved provider-temporary references and remain invalid
    even before allocation.
    """

    known_pattern = _exact_id_pattern(node_ids)
    matches: list[tuple[int, int, str]] = []
    if known_pattern is not None:
        matches.extend(
            (match.start(), match.end(), match.group(0))
            for match in known_pattern.finditer(content)
        )
    matches.extend(
        (match.start(), match.end(), match.group(0))
        for match in _PROVIDER_TEMP_ID_RE.finditer(content)
    )
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

    references: list[str] = []
    seen: set[str] = set()
    for _, _, token in matches:
        if token not in seen:
            seen.add(token)
            references.append(token)
    return tuple(references)


def strip_node_id_references(
    content: str,
    references: Collection[str],
) -> str:
    """Remove exactly ``references`` and normalize the resulting whitespace."""

    reference_pattern = _exact_id_pattern(references)
    if reference_pattern is None:
        return content
    without_ids = reference_pattern.sub("", content)
    return re.sub(r"\s+", " ", without_ids).strip()


__all__ = ["find_node_id_references", "strip_node_id_references"]
