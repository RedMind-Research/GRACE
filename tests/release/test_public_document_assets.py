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

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MARKDOWN_IMAGE = re.compile(r"!\[(?P<alt>[^]]*)\]\((?P<target>[^ )]+)")
_RAW_REPOSITORY_PREFIX = "https://raw.githubusercontent.com/RedMind-Research/GRACE/main/"
_SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


class _HTMLImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "img":
            return
        attributes = dict(attrs)
        self.images.append((attributes.get("alt") or "", attributes.get("src") or ""))


def _public_markdown_files() -> list[Path]:
    roots = [
        _REPOSITORY_ROOT / "README.md",
        _REPOSITORY_ROOT / "CONTRIBUTING.md",
        _REPOSITORY_ROOT / "SECURITY.md",
        _REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md",
        _REPOSITORY_ROOT / "reproduction" / "tau2_telecom" / "README.md",
    ]
    return roots + sorted((_REPOSITORY_ROOT / "docs").glob("*.md"))


def test_public_document_inventory_is_intentional() -> None:
    assert {path.name for path in (_REPOSITORY_ROOT / "docs").glob("*.md")} == {
        "domain-integration.md",
        "evolution-api.md",
        "network-schema.md",
        "providers.md",
    }
    assert not list((_REPOSITORY_ROOT / "assets").glob("*.md"))
    assert (_REPOSITORY_ROOT / "reproduction" / "tau2_telecom" / "README.md").is_file()


def _repository_image_target(markdown: Path, target: str) -> Path | None:
    if target.startswith(_RAW_REPOSITORY_PREFIX):
        return _REPOSITORY_ROOT / target.removeprefix(_RAW_REPOSITORY_PREFIX)
    if urlsplit(target).scheme:
        return None
    return (markdown.parent / target).resolve()


def _image_references(contents: str) -> list[tuple[str, str]]:
    references = [
        (match.group("alt"), match.group("target")) for match in _MARKDOWN_IMAGE.finditer(contents)
    ]
    parser = _HTMLImageParser()
    parser.feed(contents)
    return references + parser.images


def _is_opaque_white(element: ElementTree.Element) -> bool:
    fill = element.attrib.get("fill", "").replace(" ", "").lower()
    opacity = element.attrib.get("fill-opacity", "1")
    return fill in {"#fff", "#ffffff", "white", "rgb(100%,100%,100%)"} and opacity == "1"


def test_public_markdown_images_have_alt_text_and_repository_sources() -> None:
    referenced_repository_assets: set[Path] = set()

    for markdown in _public_markdown_files():
        contents = markdown.read_text(encoding="utf-8")
        for alt_text, image_source in _image_references(contents):
            assert alt_text.strip(), f"image is missing alt text: {markdown}"
            assert image_source.strip(), f"image is missing a source: {markdown}"
            target = _repository_image_target(markdown, image_source)
            if target is None:
                parsed = urlsplit(image_source)
                assert parsed.scheme == "https", f"external image must use HTTPS: {image_source}"
                assert parsed.hostname == "img.shields.io", (
                    f"external image host is not reviewed: {image_source}"
                )
                continue
            assert target.is_relative_to(_REPOSITORY_ROOT), (
                f"image escapes the repository: {markdown} -> {target}"
            )
            assert target.is_file(), f"image target does not exist: {markdown} -> {target}"
            referenced_repository_assets.add(target)

    svg_assets = set((_REPOSITORY_ROOT / "assets").glob("*.svg"))
    assert svg_assets <= referenced_repository_assets


def test_public_svg_assets_are_self_contained_and_accessible() -> None:
    for asset in sorted((_REPOSITORY_ROOT / "assets").glob("*.svg")):
        root = ElementTree.parse(asset).getroot()
        assert root.tag == f"{_SVG_NAMESPACE}svg"
        assert root.attrib.get("role") == "img"

        labelled_by = root.attrib.get("aria-labelledby", "").split()
        assert labelled_by
        identifiers = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
        assert set(labelled_by) <= identifiers

        width = float(root.attrib["width"])
        height = float(root.attrib["height"])
        view_box = [float(value) for value in root.attrib["viewBox"].split()]
        assert width > 0 and height > 0
        assert len(view_box) == 4 and view_box[2] > 0 and view_box[3] > 0

        forbidden = {
            f"{_SVG_NAMESPACE}foreignObject",
            f"{_SVG_NAMESPACE}image",
            f"{_SVG_NAMESPACE}script",
        }
        assert not any(element.tag in forbidden for element in root.iter())
        for element in root.iter():
            for attribute, value in element.attrib.items():
                if attribute.endswith("href"):
                    assert value.startswith("#"), f"external SVG dependency: {asset} -> {value}"


def test_public_svg_assets_have_light_mode_canvas_backgrounds() -> None:
    for name in ("grace-network-schema.svg", "grace-pipeline.svg", "pass3-checkpoints.svg"):
        root = ElementTree.parse(_REPOSITORY_ROOT / "assets" / name).getroot()
        background = next(
            element for element in root if element.attrib.get("id") == "canvas-background"
        )
        view_x, view_y, view_width, view_height = (
            float(value) for value in root.attrib["viewBox"].split()
        )
        background_x = float(background.attrib.get("x", "0"))
        background_y = float(background.attrib.get("y", "0"))
        background_width = float(background.attrib["width"])
        background_height = float(background.attrib["height"])
        assert background_x <= view_x
        assert background_y <= view_y
        assert background_x + background_width >= view_x + view_width
        assert background_y + background_height >= view_y + view_height
        assert _is_opaque_white(background)

    for name in ("grace-network-schema.tex", "pass3-checkpoints.tex"):
        assert "\\pagecolor{white}" in (_REPOSITORY_ROOT / "assets" / name).read_text(
            encoding="utf-8"
        )


def test_public_asset_license_metadata_is_self_contained() -> None:
    for name in ("grace-pipeline.svg", "grace-wordmark.svg"):
        contents = (_REPOSITORY_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: Apache-2.0" in contents
        assert "SPDX-FileCopyrightText: 2026 Dan C. Hsu and Luke Lu" in contents

    wordmark = (_REPOSITORY_ROOT / "assets/grace-wordmark.svg").read_text(encoding="utf-8")
    assert "<text" not in wordmark
    assert "<rect" not in wordmark
    assert "#272226" not in wordmark
    assert "#de8890" in wordmark

    source_url = "https://arxiv.org/abs/2607.09175"
    copyright_text = "SPDX-FileCopyrightText: 2026 Dan C. Hsu and Luke Lu"
    for name in (
        "grace-network-schema.svg",
        "grace-network-schema.tex",
        "pass3-checkpoints.svg",
        "pass3-checkpoints.tex",
    ):
        contents = (_REPOSITORY_ROOT / "assets" / name).read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: CC-BY-4.0" in contents
        assert "SPDX-License-Identifier: Apache-2.0" not in contents
        assert copyright_text in contents
        assert source_url in contents

    notices = (_REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert source_url in notices
    assert "Copyright 2026 Dan C. Hsu and Luke Lu" in notices
    assert "https://creativecommons.org/licenses/by/4.0/" in notices
    assert "changes the source format and presentation" in notices
