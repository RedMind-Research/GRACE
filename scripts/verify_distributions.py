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

"""Verify the public ``redmind-grace`` distribution and repository boundary."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from email.message import Message
from email.parser import Parser
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


_DISTRIBUTION = "redmind-grace"
_PROJECT_AUTHORS = "Dan C. Hsu, Luke Lu"
_TAU2_REVISION = "c5b2d228d850c59b749b93cf32c4745d3aa53967"
_FIRST_PARTY_PYTHON_ROOTS = (
    "src/grace",
    "reproduction",
    "examples",
    "scripts",
    "tests",
)
_APACHE_SOURCE_HEADER = """# Copyright 2026 Dan C. Hsu and Luke Lu
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

""".encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _wheel(path: Path) -> tuple[set[str], Message, str, str]:
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError(f"not a wheel: {path}")
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        metadata_files = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError(f"wheel must contain exactly one METADATA file: {path}")
        metadata_path = metadata_files[0]
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
        entry_files = [name for name in members if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_files) > 1:
            raise ValueError(f"wheel contains multiple entry_points.txt files: {path}")
        entries = archive.read(entry_files[0]).decode("utf-8") if entry_files else ""
    return members, metadata, entries, metadata_path.rsplit("/", 1)[0] + "/"


def _require_license_metadata(metadata: Message) -> None:
    _require(metadata["Metadata-Version"] == "2.4", "distribution metadata must use version 2.4")
    _require(metadata["Name"] == _DISTRIBUTION, f"unexpected distribution: {metadata['Name']}")
    _require(
        metadata["Author"] == _PROJECT_AUTHORS,
        "distribution authors differ from the reviewed joint attribution",
    )
    _require(
        metadata["License-Expression"] == "Apache-2.0",
        "distribution must declare the reviewed Apache-2.0 SPDX expression",
    )
    _require(metadata.get("License") is None, "legacy free-text License metadata is forbidden")
    _require(
        set(metadata.get_all("License-File", [])) == {"LICENSE", "NOTICE"},
        "distribution license-file metadata differs from the reviewed set",
    )
    _require(
        not any(
            classifier.startswith("License ::") for classifier in metadata.get_all("Classifier", [])
        ),
        "deprecated license classifiers are forbidden when License-Expression is present",
    )


def _require_wheel_license_files(*, members: set[str], dist_info_prefix: str) -> None:
    prefix = f"{dist_info_prefix}licenses/"
    packaged = {name.removeprefix(prefix) for name in members if name.startswith(prefix)}
    _require(
        packaged == {"LICENSE", "NOTICE"},
        "wheel license payload differs from the reviewed set",
    )


def _requirements(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path) as archive:
        metadata_file = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        message = Parser().parsestr(archive.read(metadata_file).decode("utf-8"))
    return tuple(message.get_all("Requires-Dist", []))


def _requirement_names(requirements: tuple[str, ...]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        if match is None:
            raise ValueError(f"cannot parse requirement: {requirement}")
        names.add(re.sub(r"[-_.]+", "-", match.group(0)).lower())
    return names


def _require_wheel_allowlist(
    *,
    members: set[str],
    package_prefix: str,
    dist_info_prefix: str,
    allowed_package_member: Callable[[str], bool],
) -> None:
    unexpected_roots = sorted(
        name
        for name in members
        if not name.startswith(package_prefix) and not name.startswith(dist_info_prefix)
    )
    _require(
        not unexpected_roots, f"wheel contains unexpected top-level members: {unexpected_roots}"
    )
    unexpected_package = sorted(
        name
        for name in members
        if name.startswith(package_prefix) and not allowed_package_member(name)
    )
    _require(
        not unexpected_package,
        f"wheel contains unexpected package members: {unexpected_package}",
    )


def verify_core_wheel(path: Path) -> None:
    """Verify metadata, dependencies, and package contents in the public wheel."""

    members, metadata, entries, dist_info = _wheel(path)
    requirements = _requirements(path)
    _require_license_metadata(metadata)
    _require_wheel_license_files(members=members, dist_info_prefix=dist_info)
    _require_wheel_allowlist(
        members=members,
        package_prefix="grace/",
        dist_info_prefix=dist_info,
        allowed_package_member=lambda name: name.endswith(".py") or name == "grace/py.typed",
    )
    _require("grace/__init__.py" in members, "core package is missing grace/__init__.py")
    _require("grace/qualify.py" in members, "core package is missing offline qualification")
    _require("grace/py.typed" in members, "core package is missing its typing marker")
    _require(not entries.strip(), "core wheel must not expose console scripts")
    _require(
        _requirement_names(requirements)
        == {
            "build",
            "hypothesis",
            "jsonschema",
            "litellm",
            "mypy",
            "pydantic",
            "pytest",
            "pytest-cov",
            "pyyaml",
            "ruff",
            "setuptools",
            "types-jsonschema",
            "types-pyyaml",
        },
        "core wheel dependency names differ from the reviewed allowlist",
    )
    litellm = [
        requirement for requirement in requirements if requirement.lower().startswith("litellm")
    ]
    _require(len(litellm) == 1, "core wheel must declare LiteLLM exactly once")
    _require('extra == "gemini"' in litellm[0], "LiteLLM must be gated by the gemini extra")


def _sdist_files(path: Path) -> set[str]:
    if not path.is_file() or not path.name.endswith(".tar.gz"):
        raise ValueError(f"not a .tar.gz sdist: {path}")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    _require(
        all(member.isfile() or member.isdir() for member in members),
        f"sdist contains links or special files: {path}",
    )
    paths = [PurePosixPath(member.name) for member in members]
    _require(
        all(not item.is_absolute() and ".." not in item.parts for item in paths),
        f"sdist contains an unsafe member path: {path}",
    )
    roots = {item.parts[0] for item in paths if item.parts}
    _require(len(roots) == 1, f"sdist must contain exactly one root directory: {path}")
    files = {
        "/".join(item.parts[1:])
        for item, member in zip(paths, members, strict=True)
        if member.isfile()
    }
    _require("" not in files, f"sdist contains a file at its archive root: {path}")
    return files


def _sdist_metadata(path: Path) -> Message:
    with tarfile.open(path, "r:gz") as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and len(PurePosixPath(member.name).parts) == 2
            and PurePosixPath(member.name).name == "PKG-INFO"
        ]
        _require(len(candidates) == 1, f"sdist must contain exactly one PKG-INFO file: {path}")
        extracted = archive.extractfile(candidates[0])
        if extracted is None:
            raise ValueError(f"cannot read sdist metadata: {path}")
        return Parser().parsestr(extracted.read().decode("utf-8"))


def _core_sdist_member_allowed(name: str) -> bool:
    root_files = {
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "pyproject.toml",
        "setup.cfg",
    }
    if name in root_files:
        return True
    if name.startswith("assets/"):
        return name in {"assets/grace-pipeline.svg", "assets/grace-wordmark.svg"}
    if name.startswith("docs/"):
        return name.endswith(".md") and not name.endswith(".local.md")
    if name.startswith("examples/"):
        return name.endswith(".py")
    if name in {"requirements/gemini.txt", "requirements/minimum.txt"}:
        return True
    if name.startswith("src/grace/"):
        return name.endswith(".py") or name == "src/grace/py.typed"
    if name.startswith("src/redmind_grace.egg-info/"):
        return True
    if name == "tests/conftest.py":
        return True
    return (
        name.startswith("tests/integration/") or name.startswith("tests/unit/")
    ) and name.endswith(".py")


def verify_core_sdist(path: Path) -> None:
    """Verify metadata and the product/reproduction boundary in the public sdist."""

    files = _sdist_files(path)
    _require_license_metadata(_sdist_metadata(path))
    unexpected = sorted(name for name in files if not _core_sdist_member_allowed(name))
    _require(not unexpected, f"core sdist contains unexpected members: {unexpected}")
    _require("src/grace/__init__.py" in files, "core sdist is missing the core package")
    _require(
        {"LICENSE", "NOTICE"}.issubset(files),
        "core sdist is missing its legal files",
    )
    _require(
        not any(
            name.startswith("packages/")
            or name.startswith("reproduction/")
            or name.startswith("tests/reproduction/")
            for name in files
        ),
        "core sdist crosses the repository-only reproduction boundary",
    )


def _require_regular_file(path: Path, *, description: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{description} must be a regular file")


def verify_source_license_headers(repository_root: Path) -> None:
    """Require the reviewed Apache header on every first-party Python file."""

    root = repository_root.resolve()
    for relative_root in _FIRST_PARTY_PYTHON_ROOTS:
        source_root = root / relative_root
        if not source_root.exists():
            continue
        _require(
            source_root.is_dir() and not source_root.is_symlink(),
            f"first-party source root must be a regular directory: {relative_root}",
        )
        for path in sorted(source_root.rglob("*.py")):
            relative_path = path.relative_to(root)
            _require_regular_file(path, description=f"first-party source {relative_path}")
            _require(
                path.read_bytes().startswith(_APACHE_SOURCE_HEADER),
                f"first-party source is missing the reviewed Apache header: {relative_path}",
            )


def verify_source_legal_files(repository_root: Path) -> None:
    """Verify reviewed legal inputs before any archive is built."""

    root = repository_root.resolve()
    for name in ("LICENSE", "NOTICE"):
        _require_regular_file(root / name, description=f"root {name}")
    third_party = root / "THIRD_PARTY_NOTICES.md"
    _require_regular_file(third_party, description="root third-party notice")
    notice = third_party.read_text(encoding="utf-8")
    _require("MIT License" in notice, "third-party notice is missing the tau2 MIT license")
    _require(
        _TAU2_REVISION in notice,
        "third-party notice is missing the reviewed tau2 revision",
    )
    verify_source_license_headers(root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-wheel", type=Path, required=True)
    parser.add_argument("--core-sdist", type=Path, required=True)
    arguments = parser.parse_args()

    verify_source_legal_files(Path.cwd())
    verify_core_wheel(arguments.core_wheel)
    verify_core_sdist(arguments.core_sdist)
    print("redmind-grace distribution boundary verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
