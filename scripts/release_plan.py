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

"""Validate the single public GRACE distribution version and release tag."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re

from packaging.version import InvalidVersion, Version


_DISTRIBUTION = "redmind-grace"
_VERSION_FILE = Path("src/grace/_version.py")
_RELEASE_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:rc[1-9][0-9]*)?"
)


@dataclass(frozen=True)
class ReleasePlan:
    """The one distribution qualified by the release workflow."""

    distribution: str
    version: str
    tag: str


def read_version(path: Path) -> str:
    """Read the single literal ``__version__`` assignment without importing code."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    versions: list[str] = []
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in targets
        ):
            continue
        value = statement.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ValueError(f"__version__ must be a string literal: {path}")
        versions.append(value.value)
    if len(versions) != 1:
        raise ValueError(f"expected exactly one __version__ assignment: {path}")
    return versions[0]


def _parse_release_version(value: str, *, source: str) -> str:
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"{source} is not a PEP 440 version: {value}") from error
    if str(parsed) != value:
        raise ValueError(
            f"{source} must use canonical PEP 440 spelling: {value!r} != {str(parsed)!r}"
        )
    if parsed.is_devrelease:
        raise ValueError(f"development versions cannot be release tags: {value}")
    if parsed.local is not None:
        raise ValueError(f"local versions cannot be public release tags: {value}")
    if _RELEASE_VERSION.fullmatch(value) is None:
        raise ValueError(f"{source} must be X.Y.Z or optional X.Y.ZrcN with N >= 1: {value}")
    return value


def create_release_plan(tag: str, *, repository_root: Path) -> ReleasePlan:
    """Return the core build plan and validate ``vX.Y.Z[rcN]`` release tags."""

    source_version = read_version(repository_root / _VERSION_FILE)
    if not tag:
        return ReleasePlan(distribution=_DISTRIBUTION, version=source_version, tag="")

    if not tag.startswith("v"):
        raise ValueError(f"unsupported release tag {tag!r}; expected v<PEP-440-version>")
    tag_version = tag.removeprefix("v")
    if not tag_version:
        raise ValueError(f"release tag has no version: {tag}")
    _parse_release_version(tag_version, source="release tag version")
    if tag_version != source_version:
        raise ValueError(
            f"tag version {tag_version!r} does not match {_VERSION_FILE} ({source_version!r})"
        )
    return ReleasePlan(distribution=_DISTRIBUTION, version=tag_version, tag=tag)


def _write_github_output(path: Path, plan: ReleasePlan) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"distribution={plan.distribution}\n")
        output.write(f"version={plan.version}\n")
        output.write(f"tag={plan.tag}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()

    plan = create_release_plan(arguments.tag, repository_root=arguments.repository_root.resolve())
    if arguments.github_output is not None:
        _write_github_output(arguments.github_output, plan)
    print(
        f"Release plan: distribution={plan.distribution}, version={plan.version}, "
        f"tag={plan.tag or 'working tree'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
