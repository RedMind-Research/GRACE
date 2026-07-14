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

"""Source provenance for repository-only Tau2 Telecom reproductions."""

from __future__ import annotations

import hashlib
from importlib import metadata
import platform
import re
import subprocess
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

import grace

from .harness.identity import canonical_sha256, validate_sha256


Frozen = ConfigDict(frozen=True, extra="forbid")
NonEmpty = Annotated[str, Field(min_length=1)]
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_SOURCE_SUFFIXES = frozenset({".json", ".py", ".txt", ".yaml", ".yml"})
_RUNTIME_DISTRIBUTIONS = (
    "google-auth",
    "google-cloud-aiplatform",
    "google-genai",
    "jsonschema",
    "litellm",
    "openai",
    "pydantic",
    "PyYAML",
    "tau2",
)


class Tau2TelecomRuntimeComponent(BaseModel):
    """One selected runtime dependency and its installed version."""

    model_config = Frozen

    distribution: NonEmpty
    version: NonEmpty


class Tau2TelecomRuntimeProvenance(BaseModel):
    """Python and direct execution-stack versions bound to a resumable run."""

    model_config = Frozen

    schema_version: Literal["grace.tau2-telecom-runtime.v1"] = "grace.tau2-telecom-runtime.v1"
    python_implementation: NonEmpty
    python_version: NonEmpty
    dependencies: tuple[Tau2TelecomRuntimeComponent, ...]
    runtime_hash: NonEmpty

    @classmethod
    def create(
        cls,
        *,
        python_implementation: str,
        python_version: str,
        dependencies: tuple[Tau2TelecomRuntimeComponent, ...],
    ) -> Tau2TelecomRuntimeProvenance:
        payload = {
            "python_implementation": python_implementation,
            "python_version": python_version,
            "dependencies": [item.model_dump(mode="json") for item in dependencies],
        }
        return cls(
            python_implementation=python_implementation,
            python_version=python_version,
            dependencies=dependencies,
            runtime_hash=canonical_sha256(
                {"namespace": "grace.tau2-telecom-runtime-fingerprint.v1", **payload}
            ),
        )

    @model_validator(mode="after")
    def validate_runtime(self) -> Tau2TelecomRuntimeProvenance:
        validate_sha256(self.runtime_hash, field="runtime_hash")
        names = tuple(item.distribution for item in self.dependencies)
        normalized_names = tuple(name.casefold() for name in names)
        if normalized_names != tuple(sorted(set(normalized_names))):
            raise ValueError("runtime dependencies must be unique and case-insensitively sorted")
        payload = self.model_dump(mode="json", exclude={"schema_version", "runtime_hash"})
        expected = canonical_sha256(
            {"namespace": "grace.tau2-telecom-runtime-fingerprint.v1", **payload}
        )
        if self.runtime_hash != expected:
            raise ValueError("runtime_hash does not match runtime provenance")
        return self


class Tau2TelecomSourceProvenance(BaseModel):
    """Exact software identity attached to every resumable run."""

    model_config = Frozen

    schema_version: Literal["grace.tau2-telecom-source.v1"] = "grace.tau2-telecom-source.v1"
    core_version: NonEmpty
    core_source_hash: NonEmpty
    reproduction_source_hash: NonEmpty
    runtime: Tau2TelecomRuntimeProvenance
    git_commit: str | None = None
    git_dirty: bool | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> Tau2TelecomSourceProvenance:
        for name in ("core_source_hash", "reproduction_source_hash"):
            validate_sha256(getattr(self, name), field=name)
        if self.git_commit is not None and not _GIT_COMMIT.fullmatch(self.git_commit):
            raise ValueError("git_commit must be a full lowercase Git object ID")
        if (self.git_commit is None) != (self.git_dirty is None):
            raise ValueError("git_commit and git_dirty must be available together")
        return self


def _source_tree_hash(
    *,
    root: Path,
    paths: list[Path],
    namespace: str,
) -> str:
    files: dict[str, str] = {}
    for path in paths:
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError(f"no behavior-bearing sources were found under {root}")
    return canonical_sha256({"files": files, "namespace": namespace})


def compute_core_source_hash(package_root: str | Path | None = None) -> str:
    """Hash the exact behavior-bearing files in the imported ``grace`` package."""

    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(grace.__file__).resolve().parent
    )
    if not root.is_dir() or not (root / "__init__.py").is_file():
        raise ValueError(f"GRACE package source is missing: {root}")
    return _source_tree_hash(
        root=root.parent,
        paths=sorted(root.rglob("*")),
        namespace="grace.core-source-tree.v1",
    )


def compute_reproduction_source_hash(repo_root: str | Path) -> str:
    """Hash every behavior-bearing reproduction source and pinned requirement file."""

    root = Path(repo_root).expanduser().resolve()
    source_root = root / "reproduction" / "tau2_telecom"
    if not source_root.is_dir():
        raise ValueError(f"Tau2 Telecom reproduction source is missing: {source_root}")

    paths = list(source_root.rglob("*"))
    paths.append(root / "reproduction" / "__init__.py")
    requirement = root / "requirements" / "reproduction.txt"
    if requirement.is_file():
        paths.append(requirement)
    return _source_tree_hash(
        root=root,
        paths=sorted(paths),
        namespace="grace.tau2-telecom-source-tree.v1",
    )


def _git_identity(repo_root: Path) -> tuple[str | None, bool | None]:
    """Return Git identity only when ``repo_root`` is itself the worktree root."""

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    try:
        worktree = Path(run("rev-parse", "--show-toplevel")).resolve()
        if worktree != repo_root:
            return None, None
        commit = run("rev-parse", "HEAD").lower()
        dirty = bool(
            run(
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "reproduction",
                "requirements/reproduction.txt",
            )
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None
    if not _GIT_COMMIT.fullmatch(commit):
        return None, None
    return commit, dirty


def capture_runtime_provenance() -> Tau2TelecomRuntimeProvenance:
    """Capture Python and selected direct dependencies used by the live workflow."""

    dependencies: list[Tau2TelecomRuntimeComponent] = []
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
        dependencies.append(Tau2TelecomRuntimeComponent(distribution=distribution, version=version))
    dependencies.sort(key=lambda item: item.distribution.casefold())
    return Tau2TelecomRuntimeProvenance.create(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        dependencies=tuple(dependencies),
    )


def capture_reproduction_provenance(
    repo_root: str | Path | None = None,
) -> Tau2TelecomSourceProvenance:
    """Capture the installed core version and exact repository reproduction state."""

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    commit, dirty = _git_identity(root)
    return Tau2TelecomSourceProvenance(
        core_version=grace.__version__,
        core_source_hash=compute_core_source_hash(),
        reproduction_source_hash=compute_reproduction_source_hash(root),
        runtime=capture_runtime_provenance(),
        git_commit=commit,
        git_dirty=dirty,
    )


__all__ = [
    "Tau2TelecomRuntimeComponent",
    "Tau2TelecomRuntimeProvenance",
    "Tau2TelecomSourceProvenance",
    "capture_reproduction_provenance",
    "capture_runtime_provenance",
    "compute_core_source_hash",
    "compute_reproduction_source_hash",
]
