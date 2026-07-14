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

from pathlib import Path

import pytest

from reproduction.tau2_telecom.artifacts import Tau2TelecomRunContract
from reproduction.tau2_telecom.experiment import Tau2TelecomEvaluationPlan
from reproduction.tau2_telecom.provenance import (
    Tau2TelecomSourceProvenance,
    Tau2TelecomRuntimeComponent,
    Tau2TelecomRuntimeProvenance,
    capture_reproduction_provenance,
    capture_runtime_provenance,
    compute_core_source_hash,
    compute_reproduction_source_hash,
)


def _source_tree(root: Path, *, content: str) -> None:
    (root / "reproduction").mkdir()
    (root / "reproduction" / "__init__.py").write_text("\n", encoding="utf-8")
    source = root / "reproduction" / "tau2_telecom"
    source.mkdir()
    (source / "runner.py").write_text(content, encoding="utf-8")
    requirements = root / "requirements"
    requirements.mkdir()
    (requirements / "reproduction.txt").write_text("tau2==0.1.0\n", encoding="utf-8")


def test_source_hash_changes_with_behavior_source(tmp_path: Path) -> None:
    _source_tree(tmp_path, content="VALUE = 1\n")
    before = compute_reproduction_source_hash(tmp_path)

    (tmp_path / "reproduction" / "tau2_telecom" / "runner.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    assert compute_reproduction_source_hash(tmp_path) != before


def test_reproduction_hash_includes_top_level_package(tmp_path: Path) -> None:
    _source_tree(tmp_path, content="VALUE = 1\n")
    before = compute_reproduction_source_hash(tmp_path)

    (tmp_path / "reproduction" / "__init__.py").write_text(
        '"""Changed import behavior."""\n', encoding="utf-8"
    )

    assert compute_reproduction_source_hash(tmp_path) != before


def test_reproduction_hash_includes_pinned_requirement(tmp_path: Path) -> None:
    _source_tree(tmp_path, content="VALUE = 1\n")
    before = compute_reproduction_source_hash(tmp_path)

    (tmp_path / "requirements" / "reproduction.txt").write_text("tau2==0.2.0\n", encoding="utf-8")

    assert compute_reproduction_source_hash(tmp_path) != before


def test_core_hash_changes_with_imported_package_source(tmp_path: Path) -> None:
    package = tmp_path / "grace"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = compute_core_source_hash(package)

    (package / "engine.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert compute_core_source_hash(package) != before


def test_capture_works_without_git_metadata(tmp_path: Path) -> None:
    _source_tree(tmp_path, content="VALUE = 1\n")

    provenance = capture_reproduction_provenance(tmp_path)

    assert provenance.git_commit is None
    assert provenance.git_dirty is None
    assert provenance.runtime.python_implementation
    assert provenance.runtime.python_version


def test_runtime_capture_includes_locked_model_facing_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locked = {
        "google-auth": "2.48.0",
        "google-cloud-aiplatform": "1.133.0",
        "google-genai": "1.65.0",
        "litellm": "1.84.0",
        "openai": "2.21.0",
    }
    requirement_lines = {
        line.strip()
        for line in (Path(__file__).parents[2] / "requirements" / "reproduction.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    locked_requirements = {f"{distribution}=={version}" for distribution, version in locked.items()}
    locked_requirements.remove("litellm==1.84.0")
    locked_requirements.add("litellm[google]==1.84.0")
    assert locked_requirements <= requirement_lines

    monkeypatch.setattr(
        "reproduction.tau2_telecom.provenance.metadata.version",
        lambda distribution: locked.get(distribution, "0.2.1.dev0"),
    )

    captured = capture_runtime_provenance()
    versions = {item.distribution: item.version for item in captured.dependencies}
    assert {distribution: versions[distribution] for distribution in locked} == locked
    assert versions["tau2"] == "0.2.1.dev0"


def test_runtime_provenance_rejects_case_insensitive_duplicate_names() -> None:
    with pytest.raises(ValueError, match="case-insensitively sorted"):
        Tau2TelecomRuntimeProvenance.create(
            python_implementation="CPython",
            python_version="3.13.2",
            dependencies=(
                Tau2TelecomRuntimeComponent(distribution="PyYAML", version="6.0.3"),
                Tau2TelecomRuntimeComponent(distribution="pyyaml", version="6.0.3"),
            ),
        )


def runtime(version: str = "3.13.2") -> Tau2TelecomRuntimeProvenance:
    return Tau2TelecomRuntimeProvenance.create(
        python_implementation="CPython",
        python_version=version,
        dependencies=(Tau2TelecomRuntimeComponent(distribution="pydantic", version="2.12.5"),),
    )


def test_provenance_changes_run_fingerprint() -> None:
    def contract(core_version: str) -> Tau2TelecomRunContract:
        return Tau2TelecomRunContract.create(
            run_id="provenance-test",
            method="grace",
            seed=1024,
            evaluation=Tau2TelecomEvaluationPlan.parse("offline:0"),
            experiment_config_hash="a" * 64,
            task_selection_hash="b" * 64,
            initial_instruction="instruction",
            provenance=Tau2TelecomSourceProvenance(
                core_version=core_version,
                core_source_hash="d" * 64,
                reproduction_source_hash="c" * 64,
                runtime=runtime(),
            ),
        )

    first = contract("0.1.0")
    second = contract("0.1.1")

    assert first.run_fingerprint != second.run_fingerprint


@pytest.mark.parametrize(
    "provenance",
    [
        Tau2TelecomSourceProvenance(
            core_version="0.1.0",
            core_source_hash="e" * 64,
            reproduction_source_hash="c" * 64,
            runtime=runtime(),
        ),
        Tau2TelecomSourceProvenance(
            core_version="0.1.0",
            core_source_hash="d" * 64,
            reproduction_source_hash="e" * 64,
            runtime=runtime(),
        ),
        Tau2TelecomSourceProvenance(
            core_version="0.1.0",
            core_source_hash="d" * 64,
            reproduction_source_hash="c" * 64,
            runtime=runtime("3.12.9"),
        ),
        Tau2TelecomSourceProvenance(
            core_version="0.1.0",
            core_source_hash="d" * 64,
            reproduction_source_hash="c" * 64,
            runtime=runtime(),
            git_commit="f" * 40,
            git_dirty=False,
        ),
    ],
)
def test_provenance_change_conflicts_with_bound_run(
    tmp_path: Path,
    provenance: Tau2TelecomSourceProvenance,
) -> None:
    from reproduction.tau2_telecom.artifacts import Tau2TelecomArtifactLayout

    def contract(value: Tau2TelecomSourceProvenance) -> Tau2TelecomRunContract:
        return Tau2TelecomRunContract.create(
            run_id="resume-provenance",
            method="grace",
            seed=1024,
            evaluation=Tau2TelecomEvaluationPlan.parse("offline:0"),
            experiment_config_hash="a" * 64,
            task_selection_hash="b" * 64,
            initial_instruction="instruction",
            provenance=value,
        )

    original = Tau2TelecomSourceProvenance(
        core_version="0.1.0",
        core_source_hash="d" * 64,
        reproduction_source_hash="c" * 64,
        runtime=runtime(),
    )
    layout = Tau2TelecomArtifactLayout(tmp_path, "resume-provenance")
    layout.bind_run(contract(original))

    with pytest.raises(ValueError, match="conflicts"):
        layout.bind_run(contract(provenance))
