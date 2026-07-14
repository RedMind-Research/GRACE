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

from pathlib import Path

import pytest

from scripts.release_plan import create_release_plan, read_version


def _repository(tmp_path: Path, *, version: str = "0.1.0") -> Path:
    version_file = tmp_path / "src/grace/_version.py"
    version_file.parent.mkdir(parents=True)
    version_file.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    return tmp_path


def test_non_tag_event_qualifies_the_only_distribution(tmp_path: Path) -> None:
    plan = create_release_plan("", repository_root=_repository(tmp_path))

    assert plan.distribution == "redmind-grace"
    assert plan.version == "0.1.0"
    assert plan.tag == ""


def test_standard_final_tag_selects_redmind_grace(tmp_path: Path) -> None:
    plan = create_release_plan("v0.1.0", repository_root=_repository(tmp_path))

    assert plan.distribution == "redmind-grace"
    assert plan.version == "0.1.0"
    assert plan.tag == "v0.1.0"


def test_release_candidate_is_allowed_but_not_required(tmp_path: Path) -> None:
    repository = _repository(tmp_path, version="0.2.0rc1")

    plan = create_release_plan("v0.2.0rc1", repository_root=repository)

    assert plan.version == "0.2.0rc1"


@pytest.mark.parametrize(
    ("tag", "message"),
    [
        ("v0.1.0.dev0", "development versions cannot"),
        ("v0.1.0+private", "local versions cannot"),
        ("v0.1.0-rc1", "canonical PEP 440"),
        ("v0.1", "must be X.Y.Z"),
        ("v0.1.0a1", "must be X.Y.Z"),
        ("v0.1.0.post1", "must be X.Y.Z"),
        ("vnot-a-version", "not a PEP 440 version"),
        ("release-v0.1.0", "unsupported release tag"),
    ],
)
def test_invalid_public_release_tag_is_rejected(
    tmp_path: Path,
    tag: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_release_plan(tag, repository_root=_repository(tmp_path, version="0.1.0.dev0"))


def test_tag_must_equal_source_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        create_release_plan(
            "v0.1.1",
            repository_root=_repository(tmp_path, version="0.1.0"),
        )


def test_version_reader_rejects_executable_assignment(tmp_path: Path) -> None:
    version_file = tmp_path / "_version.py"
    version_file.write_text("__version__ = compute_version()\n", encoding="utf-8")

    with pytest.raises(ValueError, match="string literal"):
        read_version(version_file)
