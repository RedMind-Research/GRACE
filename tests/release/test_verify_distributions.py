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

from scripts.verify_distributions import verify_source_legal_files, verify_source_license_headers


_REVISION = "c5b2d228d850c59b749b93cf32c4745d3aa53967"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _legal_repository(tmp_path: Path) -> Path:
    (tmp_path / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text("GRACE\n", encoding="utf-8")
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text(
        f"tau2-bench at {_REVISION}\n\nMIT License\n",
        encoding="utf-8",
    )
    return tmp_path


def test_source_legal_files_accept_the_reviewed_repository_contract(tmp_path: Path) -> None:
    verify_source_legal_files(_legal_repository(tmp_path))


def test_project_apache_license_and_notice_have_distinct_roles() -> None:
    license_text = (_REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    notice_text = (_REPOSITORY_ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "APPENDIX: How to apply the Apache License to your work." in license_text
    assert "Copyright [yyyy] [name of copyright owner]" in license_text
    assert notice_text == "GRACE\nCopyright 2026 Dan C. Hsu and Luke Lu\n"


def test_first_party_python_files_have_the_reviewed_apache_header() -> None:
    verify_source_license_headers(_REPOSITORY_ROOT)


def test_source_license_headers_reject_an_unlicensed_python_file(tmp_path: Path) -> None:
    source = tmp_path / "src" / "grace" / "unlicensed.py"
    source.parent.mkdir(parents=True)
    source.write_text("from __future__ import annotations\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing the reviewed Apache header"):
        verify_source_license_headers(tmp_path)


def test_source_legal_files_require_the_pinned_tau2_revision(tmp_path: Path) -> None:
    repository = _legal_repository(tmp_path)
    (repository / "THIRD_PARTY_NOTICES.md").write_text(
        "tau2-bench\n\nMIT License\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviewed tau2 revision"):
        verify_source_legal_files(repository)


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks are unavailable")
def test_source_legal_files_reject_symlinked_inputs(tmp_path: Path) -> None:
    repository = _legal_repository(tmp_path)
    target = repository / "LICENSE.real"
    (repository / "LICENSE").replace(target)
    (repository / "LICENSE").symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        verify_source_legal_files(repository)
