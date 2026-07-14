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

import gzip
from io import BytesIO
from pathlib import Path
import tarfile

import pytest

from scripts.normalize_sdist import UnsafeSdistError, normalize_sdist


_EPOCH = 1_700_000_000


def _member(
    name: str,
    *,
    content: bytes | None = None,
    member_type: bytes | None = None,
    metadata_variant: int = 0,
) -> tuple[tarfile.TarInfo, bytes | None]:
    info = tarfile.TarInfo(name)
    if member_type is not None:
        info.type = member_type
    elif content is None:
        info.type = tarfile.DIRTYPE
    else:
        info.type = tarfile.REGTYPE
    info.mode = 0o700 if metadata_variant else 0o775
    info.uid = 1000 + metadata_variant
    info.gid = 2000 + metadata_variant
    info.uname = f"user-{metadata_variant}"
    info.gname = f"group-{metadata_variant}"
    info.mtime = 100 + metadata_variant
    info.pax_headers = {"comment": f"variant-{metadata_variant}"}
    if content is not None:
        info.size = len(content)
    if member_type == tarfile.SYMTYPE:
        info.linkname = "../outside"
    return info, content


def _write_archive(
    path: Path,
    members: list[tuple[tarfile.TarInfo, bytes | None]],
    *,
    gzip_mtime: int,
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename=path.name,
            mode="wb",
            compresslevel=1,
            fileobj=raw,
            mtime=gzip_mtime,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for info, content in members:
                    archive.addfile(info, BytesIO(content) if content is not None else None)


def _safe_members(
    *, variant: int, reverse: bool = False
) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    members = [
        _member("package-0.1.0", metadata_variant=variant),
        _member("package-0.1.0/src", metadata_variant=variant),
        _member(
            "package-0.1.0/src/module.py",
            content=b'VALUE = "preserved"\n',
            metadata_variant=variant,
        ),
        _member(
            "package-0.1.0/README.md",
            content=b"# Package\n\x00binary-safe\n",
            metadata_variant=variant,
        ),
    ]
    return list(reversed(members)) if reverse else members


def test_different_archive_metadata_normalizes_to_identical_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_archive(first, _safe_members(variant=1), gzip_mtime=111)
    _write_archive(second, _safe_members(variant=9, reverse=True), gzip_mtime=999)

    normalize_sdist(first, epoch=_EPOCH)
    normalize_sdist(second, epoch=_EPOCH)

    assert first.read_bytes() == second.read_bytes()
    raw = first.read_bytes()
    assert raw[3] & gzip.FNAME == 0
    assert int.from_bytes(raw[4:8], "little") == _EPOCH

    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(member.name for member in members)
        for member in members:
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""
            assert member.mtime == _EPOCH
            assert member.pax_headers == {}
            assert member.mode == (0o755 if member.isdir() else 0o644)


def test_normalization_preserves_file_contents_and_structure(tmp_path: Path) -> None:
    archive_path = tmp_path / "package.tar.gz"
    _write_archive(archive_path, _safe_members(variant=3), gzip_mtime=321)

    normalize_sdist(archive_path, epoch=_EPOCH)

    with tarfile.open(archive_path, "r:gz") as archive:
        assert {member.name for member in archive.getmembers()} == {
            "package-0.1.0",
            "package-0.1.0/README.md",
            "package-0.1.0/src",
            "package-0.1.0/src/module.py",
        }
        readme = archive.extractfile("package-0.1.0/README.md")
        module = archive.extractfile("package-0.1.0/src/module.py")
        assert readme is not None
        assert module is not None
        assert readme.read() == b"# Package\n\x00binary-safe\n"
        assert module.read() == b'VALUE = "preserved"\n'


@pytest.mark.parametrize(
    "members",
    [
        [
            _member("package-0.1.0"),
            _member("package-0.1.0/link", member_type=tarfile.SYMTYPE),
        ],
        [
            _member("package-0.1.0"),
            _member("package-0.1.0/../escape", content=b"escape"),
        ],
        [
            _member("package-0.1.0"),
            _member("/package-0.1.0/escape", content=b"escape"),
        ],
        [
            _member("package-0.1.0"),
            _member("other-0.1.0"),
        ],
        [
            _member("package-0.1.0"),
            _member("package-0.1.0/data", member_type=tarfile.FIFOTYPE),
        ],
    ],
    ids=["symlink", "traversal", "absolute", "multiple-roots", "special-file"],
)
def test_unsafe_archives_are_rejected_without_replacing_original(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes | None]],
) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    _write_archive(archive_path, members, gzip_mtime=123)
    original = archive_path.read_bytes()

    with pytest.raises(UnsafeSdistError):
        normalize_sdist(archive_path, epoch=_EPOCH)

    assert archive_path.read_bytes() == original
