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

"""Normalize source distributions into reproducible ``.tar.gz`` archives.

The input archive is treated as untrusted.  It is validated in full before a
replacement is written, and the original is replaced atomically only after the
normalized archive has been flushed successfully.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile


_MAX_GZIP_EPOCH = (1 << 32) - 1


class UnsafeSdistError(ValueError):
    """Raised when an sdist does not satisfy the safe archive contract."""


@dataclass(frozen=True)
class _ArchiveMember:
    name: str
    is_directory: bool
    content: bytes


def _canonical_name(member: tarfile.TarInfo) -> str:
    name = member.name
    if not name or "\x00" in name or "\\" in name:
        raise UnsafeSdistError(f"unsafe archive member path: {name!r}")

    # A single trailing slash is conventional for directories.  Every other
    # spelling must already be canonical so that aliases cannot bypass duplicate
    # or root checks.
    candidate = name[:-1] if member.isdir() and name.endswith("/") else name
    path = PurePosixPath(candidate)
    if (
        not candidate
        or path.is_absolute()
        or not path.parts
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != candidate
    ):
        raise UnsafeSdistError(f"unsafe archive member path: {name!r}")
    return candidate


def _read_validated_members(path: Path) -> list[_ArchiveMember]:
    if not path.name.endswith(".tar.gz"):
        raise ValueError(f"source distribution must end in .tar.gz: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source distribution is not a regular file: {path}")

    members: list[_ArchiveMember] = []
    roots: set[str] = set()
    seen: dict[str, bool] = {}
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise UnsafeSdistError(
                        f"archive contains a link or special file: {member.name!r}"
                    )
                name = _canonical_name(member)
                if name in seen:
                    raise UnsafeSdistError(f"archive contains a duplicate member: {name!r}")

                parts = PurePosixPath(name).parts
                roots.add(parts[0])
                is_directory = member.isdir()
                seen[name] = is_directory
                if is_directory:
                    content = b""
                else:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise UnsafeSdistError(f"cannot read archive member: {name!r}")
                    content = extracted.read()
                    if len(content) != member.size:
                        raise UnsafeSdistError(f"archive member is truncated: {name!r}")
                members.append(
                    _ArchiveMember(name=name, is_directory=is_directory, content=content)
                )
    except (tarfile.TarError, OSError, EOFError) as error:
        raise UnsafeSdistError(f"cannot read source distribution {path}: {error}") from error

    if len(roots) != 1:
        raise UnsafeSdistError(f"archive must contain exactly one root directory: {path}")
    root = next(iter(roots))
    if seen.get(root) is not True:
        raise UnsafeSdistError(f"archive root must be an explicit directory: {root!r}")

    for name in seen:
        parent = PurePosixPath(name).parent
        while parent.parts:
            parent_name = parent.as_posix()
            if parent_name in seen and not seen[parent_name]:
                raise UnsafeSdistError(f"archive member has a regular file as its parent: {name!r}")
            parent = parent.parent
    return sorted(members, key=lambda member: member.name)


def _normalized_tar_info(member: _ArchiveMember, *, epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(member.name)
    info.type = tarfile.DIRTYPE if member.is_directory else tarfile.REGTYPE
    info.mode = 0o755 if member.is_directory else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = epoch
    info.size = 0 if member.is_directory else len(member.content)
    info.linkname = ""
    info.devmajor = 0
    info.devminor = 0
    info.pax_headers = {}
    return info


def _write_normalized_archive(path: Path, members: list[_ArchiveMember], *, epoch: int) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                    pax_headers={},
                ) as archive:
                    for member in members:
                        info = _normalized_tar_info(member, epoch=epoch)
                        if member.is_directory:
                            archive.addfile(info)
                        else:
                            archive.addfile(info, fileobj=BytesIO(member.content))
            raw.flush()
            os.fsync(raw.fileno())
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some supported filesystems do not permit fsync on directory handles.
        pass
    finally:
        os.close(descriptor)


def normalize_sdist(path: Path, *, epoch: int) -> None:
    """Safely replace ``path`` with a byte-reproducible source distribution."""

    if not 0 <= epoch <= _MAX_GZIP_EPOCH:
        raise ValueError(f"epoch must be between 0 and {_MAX_GZIP_EPOCH}: {epoch}")
    absolute_path = path.expanduser().absolute()
    members = _read_validated_members(absolute_path)
    _write_normalized_archive(absolute_path, members, epoch=epoch)


def _epoch(value: str) -> int:
    try:
        epoch = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch must be an integer") from error
    if not 0 <= epoch <= _MAX_GZIP_EPOCH:
        raise argparse.ArgumentTypeError(f"epoch must be between 0 and {_MAX_GZIP_EPOCH}")
    return epoch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize one or more source distributions reproducibly in place."
    )
    parser.add_argument("archives", nargs="+", type=Path, metavar="SDIST.tar.gz")
    parser.add_argument("--epoch", required=True, type=_epoch)
    arguments = parser.parse_args()

    for archive in arguments.archives:
        normalize_sdist(archive, epoch=arguments.epoch)
        print(f"Normalized source distribution: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
