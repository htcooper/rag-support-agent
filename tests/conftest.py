from __future__ import annotations

import io
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = "fastapi-0123456789abcdef0123456789abcdef01234567"

# (name, content) for a regular file, or (name, None, linkname) for a symlink.
Member = tuple[str, bytes | None] | tuple[str, None, str]
ArchiveFactory = Callable[[list[Member]], Path]


@pytest.fixture
def make_archive(tmp_path: Path) -> ArchiveFactory:
    def build(members: list[Member]) -> Path:
        path = tmp_path / "archive.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for member in members:
                info = tarfile.TarInfo(member[0])
                if member[1] is None:
                    info.type = tarfile.SYMTYPE
                    info.linkname = member[2]  # type: ignore[misc]
                    tar.addfile(info)
                else:
                    info.size = len(member[1])
                    tar.addfile(info, io.BytesIO(member[1]))
        return path

    return build
