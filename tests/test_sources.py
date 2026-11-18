from __future__ import annotations

from pathlib import Path

import pytest

from rag_support.sources import (
    IntegrityError,
    Source,
    download,
    load_lock,
    save_lock,
    sha256_file,
    verify,
)

SHA = "7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4"
HASH = "a" * 64


def make_source(**overrides: str) -> Source:
    fields = {
        "name": "fastapi",
        "repo": "fastapi/fastapi",
        "ref": "0.162.0",
        "commit": SHA,
        "archive_sha256": HASH,
    }
    fields.update(overrides)
    return Source(**fields)


def test_lock_round_trip(tmp_path: Path) -> None:
    source = make_source()
    lock = tmp_path / "sources.lock.json"
    save_lock(lock, source)
    assert load_lock(lock) == source


def test_archive_url_is_pinned_to_commit() -> None:
    assert make_source().archive_url == (
        f"https://codeload.github.com/fastapi/fastapi/tar.gz/{SHA}"
    )


@pytest.mark.parametrize("commit", ["main", "0.162.0", SHA[:12], SHA.upper()])
def test_rejects_moving_or_short_refs(commit: str) -> None:
    with pytest.raises(ValueError, match="commit"):
        make_source(commit=commit)


def test_rejects_malformed_repo() -> None:
    with pytest.raises(ValueError, match="repo"):
        make_source(repo="fastapi/fastapi/../../evil")


def test_verify_accepts_match() -> None:
    verify(HASH, HASH)


def test_verify_stops_on_mismatch() -> None:
    with pytest.raises(IntegrityError, match="mismatch"):
        verify("b" * 64, HASH)


def test_verify_requires_a_pin() -> None:
    with pytest.raises(IntegrityError, match="no pinned hash"):
        verify(HASH, "")


@pytest.mark.parametrize(
    "url",
    [
        "http://codeload.github.com/fastapi/fastapi/tar.gz/x",
        "https://evil.example.com/fastapi.tar.gz",
        "https://codeload.github.com.evil.example/x",
        "file:///etc/passwd",
    ],
)
def test_download_rejects_disallowed_urls(url: str, tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="not allowed"):
        download(url, tmp_path / "out.tar.gz")


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
