"""Pinned sources, the lock file, and integrity-checked downloads.

A source is pinned to an exact commit, and the SHA-256 of its archive is recorded
in ``sources.lock.json`` the first time it is pinned. Every later download must
match that hash, or ingest stops. Changing the pin is a deliberate, reviewed edit
to the lock file.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ALLOWED_DOWNLOAD_HOSTS = frozenset({"codeload.github.com"})
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
_CHUNK_SIZE = 64 * 1024
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_HASH_RE = re.compile(r"[0-9a-f]{64}")


class IntegrityError(Exception):
    """A download did not match its pinned hash, or broke a download rule."""


@dataclass(frozen=True)
class Source:
    """A GitHub repository pinned to one commit."""

    name: str
    repo: str
    ref: str
    commit: str
    archive_sha256: str

    def __post_init__(self) -> None:
        if not _REPO_RE.fullmatch(self.repo):
            raise ValueError(f"invalid repo: {self.repo!r}")
        if not _COMMIT_RE.fullmatch(self.commit):
            raise ValueError("commit must be a full 40-character lowercase hex SHA")
        if self.archive_sha256 and not _HASH_RE.fullmatch(self.archive_sha256):
            raise ValueError("archive_sha256 must be 64 lowercase hex characters")

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.repo}/tar.gz/{self.commit}"


def load_lock(path: Path) -> Source:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Source(**data)


def save_lock(path: Path, source: Source) -> None:
    path.write_text(
        json.dumps(asdict(source), indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise IntegrityError(f"download URL not allowed: {url}")


def download(url: str, dest: Path, max_bytes: int = MAX_ARCHIVE_BYTES) -> str:
    """Stream ``url`` to ``dest`` and return the SHA-256 of the bytes written.

    Only HTTPS URLs on allowlisted hosts are fetched, the final URL after any
    redirect is checked again, and the download stops past ``max_bytes``.
    """
    _check_url(url)
    # Scheme and host are checked by _check_url above.
    request = Request(url, headers={"User-Agent": "rag-support-ingest"})  # noqa: S310
    digest = hashlib.sha256()
    total = 0
    partial = dest.with_suffix(dest.suffix + ".part")
    with urlopen(request, timeout=60) as response:  # noqa: S310
        _check_url(response.geturl())
        with partial.open("wb") as out:
            while chunk := response.read(_CHUNK_SIZE):
                total += len(chunk)
                if total > max_bytes:
                    out.close()
                    partial.unlink()
                    raise IntegrityError(f"download exceeded {max_bytes} bytes")
                digest.update(chunk)
                out.write(chunk)
    partial.replace(dest)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify(actual: str, expected: str) -> None:
    if not expected:
        raise IntegrityError("no pinned hash in the lock file; run `pin` first")
    if actual != expected:
        raise IntegrityError(
            f"archive hash mismatch: expected {expected}, got {actual}. "
            "Ingest stopped. If GitHub regenerated the archive, review the change "
            "and re-pin deliberately."
        )
