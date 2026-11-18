"""Safe extraction of allowlisted files from an untrusted tar.gz archive.

Fails closed: a link, special file, path traversal, or broken size cap inside the
allowlisted path stops extraction rather than being skipped silently.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ExtractionError(Exception):
    """The archive broke an extraction rule."""


@dataclass(frozen=True)
class ExtractLimits:
    max_files: int = 2_000
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 50_000_000
    # Cap on everything the archive claims to contain, selected or not.
    max_scanned_bytes: int = 1_000_000_000


# (path prefix, file suffix) pairs that may be extracted.
DEFAULT_RULES: tuple[tuple[str, str], ...] = (
    ("docs/en/docs/", ".md"),
    ("docs_src/", ".py"),
)


def _relative_name(member_name: str) -> PurePosixPath | None:
    """Strip the archive's top-level folder (``repo-<sha>/``)."""
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ExtractionError(f"unsafe path in archive: {member_name!r}")
    if len(path.parts) < 2:
        return None
    return PurePosixPath(*path.parts[1:])


def safe_extract(
    archive: Path,
    dest: Path,
    rules: tuple[tuple[str, str], ...] = DEFAULT_RULES,
    limits: ExtractLimits | None = None,
) -> dict[str, Path]:
    """Extract files matching a (prefix, suffix) rule into an empty ``dest``.

    Returns a map of path-relative-to-repo (e.g. ``docs/en/docs/index.md``) to the
    extracted file.
    """
    limits = limits or ExtractLimits()
    if dest.exists() and any(dest.iterdir()):
        raise ExtractionError(f"destination is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    selected: list[tarfile.TarInfo] = []
    rel_names: dict[str, str] = {}
    scanned = 0
    total = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            scanned += member.size
            if scanned > limits.max_scanned_bytes:
                raise ExtractionError("archive exceeds the scanned-size cap")
            rel = _relative_name(member.name)
            if rel is None or member.isdir():
                continue
            rule = next((r for r in rules if rel.as_posix().startswith(r[0])), None)
            if rule is None:
                continue
            if not member.isfile():
                raise ExtractionError(f"link or special file in docs: {member.name!r}")
            if rel.suffix != rule[1]:
                continue
            if member.size > limits.max_file_bytes:
                raise ExtractionError(f"file exceeds size cap: {member.name!r}")
            total += member.size
            if total > limits.max_total_bytes:
                raise ExtractionError("selected files exceed the total-size cap")
            if len(selected) >= limits.max_files:
                raise ExtractionError("selected files exceed the file-count cap")
            selected.append(member)
            rel_names[member.name] = rel.as_posix()

        try:
            # The "data" filter rejects absolute paths, traversal, links that
            # escape dest, device files, and unsafe permission bits.
            tar.extractall(dest, members=selected, filter="data")  # noqa: S202
        except tarfile.FilterError as exc:
            raise ExtractionError(f"archive member rejected: {exc}") from exc

    return {rel: dest / name for name, rel in rel_names.items()}
