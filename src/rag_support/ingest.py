"""Ingest pipeline: verified download, safe extraction, sanitize, chunk, screen.

Outputs in ``data/``:
- ``chunks.jsonl``: chunks cleared for indexing
- ``quarantine.jsonl``: chunks that matched an injection pattern, held for review
- ``ingest_report.json``: what was checked, what was found, and what was held back
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_support.chunking import Chunk, chunk_markdown
from rag_support.extract import ExtractLimits, safe_extract
from rag_support.includes import CODE_ROOT, expand_includes
from rag_support.sanitize import sanitize_markdown
from rag_support.screen import screen
from rag_support.sources import (
    IntegrityError,
    Source,
    download,
    load_lock,
    save_lock,
    sha256_file,
    verify,
)

# Sanitizer findings worth a per-file entry in the report.
NOTABLE_FINDINGS = frozenset({"invisible_chars", "dangerous_elements"})


def _archive_path(source: Source, data_dir: Path) -> Path:
    return data_dir / f"{source.name}-{source.commit[:12]}.tar.gz"


def pin(
    lock_path: Path, data_dir: Path, name: str, repo: str, ref: str, commit: str
) -> Source:
    """Download a commit's archive once and record its hash (trust on first use)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    unpinned = Source(name=name, repo=repo, ref=ref, commit=commit, archive_sha256="")
    digest = download(unpinned.archive_url, _archive_path(unpinned, data_dir))
    source = Source(name=name, repo=repo, ref=ref, commit=commit, archive_sha256=digest)
    save_lock(lock_path, source)
    return source


def fetch_verified(source: Source, data_dir: Path) -> Path:
    """Return the source archive, downloading it if needed, after a hash check."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = _archive_path(source, data_dir)
    if archive.exists():
        actual = sha256_file(archive)
    else:
        actual = download(source.archive_url, archive)
    try:
        verify(actual, source.archive_sha256)
    except IntegrityError:
        archive.unlink()
        raise
    return archive


def load_allowlist(path: Path) -> dict[str, str]:
    """Chunk IDs a person reviewed and cleared, mapped to the reason."""
    if not path.exists():
        return {}
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ingest_archive(
    archive: Path,
    source: Source,
    data_dir: Path,
    allowlist: dict[str, str],
    limits: ExtractLimits | None = None,
) -> dict[str, Any]:
    raw = data_dir / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    files = safe_extract(archive, raw, limits=limits)

    totals: Counter[str] = Counter()
    notable: dict[str, dict[str, int]] = {}
    rejected: list[dict[str, str]] = []
    indexed: list[Chunk] = []
    quarantined: list[dict[str, Any]] = []
    cleared_by_review: list[str] = []
    includes_resolved = 0
    includes_unresolved: list[dict[str, str]] = []

    texts: dict[str, str] = {}
    for rel, path in sorted(files.items()):
        try:
            texts[rel] = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            rejected.append({"source_path": rel, "reason": "invalid UTF-8"})
    code_files = {k: v for k, v in texts.items() if k.startswith(CODE_ROOT)}
    docs = {k: v for k, v in texts.items() if not k.startswith(CODE_ROOT)}

    for rel, raw_text in docs.items():
        expanded = expand_includes(raw_text, code_files)
        includes_resolved += expanded.resolved
        includes_unresolved += [
            {"source_path": rel, "target": t} for t in expanded.unresolved
        ]
        result = sanitize_markdown(expanded.text)
        totals.update(result.counts)
        flagged = {k: v for k, v in result.counts.items() if k in NOTABLE_FINDINGS}
        if flagged:
            notable[rel] = flagged
        for chunk in chunk_markdown(result.text, rel):
            matches = screen(chunk.text)
            if matches and chunk.chunk_id not in allowlist:
                quarantined.append({**asdict(chunk), "patterns": matches})
                continue
            if matches:
                cleared_by_review.append(chunk.chunk_id)
            indexed.append(chunk)

    _write_jsonl(data_dir / "chunks.jsonl", (asdict(c) for c in indexed))
    _write_jsonl(data_dir / "quarantine.jsonl", quarantined)
    report: dict[str, Any] = {
        "source": asdict(source),
        "archive_hash_verified": True,
        "files_extracted": len(files),
        "doc_files": len(docs),
        "code_files": len(code_files),
        "files_rejected": rejected,
        "includes_resolved": includes_resolved,
        "includes_unresolved": includes_unresolved,
        "sanitizer_totals": dict(sorted(totals.items())),
        "files_with_notable_findings": notable,
        "chunks_indexed": len(indexed),
        "chunks_quarantined": len(quarantined),
        "quarantined": [
            {
                "chunk_id": q["chunk_id"],
                "source_path": q["source_path"],
                "heading": q["heading"],
                "patterns": q["patterns"],
            }
            for q in quarantined
        ],
        "cleared_by_review": cleared_by_review,
    }
    (data_dir / "ingest_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_ingest(lock_path: Path, data_dir: Path, allowlist_path: Path) -> dict[str, Any]:
    source = load_lock(lock_path)
    archive = fetch_verified(source, data_dir)
    return ingest_archive(archive, source, data_dir, load_allowlist(allowlist_path))
