"""Split sanitized Markdown into heading-based chunks.

Sections are split on headings outside code fences. Large sections are split on
paragraph boundaries without breaking a code block; small sections are merged
forward within the same file.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from rag_support.sanitize import split_fences

MIN_TOKENS = 300
MAX_TOKENS = 800
DOCS_ROOT = "docs/en/docs/"
DOCS_BASE_URL = "https://fastapi.tiangolo.com/"
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
# mkdocs heading attributes, e.g. "## Install { #install }"
HEADING_ATTR_RE = re.compile(r"\s*\{[^}]*\}\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str
    heading: str
    url: str
    text: str
    sha256: str


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def doc_url(source_path: str) -> str:
    """Map ``docs/en/docs/tutorial/first-steps.md`` to its docs-site URL."""
    rel = PurePosixPath(source_path.removeprefix(DOCS_ROOT))
    if rel.name == "index.md":
        page = rel.parent.as_posix()
        return DOCS_BASE_URL if page == "." else f"{DOCS_BASE_URL}{page}/"
    return f"{DOCS_BASE_URL}{rel.with_suffix('').as_posix()}/"


def _sections(text: str) -> list[tuple[str, str]]:
    """Return (heading path, body) pairs; headings inside code are ignored."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    heading = ""
    buf: list[str] = []
    for is_code, segment in split_fences(text):
        if is_code:
            buf.append(segment)
            continue
        for line in segment.splitlines(keepends=True):
            match = HEADING_RE.match(line)
            if not match:
                buf.append(line)
                continue
            sections.append((heading, "".join(buf)))
            level = len(match.group(1))
            path = path[: level - 1] + [HEADING_ATTR_RE.sub("", match.group(2))]
            heading = " > ".join(path)
            buf = [line]
    sections.append((heading, "".join(buf)))
    return [(h, body) for h, body in sections if body.strip()]


def _blocks(body: str) -> list[str]:
    """Split into paragraphs, keeping each fenced code block whole."""
    blocks: list[str] = []
    for is_code, segment in split_fences(body):
        if is_code:
            blocks.append(segment)
            continue
        for paragraph in re.split(r"(?<=\n\n)", segment):
            if estimate_tokens(paragraph) > MAX_TOKENS:
                # e.g. a long bullet list with no blank lines: split on lines
                blocks.extend(paragraph.splitlines(keepends=True))
            elif paragraph:
                blocks.append(paragraph)
    return blocks


def _split_large(heading: str, body: str) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    current = ""
    for block in _blocks(body):
        if current and estimate_tokens(current + block) > MAX_TOKENS:
            pieces.append((heading, current))
            current = ""
        current += block
    if current.strip():
        pieces.append((heading, current))
    return pieces


def _merge_small(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for heading, body in sections:
        if merged:
            prev_heading, prev_body = merged[-1]
            if (
                estimate_tokens(prev_body) < MIN_TOKENS
                and estimate_tokens(prev_body + body) <= MAX_TOKENS
            ):
                merged[-1] = (prev_heading, prev_body.rstrip() + "\n\n" + body)
                continue
        merged.append((heading, body))
    return merged


def chunk_markdown(text: str, source_path: str) -> list[Chunk]:
    pieces = [p for h, body in _sections(text) for p in _split_large(h, body)]
    url = doc_url(source_path)
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for heading, body in _merge_small(pieces):
        body = body.strip()
        # A page can include the same code example in several sections; index it
        # once so IDs stay unique and duplicates don't crowd out search results.
        if body in seen:
            continue
        seen.add(body)
        chunk_id = hashlib.sha256(f"{source_path}\x00{body}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_path=source_path,
                heading=heading,
                url=url,
                text=body,
                sha256=hashlib.sha256(body.encode()).hexdigest(),
            )
        )
    return chunks
