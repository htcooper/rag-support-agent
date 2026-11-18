"""Sanitize untrusted Markdown before it is chunked, embedded, or shown to a model.

Prose is cleaned of HTML comments, dangerous HTML elements, other tags, images,
and links to domains outside the allowlist. Fenced code blocks and inline code
spans are kept as inert text. Invisible characters are removed everywhere,
including inside code.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

ALLOWED_LINK_DOMAINS = frozenset(
    {
        "fastapi.tiangolo.com",
        "github.com",
        "python.org",
        "docs.pydantic.dev",
        "starlette.dev",
        "www.starlette.io",
    }
)

# Zero-width, bidi-control, and other invisible characters, plus the Unicode
# "tag" block (U+E0000-U+E007F), which can carry hidden ASCII text.
INVISIBLE_RE = re.compile(
    "[\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff"
    "\U000e0000-\U000e007f]"
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
DANGEROUS_ELEMENT_RE = re.compile(
    r"<(script|style|iframe|object|embed|template|noscript)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
# Link targets may contain one level of balanced parentheses, e.g. "x()".
_TARGET = r"(?:[^()\s]|\([^()\s]*\))+"
IMAGE_RE = re.compile(rf"!\[([^\]]*)\]\({_TARGET}(?:\s+\"[^\"]*\")?\)")
LINK_RE = re.compile(rf"\[([^\]]+)\]\(({_TARGET})(?:\s+\"[^\"]*\")?\)")
INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class SanitizeResult:
    text: str
    counts: Counter[str] = field(default_factory=Counter)


def is_allowed_url(url: str) -> bool:
    """Relative links stay inside the docs; absolute ones must be allowlisted."""
    parsed = urlparse(url)
    if not parsed.scheme and not parsed.netloc:
        return not url.lower().startswith(("javascript:", "data:", "vbscript:"))
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_LINK_DOMAINS)


def split_fences(text: str) -> list[tuple[bool, str]]:
    """Split into (is_code, segment) pieces on fenced code blocks."""
    segments: list[tuple[bool, str]] = []
    buf: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if fence is None and match:
            if buf:
                segments.append((False, "".join(buf)))
            buf, fence = [line], match.group(1)
        elif fence is not None and match and match.group(1) == fence:
            buf.append(line)
            segments.append((True, "".join(buf)))
            buf, fence = [], None
        else:
            buf.append(line)
    if buf:
        # An unclosed fence is treated as code, matching how Markdown renders it.
        segments.append((fence is not None, "".join(buf)))
    return segments


def _clean_prose(prose: str, counts: Counter[str]) -> str:
    parts = INLINE_CODE_RE.split(prose)
    for i in range(0, len(parts), 2):  # even indexes are prose, odd are code spans
        part = parts[i]
        part, n = HTML_COMMENT_RE.subn("", part)
        counts["html_comments"] += n
        part, n = DANGEROUS_ELEMENT_RE.subn("", part)
        counts["dangerous_elements"] += n

        def autolink(m: re.Match[str]) -> str:
            if is_allowed_url(m.group(1)):
                return m.group(1)
            counts["links_removed"] += 1
            return ""

        part = AUTOLINK_RE.sub(autolink, part)
        part, n = HTML_TAG_RE.subn("", part)
        counts["html_tags"] += n
        part, n = IMAGE_RE.subn(r"\1", part)
        counts["images"] += n

        def link(m: re.Match[str]) -> str:
            if is_allowed_url(m.group(2)):
                return m.group(0)
            counts["links_removed"] += 1
            return m.group(1)

        parts[i] = LINK_RE.sub(link, part)
    return "".join(parts)


def sanitize_markdown(text: str) -> SanitizeResult:
    counts: Counter[str] = Counter()
    text = unicodedata.normalize("NFC", text)
    text, n = INVISIBLE_RE.subn("", text)
    counts["invisible_chars"] += n
    out = [
        segment if is_code else _clean_prose(segment, counts)
        for is_code, segment in split_fences(text)
    ]
    return SanitizeResult(text="".join(out), counts=+counts)
