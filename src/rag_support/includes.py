"""Inline the code examples that FastAPI's docs pull in with include directives.

A directive looks like ``{* ../../docs_src/first_steps/tutorial001_py310.py hl[3] *}``.
Paths are relative to the docs project folder (``docs/en/``). A directive is
inlined only when its target resolves to a file inside ``docs_src/`` that was
extracted from the pinned archive; anything else is left as written.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

DIRECTIVE_RE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")
DOCS_PROJECT_DIR = "docs/en"
CODE_ROOT = "docs_src/"


@dataclass
class IncludeResult:
    text: str
    resolved: int = 0
    unresolved: list[str] = field(default_factory=list)


def resolve_target(target: str) -> str | None:
    """Return the repo-relative path if ``target`` stays inside ``docs_src/``."""
    if target.startswith("/") or "\\" in target:
        return None
    path = posixpath.normpath(posixpath.join(DOCS_PROJECT_DIR, target))
    return path if path.startswith(CODE_ROOT) else None


def _fenced(code: str) -> str | None:
    """Wrap code in a fence it can't close early, or None if neither fits."""
    for fence in ("```", "~~~"):
        if fence not in code:
            return f"{fence}python\n{code.rstrip()}\n{fence}"
    return None


def expand_includes(text: str, code_files: dict[str, str]) -> IncludeResult:
    result = IncludeResult(text="")

    def replace(match: re.Match[str]) -> str:
        path = resolve_target(match.group(1))
        code = code_files.get(path) if path else None
        block = _fenced(code) if code is not None else None
        if block is None:
            result.unresolved.append(match.group(1))
            return match.group(0)
        result.resolved += 1
        return block

    result.text = DIRECTIVE_RE.sub(replace, text)
    return result
