"""The project's own source must not contain the hidden characters it screens for."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_are_ascii() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{n}"
        for folder in ("src", "tests")
        for path in sorted((ROOT / folder).rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if not line.isascii()
    ]
    assert offenders == [], "use \\u escapes instead of literal non-ASCII characters"
