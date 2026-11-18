"""Screen chunks for text that looks like instructions aimed at a model.

Matching runs on an NFKC-normalized, case-folded copy, so fullwidth or other
compatibility lookalike characters can't slip a phrase past the patterns. The
stored text is not changed. Matches are quarantined for human review, not
silently dropped.
"""

from __future__ import annotations

import re
import unicodedata

PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}"
        r"\b(previous|prior|above|earlier|all|any|your)\b.{0,40}"
        r"\b(instructions?|prompts?|rules?|directions?|guidelines?)\b"
    ),
    "role_marker": re.compile(
        r"(?m)^\s*(system|assistant|human)\s*:|</?system>|<\|im_(start|end)\|>|\[/?inst\]"
    ),
    "persona_switch": re.compile(
        r"\byou are now (an?|the|my) .{0,20}\b(ai|assistant|model|bot|agent|chatbot)\b"
    ),
    "new_instructions": re.compile(r"\b(new|updated|real) (system )?instructions\s*:"),
    "reveal_prompt": re.compile(
        r"\b(reveal|print|show|output|repeat|leak)\b.{0,30}"
        r"\b(system prompt|your (instructions|prompt))\b"
    ),
    "base64_blob": re.compile(r"[a-z0-9+/]{120,}={0,2}"),
}


def normalize_for_screening(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def screen(text: str) -> list[str]:
    """Return the names of the patterns that match, in a stable order."""
    folded = normalize_for_screening(text)
    return [name for name, pattern in PATTERNS.items() if pattern.search(folded)]
