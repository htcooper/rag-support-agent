"""Query rewriting: restate a question in the docs' vocabulary before searching.

Users describe symptoms ("my React app gets blocked"); the docs name features
("CORSMiddleware"). A small, cheap model bridges the two. The rewrite is used
only to search: Claude never sees it as a source, so answers still come only
from retrieved documents.

The instructions deliberately name no example features, so the rewriter isn't
fed the answers to the eval's questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REWRITE_MODEL = "claude-haiku-5-5"
REWRITE_EFFORT = "low"
REWRITE_MAX_TOKENS = 1024
MAX_REWRITE_CHARS = 300

REWRITE_SYSTEM = """\
You turn a user's question about FastAPI, the Python web framework, into a \
search query for the FastAPI documentation.

Write one line that names the FastAPI features, classes, functions, parameters, \
and concepts whose documentation would answer the question, using the \
documentation's own terms. Don't answer the question. Output only the query."""


@dataclass
class Rewrite:
    text: str  # empty when rewriting failed; search then uses the question alone
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None


def search_text(question: str, rewrite: Rewrite) -> str:
    """Search with the question plus its rewrite, keeping the user's own words."""
    return f"{question}\n{rewrite.text}" if rewrite.text else question


def rewrite_query(question: str, client: Any, model: str = REWRITE_MODEL) -> Rewrite:
    response = client.messages.create(
        model=model,
        max_tokens=REWRITE_MAX_TOKENS,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": question}],
        output_config={"effort": REWRITE_EFFORT},
    )
    rewrite = Rewrite(
        text="",
        model=response.model,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        stop_reason=response.stop_reason,
    )
    if response.stop_reason == "end_turn":
        text = next((b.text for b in response.content if b.type == "text"), "")
        rewrite.text = " ".join(text.split())[:MAX_REWRITE_CHARS]
    return rewrite
