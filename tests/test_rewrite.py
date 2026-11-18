from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from rag_support.rewrite import (
    MAX_REWRITE_CHARS,
    REWRITE_MODEL,
    REWRITE_SYSTEM,
    Rewrite,
    rewrite_query,
    search_text,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeMessages:
    def __init__(self, text: str, stop: str = "end_turn") -> None:
        self.text, self.stop = text, stop
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> NS:
        self.requests.append(kwargs)
        return NS(
            model=REWRITE_MODEL,
            stop_reason=self.stop,
            content=[NS(type="thinking", thinking=""), NS(type="text", text=self.text)],
            usage=NS(input_tokens=120, output_tokens=18),
        )


def test_rewrite_returns_one_cleaned_line() -> None:
    messages = FakeMessages("  CORS   CORSMiddleware\nallow_origins  ")
    rewrite = rewrite_query("React app blocked", NS(messages=messages))
    assert rewrite.text == "CORS CORSMiddleware allow_origins"
    assert rewrite.usage == {"input_tokens": 120, "output_tokens": 18}
    request = messages.requests[0]
    assert request["model"] == REWRITE_MODEL
    assert request["system"] == REWRITE_SYSTEM
    assert request["messages"] == [{"role": "user", "content": "React app blocked"}]
    assert request["output_config"] == {"effort": "low"}


def test_rewrite_is_capped() -> None:
    rewrite = rewrite_query("q", NS(messages=FakeMessages("word " * 200)))
    assert len(rewrite.text) <= MAX_REWRITE_CHARS


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_failed_rewrite_falls_back_to_the_question(stop: str) -> None:
    rewrite = rewrite_query("How do I X?", NS(messages=FakeMessages("partial", stop)))
    assert rewrite.text == "" and rewrite.stop_reason == stop
    assert search_text("How do I X?", rewrite) == "How do I X?"


def test_search_keeps_the_users_words_and_adds_the_rewrite() -> None:
    assert search_text("q", Rewrite(text="CORSMiddleware")) == "q\nCORSMiddleware"


def test_rewriter_instructions_name_no_features_from_the_eval() -> None:
    """Naming features in the prompt would feed the rewriter the eval's answers."""
    terms = {"CORS", "BackgroundTasks", "response_model", "UploadFile", "root_path"}
    terms |= {"StreamingResponse", "middleware", "yield", "Depends", "lifespan"}
    for line in (ROOT / "evals" / "cases.jsonl").read_text("utf-8").splitlines():
        terms.add(json.loads(line)["id"])
    found = sorted(t for t in terms if t.lower() in REWRITE_SYSTEM.lower())
    assert found == []
