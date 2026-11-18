from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import pytest

from rag_support.answer import (
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    TRUNCATED_NOTE,
    answer_question,
    build_request,
    parse_response,
)
from rag_support.chunking import Chunk
from rag_support.retrieve import Hit


def chunk(key: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=key,
        source_path=f"docs/en/docs/{key}.md",
        heading=key.upper(),
        url=f"https://fastapi.tiangolo.com/{key}/",
        text=text,
        sha256="0" * 64,
    )


HITS = [
    Hit(chunk("cors", "Use CORSMiddleware to allow other origins."), 0.5),
    Hit(chunk("files", "Use UploadFile to receive files."), 0.4),
]


def cite(index: int, text: str) -> NS:
    return NS(type="char_location", document_index=index, cited_text=text)


def text_block(text: str, *citations: NS) -> NS:
    return NS(type="text", text=text, citations=list(citations) or None)


def response(*content: NS, stop: str = "end_turn", fallback: bool = False) -> NS:
    iterations = [NS(type="fallback_message")] if fallback else None
    return NS(
        model="claude-opus-5-5",
        stop_reason=stop,
        content=list(content),
        usage=NS(input_tokens=1200, output_tokens=150, iterations=iterations),
    )


def test_request_keeps_documents_out_of_the_system_prompt() -> None:
    req = build_request("How do I enable CORS?", HITS, "claude-opus-5-5", "medium")
    assert req["system"] == SYSTEM_PROMPT
    assert "CORSMiddleware" not in req["system"]
    content = req["messages"][0]["content"]
    assert [c["type"] for c in content] == ["document", "document", "text"]
    assert content[0]["source"]["data"] == HITS[0].chunk.text
    assert content[0]["citations"] == {"enabled": True}
    assert content[0]["context"] == "Source: https://fastapi.tiangolo.com/cors/"
    assert content[-1]["text"] == "How do I enable CORS?"
    assert req["fallbacks"] == "default"
    assert req["output_config"] == {"effort": "medium"}
    assert "tools" not in req


def test_citations_map_to_chunks_and_add_markers() -> None:
    res = response(
        text_block("Add the middleware.", cite(0, "Use CORSMiddleware")),
        text_block(" Thinking blocks are skipped."),
    )
    answer = parse_response("q", HITS, res)
    assert answer.text == "Add the middleware.[1] Thinking blocks are skipped."
    assert answer.cited_ranks == [1]
    assert answer.citations[0].chunk.chunk_id == "cors"
    assert answer.grounded


def test_citation_to_unknown_document_is_dropped() -> None:
    answer = parse_response("q", HITS, response(text_block("x", cite(7, "Use"))))
    assert answer.citations == [] and answer.invalid_citations == 1
    assert not answer.grounded


def test_citation_quoting_text_not_in_the_chunk_is_dropped() -> None:
    res = response(text_block("x", cite(1, "text that was never in the docs")))
    answer = parse_response("q", HITS, res)
    assert answer.citations == [] and answer.invalid_citations == 1


def test_answer_without_citations_is_not_grounded() -> None:
    res = response(text_block("I couldn't find that in the FastAPI docs."))
    answer = parse_response("q", HITS, res)
    assert not answer.grounded and answer.invalid_citations == 0


def test_refusal_returns_fixed_text() -> None:
    res = response(text_block("partial"), stop="refusal")
    answer = parse_response("q", HITS, res)
    assert answer.refused and not answer.grounded
    assert answer.text == REFUSAL_TEXT


def test_truncated_answer_is_marked() -> None:
    res = response(text_block("Long answer"), stop="max_tokens")
    assert parse_response("q", HITS, res).text.endswith(TRUNCATED_NOTE)


def test_non_text_blocks_are_ignored_and_fallback_is_reported() -> None:
    res = response(
        NS(type="thinking", thinking=""),
        NS(type="fallback"),
        text_block("Answer.", cite(1, "UploadFile")),
        fallback=True,
    )
    answer = parse_response("q", HITS, res)
    assert answer.text == "Answer.[2]" and answer.fallback_used


def test_links_outside_the_allowlist_are_removed() -> None:
    res = response(
        text_block(
            "See [the docs](https://fastapi.tiangolo.com/tutorial/cors/) or "
            "[this](https://evil.example/x) and https://evil.example/y.",
            cite(0, "CORSMiddleware"),
        )
    )
    answer = parse_response("q", HITS, res)
    assert "fastapi.tiangolo.com/tutorial/cors/" in answer.text
    assert "evil.example" not in answer.text
    assert answer.links_removed == 2


class FakeRetriever:
    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        return HITS[:k]


class FakeClient:
    def __init__(self, res: NS) -> None:
        self.requests: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(create=self._create))
        self._res = res

    def _create(self, **kwargs: Any) -> NS:
        self.requests.append(kwargs)
        return self._res


def test_answer_question_uses_env_model_and_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    client = FakeClient(response(text_block("A.", cite(0, "CORSMiddleware"))))
    answer = answer_question(
        "cors?", FakeRetriever(), client, k=1  # type: ignore[arg-type]
    )
    assert client.requests[0]["model"] == "claude-sonnet-5-5"
    assert len(client.requests[0]["messages"][0]["content"]) == 2
    assert answer.grounded
