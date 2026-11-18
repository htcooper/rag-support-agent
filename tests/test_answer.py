from __future__ import annotations

import json
import re
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

from rag_support.answer import (
    REFUSAL_TEXT,
    STRICT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TRUNCATED_NOTE,
    answer_question,
    build_request,
    parse_response,
)
from rag_support.chunking import Chunk
from rag_support.retrieve import Hit
from tests.fakes_tracing import RecordingTracer


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


class FakeEmbedder:
    model = "text-embedding-3-small"

    def __init__(self) -> None:
        self.tokens_used = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.tokens_used += 7 * len(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeRetriever:
    def __init__(self) -> None:
        self.embedder = FakeEmbedder()

    def embed_query(self, query: str) -> np.ndarray:
        vector: np.ndarray = self.embedder.embed([query])[0]
        return vector

    def search(self, vector: np.ndarray, k: int = 5) -> list[Hit]:
        return HITS[:k]


class FakeClient:
    """Answers through client.beta.messages; rewrites through client.messages."""

    def __init__(self, res: NS, rewrite_text: str = "CORS CORSMiddleware") -> None:
        self.requests: list[dict[str, Any]] = []
        self.rewrite_requests: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._rewrite)
        self._res = res
        self._rewrite_text = rewrite_text

    def _create(self, **kwargs: Any) -> NS:
        self.requests.append(kwargs)
        return self._res

    def _rewrite(self, **kwargs: Any) -> NS:
        self.rewrite_requests.append(kwargs)
        return NS(
            model="claude-haiku-5-5",
            stop_reason="end_turn",
            content=[NS(type="text", text=self._rewrite_text)],
            usage=NS(input_tokens=100, output_tokens=10),
        )


class RecordingRetriever(FakeRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return super().embed_query(query)


def test_rewrite_changes_the_search_but_not_what_claude_is_asked() -> None:
    tracer = RecordingTracer()
    retriever = RecordingRetriever()
    client = FakeClient(response(text_block("A.", cite(0, "CORSMiddleware"))))
    answer = answer_question(
        "My React app gets blocked.",
        retriever,  # type: ignore[arg-type]
        client,
        model="claude-opus-5-5",
        tracer=tracer,
        rewrite=True,
    )
    assert retriever.queries == ["My React app gets blocked.\nCORS CORSMiddleware"]
    question_block = client.requests[0]["messages"][0]["content"][-1]
    assert question_block == {"type": "text", "text": "My React app gets blocked."}
    assert answer.rewrite is not None and answer.rewrite.text == "CORS CORSMiddleware"
    assert [(s.name, s.parent) for s in tracer.spans] == [
        ("answer_question", None),
        ("retrieve", "answer_question"),
        ("rewrite_query", "retrieve"),
        ("embed_query", "retrieve"),
        ("generate", "answer_question"),
    ]
    rewrite_span = tracer.by_name("rewrite_query")
    assert rewrite_span.updates["usage_details"] == {"input": 100, "output": 10}
    assert rewrite_span.updates["cost_details"]["total"] == pytest.approx(
        100 * 0.10 / 1e6 + 10 * 0.50 / 1e6
    )


def test_no_rewrite_by_default() -> None:
    retriever = RecordingRetriever()
    client = FakeClient(response(text_block("A.", cite(0, "CORSMiddleware"))))
    answer = answer_question(
        "cors?", retriever, client, model="claude-opus-5-5"  # type: ignore[arg-type]
    )
    assert retriever.queries == ["cors?"] and client.rewrite_requests == []
    assert answer.rewrite is None


def test_answer_question_uses_env_model_and_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    client = FakeClient(response(text_block("A.", cite(0, "CORSMiddleware"))))
    answer = answer_question(
        "cors?", FakeRetriever(), client, k=1  # type: ignore[arg-type]
    )
    assert client.requests[0]["model"] == "claude-sonnet-5-5"
    assert len(client.requests[0]["messages"][0]["content"]) == 2
    assert answer.grounded
    assert answer.trace_url is None  # tracing is off by default


def traced_answer(public: bool = False) -> tuple[RecordingTracer, Any]:
    tracer = RecordingTracer()
    client = FakeClient(response(text_block("A.", cite(0, "CORSMiddleware"))))
    answer = answer_question(
        "How do I enable CORS?",
        FakeRetriever(),  # type: ignore[arg-type]
        client,
        model="claude-opus-5-5",
        tracer=tracer,
        public=public,
    )
    return tracer, answer


def test_trace_is_retrieve_then_generate_with_embedding_inside_retrieve() -> None:
    tracer, answer = traced_answer()
    assert [(s.name, s.as_type, s.parent) for s in tracer.spans] == [
        ("answer_question", "chain", None),
        ("retrieve", "retriever", "answer_question"),
        ("embed_query", "embedding", "retrieve"),
        ("generate", "generation", "answer_question"),
    ]
    assert tracer.by_name("answer_question").updates["output"]["grounded"] is True
    assert answer.trace_url == "https://langfuse.example/trace/1"
    assert not tracer.public


def test_trace_records_retrieval_and_generation_details() -> None:
    tracer, _ = traced_answer()
    hits = tracer.by_name("retrieve").updates["output"]
    assert [h["chunk_id"] for h in hits] == ["cors", "files"]
    assert hits[0]["url"] == "https://fastapi.tiangolo.com/cors/"

    embed = tracer.by_name("embed_query").updates
    assert embed["usage_details"] == {"input": 7}

    gen = tracer.by_name("generate")
    assert gen.fields["model"] == "claude-opus-5-5"
    assert gen.updates["usage_details"] == {"input": 1200, "output": 150}
    assert gen.updates["cost_details"]["total"] == pytest.approx(
        1200 * 4 / 1e6 + 150 * 20 / 1e6
    )
    assert gen.updates["metadata"]["cited_ranks"] == [1]


def test_public_flag_marks_the_trace_public() -> None:
    tracer, _ = traced_answer(public=True)
    assert tracer.public


def test_trace_contains_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "sk-test-sentinel-0000000000000000"
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
    ):
        monkeypatch.setenv(var, sentinel)
    tracer, _ = traced_answer()
    payload = json.dumps(
        [{"fields": s.fields, "updates": s.updates} for s in tracer.spans], default=str
    )
    assert sentinel not in payload
    assert not re.search(r"\b(sk|pk)-[A-Za-z0-9-]{8,}", payload)


def test_strict_changes_only_the_system_prompt() -> None:
    plain = FakeClient(response(text_block("A.", cite(0, "CORSMiddleware"))))
    strict = FakeClient(response(text_block("A.", cite(0, "CORSMiddleware"))))
    a = answer_question(
        "cors?", FakeRetriever(), plain, model="m"  # type: ignore[arg-type]
    )
    b = answer_question(
        "cors?", FakeRetriever(), strict, model="m", strict=True  # type: ignore[arg-type]
    )
    req_a, req_b = plain.requests[0], strict.requests[0]
    assert req_a["system"] == SYSTEM_PROMPT and a.system_prompt == SYSTEM_PROMPT
    assert req_b["system"] == STRICT_SYSTEM_PROMPT
    assert b.system_prompt == STRICT_SYSTEM_PROMPT
    assert {k: v for k, v in req_a.items() if k != "system"} == {
        k: v for k, v in req_b.items() if k != "system"
    }


def test_strict_prompt_extends_the_original_unchanged() -> None:
    assert STRICT_SYSTEM_PROMPT.startswith(SYSTEM_PROMPT)
    assert '"Source:"' in STRICT_SYSTEM_PROMPT
