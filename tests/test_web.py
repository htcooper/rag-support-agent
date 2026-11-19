from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

from rag_support.answer import STRICT_SYSTEM_PROMPT, SYSTEM_PROMPT, Answer
from rag_support.rewrite import Rewrite
from rag_support.web import MAX_QUESTION_CHARS, answer_cost, create_app
from tests.fakes_tracing import RecordingTracer
from tests.test_answer import FakeClient, FakeRetriever, cite, response, text_block

REQUEST = httpx2.Request("POST", "https://api.example/v1")
QUESTION = "How do I enable CORS?"


def cited_answer() -> NS:
    return response(text_block("Add the middleware.", cite(0, "Use CORSMiddleware")))


class Harness:
    def __init__(self, client: Any = None, retriever: Any = None) -> None:
        self.tracers: list[RecordingTracer] = []
        self.client = client or FakeClient(cited_answer())
        fake_retriever: Any = retriever or FakeRetriever()
        app = create_app(fake_retriever, self.client, self.new_tracer)
        # Errors without a message should become a 500, not fail the test run.
        self.http = TestClient(app, raise_server_exceptions=False)

    def new_tracer(self) -> RecordingTracer:
        tracer = RecordingTracer()
        self.tracers.append(tracer)
        return tracer

    def ask(self, **body: Any) -> Any:
        return self.http.post("/api/ask", json={"question": QUESTION, **body})


def test_page_is_served() -> None:
    page = Harness().http.get("/")
    assert page.status_code == 200
    assert "<title>Support Agent</title>" in page.text


def test_api_docs_are_off() -> None:
    http = Harness().http
    assert http.get("/docs").status_code == 404
    assert http.get("/openapi.json").status_code == 404


def test_answer_payload_has_what_both_views_show() -> None:
    h = Harness()
    res = h.ask()
    assert res.status_code == 200
    data = res.json()
    assert data["question"] == QUESTION
    assert data["text"] == "Add the middleware.[1]"
    assert data["grounded"] is True and data["refused"] is False
    assert data["citations"] == [
        {
            "rank": 1,
            "heading": "CORS",
            "url": "https://fastapi.tiangolo.com/cors/",
            "cited_text": "Use CORSMiddleware",
        }
    ]
    assert [(h["rank"], h["cited"]) for h in data["hits"]] == [(1, True), (2, False)]
    assert data["hits"][0]["preview"] == "Use CORSMiddleware to allow other origins."
    assert data["rewrite"] == "CORS CORSMiddleware"
    assert data["model"] == "claude-opus-5-5"
    assert data["usage"] == {"input_tokens": 1200, "output_tokens": 150}
    # Opus 1200 in / 150 out + Haiku 100 in / 10 out, at the prices in tracing.py.
    assert data["cost_usd"] == pytest.approx(0.0078, abs=1e-4)
    assert isinstance(data["latency_s"], float)
    assert data["notes"] == []
    assert data["trace_url"] == "https://langfuse.example/trace/1"
    assert h.tracers[0].flushed


def test_defaults_rewrite_and_use_the_strict_prompt() -> None:
    h = Harness()
    h.ask()
    assert len(h.client.rewrite_requests) == 1
    assert h.client.requests[0]["system"] == STRICT_SYSTEM_PROMPT


def test_rewrite_and_strict_can_be_turned_off() -> None:
    h = Harness()
    data = h.ask(rewrite=False, strict=False).json()
    assert h.client.rewrite_requests == []
    assert h.client.requests[0]["system"] == SYSTEM_PROMPT
    assert data["rewrite"] is None


def test_question_is_stripped_before_answering() -> None:
    h = Harness()
    h.http.post("/api/ask", json={"question": f"  {QUESTION}\n"})
    assert h.client.requests[0]["messages"][0]["content"][-1]["text"] == QUESTION


@pytest.mark.parametrize("question", ["", "   \n ", "x" * (MAX_QUESTION_CHARS + 1)])
def test_empty_or_too_long_questions_are_rejected(question: str) -> None:
    h = Harness()
    res = h.http.post("/api/ask", json={"question": question})
    assert res.status_code == 422
    assert h.client.requests == []  # nothing was sent to Claude


def test_question_at_the_length_limit_is_accepted() -> None:
    res = Harness().http.post("/api/ask", json={"question": "x" * MAX_QUESTION_CHARS})
    assert res.status_code == 200


class FailingClient(FakeClient):
    def __init__(self, exc: Exception) -> None:
        super().__init__(cited_answer())
        self._exc = exc

    def _create(self, **kwargs: Any) -> NS:
        raise self._exc


class FailingRetriever(FakeRetriever):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def embed_query(self, query: str) -> Any:
        raise self._exc


def status_error(cls: Any, code: int) -> Exception:
    exc: Exception = cls(
        "server said no", response=httpx2.Response(code, request=REQUEST), body=None
    )
    return exc


@pytest.mark.parametrize(
    ("harness", "status", "detail"),
    [
        (
            lambda: Harness(
                client=FailingClient(status_error(anthropic.RateLimitError, 429))
            ),
            429,
            "Rate limited or spend cap reached. Try again later.",
        ),
        (
            lambda: Harness(
                client=FailingClient(anthropic.APIConnectionError(request=REQUEST))
            ),
            503,
            "Couldn't reach the Anthropic API. Check the network connection.",
        ),
        (
            # The question is embedded with OpenAI before Claude is called.
            lambda: Harness(
                retriever=FailingRetriever(
                    status_error(openai.AuthenticationError, 401)
                )
            ),
            502,
            "OpenAI rejected the API key. Check OPENAI_API_KEY in .env.",
        ),
    ],
)
def test_api_errors_return_their_message(
    harness: Any, status: int, detail: str
) -> None:
    h = harness()
    res = h.ask()
    assert res.status_code == status
    assert res.json() == {"detail": detail}
    assert h.tracers[0].flushed  # the trace is still sent


def test_unexpected_errors_are_a_500_and_still_flush_the_trace() -> None:
    malformed = anthropic.APIResponseValidationError(
        response=httpx2.Response(200, request=REQUEST), body=None
    )
    h = Harness(client=FailingClient(malformed))
    assert h.ask().status_code == 500
    assert h.tracers[0].flushed


def test_cost_is_unknown_for_a_model_without_a_price() -> None:
    answer = Answer(question="q", text="", hits=[], model="some-new-model")
    answer.usage = {"input_tokens": 10, "output_tokens": 10}
    assert answer_cost(answer) is None


def test_cost_adds_the_rewrite_call() -> None:
    answer = Answer(question="q", text="", hits=[], model="claude-opus-5-5")
    answer.usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert answer_cost(answer) == 4.0
    answer.rewrite = Rewrite(
        text="x",
        model="claude-haiku-5-5",
        usage={"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert answer_cost(answer) == 4.1
