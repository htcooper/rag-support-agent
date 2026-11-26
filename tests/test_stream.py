from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace as NS
from typing import Any

import anthropic

from rag_support.stream import FilteredText, StreamingClient
from tests.test_answer import FakeClient, cite, response, text_block
from tests.test_web import QUESTION, Harness, cited_answer, status_error

EVIL_TEXT = (
    "See [the guide](https://evil.example/steal) or https://evil.example/x for "
    "more, but https://github.com.evil.example/y is not GitHub. The docs are at "
    "https://fastapi.tiangolo.com/tutorial/cors/ and [GitHub](https://github.com/x).\n"
    "\n"
    "```python\n"
    'app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"])\n'
    "```\n"
    "Done."
)


def snapshots(deltas: list[str]) -> list[str]:
    text = FilteredText()
    return [s for d in deltas if (s := text.add(d)) is not None]


def every_split(text: str) -> Iterator[list[str]]:
    """The text as two deltas at every position, then one character at a time."""
    for i in range(1, len(text)):
        yield [text[:i], text[i:]]
    yield list(text)


def test_no_snapshot_ever_shows_a_link_outside_the_allowlist() -> None:
    for deltas in every_split(EVIL_TEXT):
        for snapshot in snapshots(deltas):
            assert "evil" not in snapshot, deltas
            # Never a prefix of the disguised URL that would pass as GitHub.
            assert "https://github.com." not in snapshot, deltas


def test_allowed_links_and_code_urls_come_through() -> None:
    final = snapshots(list(EVIL_TEXT))[-1]
    assert "https://fastapi.tiangolo.com/tutorial/cors/" in final
    assert "[GitHub](https://github.com/x)" in final
    assert '"http://localhost:3000"' in final  # code is left as written
    assert "[link removed]" in final


def test_the_unfinished_last_word_is_held_back() -> None:
    text = FilteredText()
    assert text.add("Use https://fastapi.tia") == "Use "
    assert text.add("ngolo") is None
    assert text.add(".com/cors/ now") == "Use https://fastapi.tiangolo.com/cors/ "
    assert text.raw == "Use https://fastapi.tiangolo.com/cors/ now"


def test_a_link_split_across_lines_is_filtered_once_it_closes() -> None:
    shown = snapshots(["[see\n", "this](https://evil.example/a) ok "])
    assert shown == ["[see\n", "see\nthis ok "]


class StreamingFakeClient(FakeClient):
    """A FakeClient that also streams the answer as the given deltas."""

    def __init__(self, res: NS, deltas: list[str]) -> None:
        super().__init__(res)
        self.stream_requests: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(create=self._create, stream=self._stream))
        self._deltas = deltas

    @contextmanager
    def _stream(self, **kwargs: Any) -> Iterator[NS]:
        self.stream_requests.append(kwargs)
        yield NS(
            text_stream=iter(self._deltas),
            get_final_message=lambda: self._res,
        )


def test_streaming_client_sends_the_same_request_and_returns_the_final_message() -> (
    None
):
    res = cited_answer()
    fake = StreamingFakeClient(res, ["Add ", "the ", "middleware."])
    seen: list[str] = []
    started: list[bool] = []
    client = StreamingClient(fake, seen.append, lambda: started.append(True))
    request = {"model": "m", "max_tokens": 10, "fallbacks": "default"}
    assert client.beta.messages.create(**request) is res
    assert fake.stream_requests == [request]
    assert fake.requests == []  # nothing went through the non-streaming call
    assert seen == ["Add ", "the ", "middleware."]
    assert started == [True]
    assert client.messages is fake.messages  # the rewrite is passed through


def events(res: Any) -> list[dict[str, Any]]:
    frames = [f for f in res.text.split("\n\n") if f.startswith("data: ")]
    return [json.loads(f.removeprefix("data: ")) for f in frames]


def ask_stream(h: Harness, **body: Any) -> Any:
    return h.http.post("/api/ask/stream", json={"question": QUESTION, **body})


def test_stream_sends_status_text_then_the_same_answer_as_api_ask() -> None:
    res = response(text_block("Add the middleware.", cite(0, "Use CORSMiddleware")))
    fake = StreamingFakeClient(res, ["Add ", "the ", "middleware."])
    h = Harness(client=fake)
    streamed = ask_stream(h)
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    got = events(streamed)
    assert [e["type"] for e in got] == [
        "status",
        "status",
        "text",
        "text",
        "answer",
    ]
    assert got[0]["stage"] == "searching" and got[1]["stage"] == "thinking"
    assert [e["text"] for e in got[2:4]] == ["Add ", "Add the "]

    plain = h.ask().json()
    final = got[-1]["answer"]
    final.pop("latency_s")
    plain.pop("latency_s")
    assert final == plain
    assert fake.stream_requests == fake.requests  # same request either way
    assert all(t.flushed for t in h.tracers)


def test_streamed_text_is_link_filtered() -> None:
    res = response(text_block(EVIL_TEXT))
    h = Harness(client=StreamingFakeClient(res, list(EVIL_TEXT)))
    got = events(ask_stream(h))
    texts = [e["text"] for e in got if e["type"] == "text"]
    assert texts and all("evil" not in t for t in texts)
    assert "evil" not in got[-1]["answer"]["text"]


def test_stream_reports_api_errors_as_an_event() -> None:
    class Failing(StreamingFakeClient):
        @contextmanager
        def _stream(self, **kwargs: Any) -> Iterator[NS]:
            raise status_error(anthropic.RateLimitError, 429)
            yield NS()  # pragma: no cover

    h = Harness(client=Failing(cited_answer(), []))
    got = events(ask_stream(h))
    assert got[-1] == {
        "type": "error",
        "status": 429,
        "detail": "Rate limited or spend cap reached. Try again later.",
    }
    assert h.tracers[0].flushed


def test_stream_reports_unexpected_errors_without_details() -> None:
    class Broken(StreamingFakeClient):
        @contextmanager
        def _stream(self, **kwargs: Any) -> Iterator[NS]:
            raise RuntimeError("internal detail")
            yield NS()  # pragma: no cover

    h = Harness(client=Broken(cited_answer(), []))
    got = events(ask_stream(h))
    assert got[-1] == {"type": "error", "status": 500, "detail": None}
    assert "internal detail" not in json.dumps(got)


def test_stream_rejects_invalid_questions_before_streaming() -> None:
    fake = StreamingFakeClient(cited_answer(), [])
    res = Harness(client=fake).http.post("/api/ask/stream", json={"question": "  "})
    assert res.status_code == 422
    assert fake.stream_requests == []
