from __future__ import annotations

import argparse
from typing import Any

import anthropic
import httpx2
import pytest

from rag_support import __main__ as cli
from rag_support.errors import describe_api_error
from rag_support.retrieve import Index

# The Anthropic SDK builds its errors from httpx2 requests and responses.
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type[anthropic.APIStatusError], code: int) -> Exception:
    response = httpx2.Response(code, request=REQUEST)
    return cls("server said no", response=response, body=None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            status_error(anthropic.AuthenticationError, 401),
            "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.",
        ),
        (
            status_error(anthropic.PermissionDeniedError, 403),
            "The API key can't use this model or feature: server said no",
        ),
        (
            status_error(anthropic.RateLimitError, 429),
            "Rate limited or spend cap reached. Try again later.",
        ),
        (
            status_error(anthropic.InternalServerError, 500),
            "Anthropic API error 500: server said no",
        ),
        (
            anthropic.APIConnectionError(request=REQUEST),
            "Couldn't reach the Anthropic API. Check the network connection.",
        ),
        (
            # A timeout is a kind of connection error.
            anthropic.APITimeoutError(request=REQUEST),
            "Couldn't reach the Anthropic API. Check the network connection.",
        ),
    ],
)
def test_each_api_error_gets_its_message(exc: Exception, expected: str) -> None:
    assert describe_api_error(exc) == expected


def test_other_errors_get_no_message() -> None:
    assert describe_api_error(ValueError("not an API error")) is None
    response = httpx2.Response(200, request=REQUEST)
    malformed = anthropic.APIResponseValidationError(response=response, body=None)
    assert describe_api_error(malformed) is None


class FlushTracer:
    flushed = False

    def flush(self) -> None:
        self.flushed = True


def ask_args() -> argparse.Namespace:
    return argparse.Namespace(
        question="How do I enable CORS?",
        effort="medium",
        public=False,
        rewrite=True,
        strict=True,
    )


def patch_pipeline(
    monkeypatch: pytest.MonkeyPatch, tracer: FlushTracer, exc: Exception
) -> None:
    class StubIndex:
        info = {"model": "text-embedding-3-small"}

    def fail(*args: Any, **kwargs: Any) -> None:
        raise exc

    monkeypatch.setattr(Index, "load", staticmethod(lambda data_dir: StubIndex()))
    monkeypatch.setattr(cli, "Retriever", lambda index, embedder: object())
    monkeypatch.setattr(cli, "OpenAIEmbedder", lambda model: object())
    monkeypatch.setattr(cli, "make_tracer", lambda: tracer)
    monkeypatch.setattr(anthropic, "Anthropic", lambda: object())
    monkeypatch.setattr(cli, "answer_question", fail)


def test_ask_exits_with_the_message_and_flushes_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = FlushTracer()
    patch_pipeline(monkeypatch, tracer, status_error(anthropic.RateLimitError, 429))
    with pytest.raises(SystemExit) as exited:
        cli.cmd_ask(ask_args())
    assert exited.value.code == "Rate limited or spend cap reached. Try again later."
    assert tracer.flushed


def test_ask_reraises_an_api_error_without_a_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = FlushTracer()
    response = httpx2.Response(200, request=REQUEST)
    malformed = anthropic.APIResponseValidationError(response=response, body=None)
    patch_pipeline(monkeypatch, tracer, malformed)
    with pytest.raises(anthropic.APIResponseValidationError):
        cli.cmd_ask(ask_args())
    assert tracer.flushed
