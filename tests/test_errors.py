from __future__ import annotations

import argparse
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from rag_support import __main__ as cli
from rag_support.errors import describe_api_error, http_status_for
from rag_support.retrieve import Index

# Both SDKs build their errors from httpx2 requests and responses.
REQUEST = httpx2.Request("POST", "https://api.example/v1")

StatusError = type[anthropic.APIStatusError] | type[openai.APIStatusError]


def status_error(cls: StatusError, code: int) -> Exception:
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
        (
            status_error(openai.AuthenticationError, 401),
            "OpenAI rejected the API key. Check OPENAI_API_KEY in .env.",
        ),
        (
            status_error(openai.PermissionDeniedError, 403),
            "The OpenAI key can't use the embedding model: server said no",
        ),
        (
            status_error(openai.RateLimitError, 429),
            "OpenAI rate limit or quota reached. Try again later.",
        ),
        (
            status_error(openai.InternalServerError, 500),
            "OpenAI API error 500: server said no",
        ),
        (
            openai.APITimeoutError(request=REQUEST),
            "Couldn't reach the OpenAI API. Check the network connection.",
        ),
    ],
)
def test_each_api_error_gets_its_message(exc: Exception, expected: str) -> None:
    assert describe_api_error(exc) == expected


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (status_error(anthropic.RateLimitError, 429), 429),
        (status_error(openai.RateLimitError, 429), 429),
        (anthropic.APIConnectionError(request=REQUEST), 503),
        (openai.APIConnectionError(request=REQUEST), 503),
        (status_error(anthropic.AuthenticationError, 401), 502),
        (status_error(openai.InternalServerError, 500), 502),
    ],
)
def test_http_status_for_each_kind_of_api_error(exc: Exception, status: int) -> None:
    assert http_status_for(exc) == status


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
