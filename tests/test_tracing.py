from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from rag_support.tracing import (
    LangfuseTracer,
    NoopTracer,
    cost_details,
    langfuse_configured,
    make_tracer,
)


def test_cost_details_for_known_and_unknown_models() -> None:
    cost = cost_details("claude-opus-5-5", 1_000_000, 100_000)
    assert cost == pytest.approx({"input": 4.0, "output": 2.0, "total": 6.0})
    assert cost_details("text-embedding-3-small", 500_000)["total"] == pytest.approx(
        0.01
    )
    assert cost_details("some-unknown-model", 100, 100) == {}


@pytest.mark.parametrize(
    ("public_key", "secret_key", "expected"),
    [("pk-lf-1", "sk-lf-1", True), ("pk-lf-1", "", False), ("", "sk-lf-1", False)],
)
def test_tracing_needs_both_keys(
    monkeypatch: pytest.MonkeyPatch, public_key: str, secret_key: str, expected: bool
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)
    assert langfuse_configured() is expected


def test_make_tracer_is_noop_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = make_tracer()
    assert isinstance(tracer, NoopTracer)
    with tracer.span("x", as_type="generation", input="q") as span:
        span.update(output="a")
    assert tracer.trace_url() is None


def test_langfuse_exports_only_this_projects_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Other libraries' spans (e.g. the Anthropic SDK's) carry account IDs."""
    import langfuse

    captured: dict[str, Any] = {}
    monkeypatch.setattr(langfuse, "Langfuse", lambda **kw: captured.update(kw))
    LangfuseTracer()
    assert captured["should_export_span"] is langfuse.is_langfuse_span


class FakeLangfuse:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Iterator[str]:
        self.calls.append(("start", kwargs))
        yield "observation"

    def set_current_trace_as_public(self) -> None:
        self.calls.append(("public", None))

    def get_current_trace_id(self) -> str:
        return "abc123"

    def get_trace_url(self, trace_id: str) -> str:
        return f"https://us.cloud.langfuse.com/project/p/traces/{trace_id}"

    def flush(self) -> None:
        self.calls.append(("flush", None))


def test_langfuse_tracer_delegates_to_the_client() -> None:
    lf = FakeLangfuse()
    tracer = LangfuseTracer(client=lf)
    with tracer.span("retrieve", as_type="retriever", input={"k": 5}):
        tracer.make_public()
        url = tracer.trace_url()
    tracer.flush()
    assert lf.calls == [
        ("start", {"name": "retrieve", "as_type": "retriever", "input": {"k": 5}}),
        ("public", None),
        ("flush", None),
    ]
    assert url == "https://us.cloud.langfuse.com/project/p/traces/abc123"
