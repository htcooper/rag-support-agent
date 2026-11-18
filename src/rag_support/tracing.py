"""Tracing: one Langfuse trace per question, or nothing when Langfuse isn't set up.

The pipeline talks to the small ``Tracer`` interface below, so the code and the
tests don't depend on Langfuse. Tracing is on only when both Langfuse keys are in
the environment. Spans carry question and document data only, never settings or
keys.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

# US dollars per million tokens (input, output), from the providers' price lists
# on 2026-11-17. Used to show the cost of each step in a trace.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "text-embedding-3-small": (0.02, 0.0),
}


def cost_details(
    model: str, input_tokens: int, output_tokens: int = 0
) -> dict[str, float]:
    """Dollar cost of one call, or {} for a model without a known price."""
    if model not in PRICES_PER_MTOK:
        return {}
    in_price, out_price = PRICES_PER_MTOK[model]
    cost_in = input_tokens * in_price / 1e6
    cost_out = output_tokens * out_price / 1e6
    return {"input": cost_in, "output": cost_out, "total": cost_in + cost_out}


class Span(Protocol):
    def update(self, **fields: Any) -> Any: ...


class Tracer(Protocol):
    def span(
        self, name: str, as_type: str = "span", **fields: Any
    ) -> AbstractContextManager[Span]: ...

    def make_public(self) -> None: ...

    def trace_id(self) -> str | None: ...

    def trace_url(self) -> str | None: ...

    def score(
        self, trace_id: str, name: str, value: float, comment: str | None = None
    ) -> None: ...

    def flush(self) -> None: ...


class _NoopSpan:
    def update(self, **fields: Any) -> _NoopSpan:
        return self


class NoopTracer:
    """Used when Langfuse isn't configured: every call does nothing."""

    @contextmanager
    def span(self, name: str, as_type: str = "span", **fields: Any) -> Iterator[Span]:
        yield _NoopSpan()

    def make_public(self) -> None:
        pass

    def trace_id(self) -> str | None:
        return None

    def trace_url(self) -> str | None:
        return None

    def score(
        self, trace_id: str, name: str, value: float, comment: str | None = None
    ) -> None:
        pass

    def flush(self) -> None:
        pass


class LangfuseTracer:
    def __init__(self, client: Any = None) -> None:
        if client is None:
            from langfuse import Langfuse, is_langfuse_span

            # Export only this project's spans. By default Langfuse also exports
            # spans from other libraries' OpenTelemetry hooks; the Anthropic SDK's
            # span carries the account's organization and workspace IDs and would
            # count each Claude call's cost a second time.
            client = Langfuse(should_export_span=is_langfuse_span)
        self._lf = client

    @contextmanager
    def span(self, name: str, as_type: str = "span", **fields: Any) -> Iterator[Span]:
        with self._lf.start_as_current_observation(
            name=name, as_type=as_type, **fields
        ) as observation:
            yield observation

    def make_public(self) -> None:
        """Make the current trace viewable by anyone with its link."""
        self._lf.set_current_trace_as_public()

    def trace_id(self) -> str | None:
        trace_id = self._lf.get_current_trace_id()
        return str(trace_id) if trace_id else None

    def trace_url(self) -> str | None:
        trace_id = self.trace_id()
        return str(self._lf.get_trace_url(trace_id=trace_id)) if trace_id else None

    def score(
        self, trace_id: str, name: str, value: float, comment: str | None = None
    ) -> None:
        """Attach an eval score to a trace (shown in the trace's Scores tab)."""
        self._lf.create_score(
            trace_id=trace_id, name=name, value=value, comment=comment
        )

    def flush(self) -> None:
        self._lf.flush()


def langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def make_tracer() -> Tracer:
    return LangfuseTracer() if langfuse_configured() else NoopTracer()
