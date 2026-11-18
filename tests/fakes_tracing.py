"""A tracer that records spans in memory, for tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordedSpan:
    name: str
    as_type: str
    fields: dict[str, Any]
    parent: str | None
    updates: dict[str, Any] = field(default_factory=dict)

    def update(self, **fields: Any) -> RecordedSpan:
        self.updates.update(fields)
        return self


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []
        self.public = False
        self.flushed = False
        self.scores: list[tuple[str, str, float, str | None]] = []
        self._stack: list[str] = []

    @contextmanager
    def span(
        self, name: str, as_type: str = "span", **fields: Any
    ) -> Iterator[RecordedSpan]:
        span = RecordedSpan(
            name, as_type, fields, self._stack[-1] if self._stack else None
        )
        self.spans.append(span)
        self._stack.append(name)
        try:
            yield span
        finally:
            self._stack.pop()

    def make_public(self) -> None:
        self.public = True

    def trace_id(self) -> str | None:
        return "trace-1"

    def trace_url(self) -> str | None:
        return "https://langfuse.example/trace/1"

    def score(
        self, trace_id: str, name: str, value: float, comment: str | None = None
    ) -> None:
        self.scores.append((trace_id, name, value, comment))

    def flush(self) -> None:
        self.flushed = True

    def by_name(self, name: str) -> RecordedSpan:
        return next(s for s in self.spans if s.name == name)
