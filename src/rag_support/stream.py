"""Stream the answer's text to the web UI while Claude writes it.

answer_question is reused unchanged: StreamingClient wraps the Anthropic client
so the generate call is sent as a stream, and returns the same final Message
that parse_response reads. The request, the trace, and the validated answer are
the same as without streaming.

Security model: streamed text is model output, like the final answer, so the
link allowlist applies before any of it leaves the server. FilteredText holds
back the last, unfinished word (a URL can't be checked until it's complete) and
runs filter_links on everything released so far. Citations are not streamed;
they arrive only with the final, validated answer.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from rag_support.sanitize import filter_links


class FilteredText:
    """Collects streamed text and returns snapshots that are safe to show.

    A snapshot is the text up to the last whitespace, with filter_links applied.
    A URL or Markdown link target can't contain whitespace, so the released text
    never ends partway through one: "https://github.com" can't be shown and then
    turn out to be "https://github.com.evil.example". Each snapshot replaces the
    one before, since filtering can change earlier text once a link closes.
    """

    def __init__(self) -> None:
        self._raw = ""
        self._released = 0
        self._last = ""

    @property
    def raw(self) -> str:
        return self._raw

    def add(self, delta: str) -> str | None:
        """Add a delta; return a new snapshot, or None if nothing new is safe."""
        self._raw += delta
        end = len(self._raw)
        while end > self._released and not self._raw[end - 1].isspace():
            end -= 1
        if end <= self._released:
            return None
        self._released = end
        snapshot, _ = filter_links(self._raw[:end])
        if snapshot == self._last:
            return None
        self._last = snapshot
        return snapshot


class StreamingClient:
    """An Anthropic client whose beta.messages.create streams the response.

    answer_question calls ``client.beta.messages.create(**request)`` for the
    answer and ``client.messages.create`` for the rewrite. The rewrite is passed
    through; the answer is sent with ``beta.messages.stream(**request)``, each
    text delta goes to ``on_text``, and the final Message is returned.
    """

    def __init__(
        self,
        client: Any,
        on_text: Callable[[str], None],
        on_generate: Callable[[], None] = lambda: None,
    ) -> None:
        self._client = client
        self._on_text = on_text
        self._on_generate = on_generate
        self.messages = client.messages
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        self._on_generate()
        with self._client.beta.messages.stream(**request) as stream:
            for text in stream.text_stream:
                self._on_text(text)
            return stream.get_final_message()
