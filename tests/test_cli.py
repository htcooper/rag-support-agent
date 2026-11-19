from __future__ import annotations

import argparse
from typing import Any

import pytest

from rag_support import __main__ as cli


def parsed_ask(monkeypatch: pytest.MonkeyPatch, *flags: str) -> argparse.Namespace:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "cmd_ask", seen.append)
    cli.main(["ask", "How do I enable CORS?", *flags])
    return seen[0]


def test_ask_uses_the_measured_improvements_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = parsed_ask(monkeypatch)
    assert args.rewrite is True and args.strict is True


def test_ask_can_turn_each_improvement_off(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parsed_ask(monkeypatch, "--no-rewrite", "--no-strict")
    assert args.rewrite is False and args.strict is False


def test_ui_serves_on_localhost_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import uvicorn

    from rag_support.retrieve import Index

    class StubIndex:
        info = {"model": "text-embedding-3-small"}

    served: dict[str, Any] = {}
    monkeypatch.setattr(Index, "load", staticmethod(lambda data_dir: StubIndex()))
    monkeypatch.setattr(cli, "Retriever", lambda index, embedder: object())
    monkeypatch.setattr(cli, "OpenAIEmbedder", lambda model: object())
    monkeypatch.setattr(anthropic, "Anthropic", lambda: object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(kw))
    cli.main(["ui", "--port", "8123"])
    assert served["host"] == "127.0.0.1"
    assert served["port"] == 8123
