from __future__ import annotations

import argparse

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
