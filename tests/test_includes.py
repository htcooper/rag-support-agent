from __future__ import annotations

import pytest

from rag_support.includes import expand_includes, resolve_target

CODE = {"docs_src/first_steps/tutorial001.py": "from fastapi import FastAPI\n"}


def test_inlines_code_as_a_python_block() -> None:
    text = "Intro\n\n{* ../../docs_src/first_steps/tutorial001.py hl[1] *}\n\nAfter\n"
    result = expand_includes(text, CODE)
    assert result.resolved == 1 and result.unresolved == []
    assert result.text == (
        "Intro\n\n```python\nfrom fastapi import FastAPI\n```\n\nAfter\n"
    )


@pytest.mark.parametrize(
    "target",
    [
        "../../docs_src/../setup.py",  # normalizes outside docs_src
        "../../fastapi/openapi/docs.py",  # library source, not an example
        "/etc/passwd",
        "..\\..\\docs_src\\first_steps\\tutorial001.py",
        "../../../docs_src/first_steps/tutorial001.py",  # escapes the repo
    ],
)
def test_rejects_targets_outside_docs_src(target: str) -> None:
    assert resolve_target(target) is None


def test_leaves_unresolvable_directives_unchanged() -> None:
    text = "{* ../../docs_src/missing.py *}\n{* ../../fastapi/openapi/docs.py *}\n"
    result = expand_includes(text, CODE)
    assert result.text == text
    assert result.unresolved == [
        "../../docs_src/missing.py",
        "../../fastapi/openapi/docs.py",
    ]


def test_code_containing_backticks_uses_a_tilde_fence() -> None:
    code = {"docs_src/a.py": 'doc = """\n```\nexample\n```\n"""\n'}
    result = expand_includes("{* ../../docs_src/a.py *}", code)
    assert result.text.startswith("~~~python\n") and result.text.endswith("\n~~~")


def test_code_containing_both_fence_styles_is_not_inlined() -> None:
    code = {"docs_src/a.py": "s = '```'\nt = '~~~'\n"}
    result = expand_includes("{* ../../docs_src/a.py *}", code)
    assert result.resolved == 0 and result.unresolved == ["../../docs_src/a.py"]
