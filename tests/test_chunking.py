from __future__ import annotations

import pytest

from rag_support.chunking import MAX_TOKENS, chunk_markdown, doc_url, estimate_tokens

PATH = "docs/en/docs/tutorial/first-steps.md"


def para(n_tokens: int, word: str = "lorem") -> str:
    return (word + " ") * (n_tokens * 4 // (len(word) + 1)) + "\n\n"


@pytest.mark.parametrize(
    ("path", "url"),
    [
        ("docs/en/docs/index.md", "https://fastapi.tiangolo.com/"),
        ("docs/en/docs/tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/"),
        (PATH, "https://fastapi.tiangolo.com/tutorial/first-steps/"),
    ],
)
def test_doc_url(path: str, url: str) -> None:
    assert doc_url(path) == url


def test_splits_on_headings_with_heading_path() -> None:
    text = f"# Guide\n\n{para(400)}## Install {{ #install }}\n\n{para(400)}"
    chunks = chunk_markdown(text, PATH)
    assert [c.heading for c in chunks] == ["Guide", "Guide > Install"]
    assert chunks[1].text.startswith("## Install")


def test_hash_inside_code_is_not_a_heading() -> None:
    text = f"# Run\n\n{para(400)}```bash\n# install deps\npip install x\n```\n"
    chunks = chunk_markdown(text, PATH)
    assert len(chunks) == 1
    assert "# install deps" in chunks[0].text


def test_large_section_split_without_breaking_code() -> None:
    code = "```python\n" + "x = 1\n" * 200 + "```\n\n"
    text = "# Big\n\n" + para(500) + code + para(500)
    chunks = chunk_markdown(text, PATH)
    assert len(chunks) > 1
    assert all(c.heading == "Big" for c in chunks)
    whole = [c for c in chunks if "```python" in c.text]
    assert len(whole) == 1 and whole[0].text.count("```") == 2


def test_long_list_without_blank_lines_is_split_on_lines() -> None:
    bullets = "".join(
        f"* Change number {i} with some detail text.\n" for i in range(400)
    )
    chunks = chunk_markdown("# Release notes\n\n" + bullets, PATH)
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= MAX_TOKENS for c in chunks)
    assert sum(c.text.count("* Change number") for c in chunks) == 400


def test_small_sections_merge_forward() -> None:
    text = "# A\n\nshort\n\n## B\n\nalso short\n\n## C\n\n" + para(400)
    chunks = chunk_markdown(text, PATH)
    assert len(chunks) == 1
    assert chunks[0].heading == "A"
    assert "## B" in chunks[0].text and "## C" in chunks[0].text


def test_heading_only_sections_are_kept_by_merging() -> None:
    chunks = chunk_markdown("# Title\n\n## Empty\n\n## Also empty\n", PATH)
    assert len(chunks) == 1
    assert "## Empty" in chunks[0].text


def test_empty_document_has_no_chunks() -> None:
    assert chunk_markdown("\n\n", PATH) == []


def test_chunks_stay_under_max_except_single_large_blocks() -> None:
    text = "# T\n\n" + "".join(para(200, w) for w in ("a", "bb", "ccc", "dddd", "e"))
    for chunk in chunk_markdown(text, PATH):
        assert estimate_tokens(chunk.text) <= MAX_TOKENS


def test_identical_chunks_in_one_file_are_indexed_once() -> None:
    code = "```python\n" + "data = 1\n" * 500 + "```\n\n"
    text = f"# A\n\n{para(400)}{code}## B\n\n{para(400)}{code}"
    chunks = chunk_markdown(text, PATH)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert sum(c.text == code.strip() for c in chunks) == 1


def test_chunk_ids_are_stable_and_content_addressed() -> None:
    text = "# A\n\n" + para(400)
    first = chunk_markdown(text, PATH)[0]
    assert chunk_markdown(text, PATH)[0].chunk_id == first.chunk_id
    assert chunk_markdown(text + "more\n", PATH)[0].chunk_id != first.chunk_id
