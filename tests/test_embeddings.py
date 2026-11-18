from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rag_support.embeddings import OpenAIEmbedder, normalize


class FakeEmbeddingsAPI:
    """Mimics client.embeddings.create, returning rows out of order."""

    def __init__(self, dims: int = 3, drop: int = 0) -> None:
        self.dims = dims
        self.drop = drop
        self.batches: list[list[str]] = []

    def create(self, model: str, input: list[str]) -> Any:
        self.batches.append(input)
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0, 0.0])
            for i, text in enumerate(input)
        ][self.drop :]
        return SimpleNamespace(
            data=list(reversed(data)),
            usage=SimpleNamespace(total_tokens=10 * len(input)),
        )


def test_normalize_gives_unit_rows_and_keeps_zero_rows() -> None:
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_openai_embedder_batches_restores_order_and_counts_tokens() -> None:
    api = FakeEmbeddingsAPI()
    embedder = OpenAIEmbedder(batch_size=2, client=SimpleNamespace(embeddings=api))
    texts = ["a", "bbb", "cc", "dddd", "e"]
    vectors = embedder.embed(texts)

    assert [len(b) for b in api.batches] == [2, 2, 1]
    assert embedder.tokens_used == 50
    assert vectors.shape == (5, 3)
    expected = normalize(np.array([[len(t), 1.0, 0.0] for t in texts]))
    np.testing.assert_allclose(vectors, expected, rtol=1e-6)


def test_openai_embedder_fails_if_rows_are_missing() -> None:
    api = FakeEmbeddingsAPI(drop=1)
    embedder = OpenAIEmbedder(client=SimpleNamespace(embeddings=api))
    with pytest.raises(RuntimeError, match="expected 2 embeddings, got 1"):
        embedder.embed(["a", "b"])
