"""Text embedders. Real runs use OpenAI; tests use a deterministic fake."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "text-embedding-3-small"
Vectors = NDArray[np.float32]


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> Vectors: ...


def normalize(vectors: NDArray[Any]) -> Vectors:
    """Scale rows to unit length so a dot product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit: Vectors = (vectors / norms).astype(np.float32)
    return unit


class OpenAIEmbedder:
    """Embeds in batches and keeps a running count of tokens billed."""

    def __init__(
        self, model: str = DEFAULT_MODEL, batch_size: int = 100, client: Any = None
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.tokens_used = 0
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    def embed(self, texts: list[str]) -> Vectors:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            self.tokens_used += response.usage.total_tokens
            rows.extend(
                d.embedding for d in sorted(response.data, key=lambda d: d.index)
            )
        if len(rows) != len(texts):
            raise RuntimeError(f"expected {len(texts)} embeddings, got {len(rows)}")
        return normalize(np.array(rows, dtype=np.float32).reshape(len(texts), -1))
