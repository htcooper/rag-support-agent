"""Test doubles that need no network or API keys."""

from __future__ import annotations

import re
import zlib

import numpy as np

from rag_support.embeddings import Vectors, normalize


class HashingEmbedder:
    """Bag-of-words vectors: texts that share words score as similar.

    Uses crc32 rather than ``hash()`` so vectors are the same in every process.
    """

    def __init__(self, dims: int = 512, model: str = "fake-hashing") -> None:
        self.dims = dims
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> Vectors:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % self.dims] += 1.0
        return normalize(vectors)
