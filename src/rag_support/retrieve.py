"""Load the vector index and return the chunks closest to a question."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rag_support.chunking import Chunk
from rag_support.embeddings import Embedder, Vectors
from rag_support.index import CHUNKS_FILE, INFO_FILE, VECTORS_FILE, load_chunks
from rag_support.sources import sha256_file


class StaleIndexError(Exception):
    """The index does not match the current chunks file."""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class SupportsRetrieval(Protocol):
    """What answer_question needs from a retriever (the eval wraps one)."""

    embedder: Embedder

    def embed_query(self, query: str) -> Vectors: ...

    def search(self, query_vector: Vectors, k: int = 5) -> list[Hit]: ...


class Index:
    def __init__(
        self, chunks: list[Chunk], vectors: Vectors, info: dict[str, Any]
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise StaleIndexError(
                f"{len(chunks)} chunks but {vectors.shape[0]} vectors"
            )
        self.chunks = chunks
        self.vectors = vectors
        self.info = info

    @classmethod
    def load(cls, data_dir: Path) -> Index:
        info: dict[str, Any] = json.loads(
            (data_dir / INFO_FILE).read_text(encoding="utf-8")
        )
        chunks_path = data_dir / CHUNKS_FILE
        if sha256_file(chunks_path) != info["chunks_sha256"]:
            raise StaleIndexError("chunks.jsonl changed since indexing; rebuild it")
        # allow_pickle=False: a tampered .npy file can't run code on load.
        vectors = np.load(data_dir / VECTORS_FILE, allow_pickle=False)
        return cls(load_chunks(chunks_path), vectors, info)

    def search(self, query_vector: Vectors, k: int) -> list[Hit]:
        scores = self.vectors @ query_vector
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in top]


class Retriever:
    def __init__(self, index: Index, embedder: Embedder) -> None:
        if embedder.model != index.info["model"]:
            raise ValueError(
                f"index was built with {index.info['model']!r}, "
                f"not {embedder.model!r}; vectors from different models don't mix"
            )
        self.index = index
        self.embedder = embedder

    def embed_query(self, query: str) -> Vectors:
        if not query.strip():
            raise ValueError("query is empty")
        vector: Vectors = self.embedder.embed([query])[0]
        return vector

    def search(self, query_vector: Vectors, k: int = 5) -> list[Hit]:
        return self.index.search(query_vector, k)

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        return self.search(self.embed_query(query), k)
