"""Build the vector index for ``chunks.jsonl``.

Writes ``embeddings.npy`` (row n is line n of ``chunks.jsonl``) and
``index_info.json``, which records the embedding model and a hash of the chunks
file so a stale index is detected instead of returning the wrong chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_support.chunking import Chunk
from rag_support.embeddings import Embedder
from rag_support.sources import sha256_file

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "embeddings.npy"
INFO_FILE = "index_info.json"


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def embed_text(chunk: Chunk) -> str:
    """The heading path gives each chunk context, e.g. 'Security > OAuth2'."""
    return f"{chunk.heading}\n\n{chunk.text}" if chunk.heading else chunk.text


def build_index(data_dir: Path, embedder: Embedder) -> dict[str, Any]:
    chunks_path = data_dir / CHUNKS_FILE
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"no chunks in {chunks_path}; run ingest first")
    vectors = embedder.embed([embed_text(c) for c in chunks])

    np.save(data_dir / VECTORS_FILE, vectors, allow_pickle=False)
    info: dict[str, Any] = {
        "model": embedder.model,
        "dimensions": int(vectors.shape[1]),
        "count": len(chunks),
        "chunks_sha256": sha256_file(chunks_path),
        "tokens_used": getattr(embedder, "tokens_used", None),
    }
    (data_dir / INFO_FILE).write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return info
