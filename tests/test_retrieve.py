from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rag_support.chunking import Chunk
from rag_support.index import build_index, embed_text
from rag_support.retrieve import Index, Retriever, StaleIndexError
from tests.fakes import HashingEmbedder

TOPICS = {
    "cors": ("Middleware > CORS", "Add CORSMiddleware to allow cross origin requests"),
    "oauth": ("Security > OAuth2", "Use OAuth2PasswordBearer to read a bearer token"),
    "files": ("Request Files", "Receive uploaded files with UploadFile and File"),
    "bg": ("Background Tasks", "Run work after returning a response with tasks"),
}


def make_chunk(key: str, heading: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=key,
        source_path=f"docs/en/docs/{key}.md",
        heading=heading,
        url=f"https://fastapi.tiangolo.com/{key}/",
        text=text,
        sha256="0" * 64,
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for key, (heading, text) in TOPICS.items():
            f.write(json.dumps(asdict(make_chunk(key, heading, text))) + "\n")
    return tmp_path


def test_build_index_writes_vectors_and_info(data_dir: Path) -> None:
    embedder = HashingEmbedder(dims=64)
    info = build_index(data_dir, embedder)

    assert info["model"] == "fake-hashing"
    assert info["dimensions"] == 64 and info["count"] == 4
    assert np.load(data_dir / "embeddings.npy").shape == (4, 64)
    # The heading path is embedded along with the text.
    assert embedder.calls[0][1].startswith("Security > OAuth2\n\n")


def test_retrieve_ranks_the_matching_chunk_first(data_dir: Path) -> None:
    embedder = HashingEmbedder()
    build_index(data_dir, embedder)
    retriever = Retriever(Index.load(data_dir), embedder)

    hits = retriever.retrieve("How do I accept uploaded files?", k=2)
    assert [h.chunk.chunk_id for h in hits][0] == "files"
    assert len(hits) == 2 and hits[0].score >= hits[1].score


def test_k_larger_than_index_returns_everything(data_dir: Path) -> None:
    embedder = HashingEmbedder()
    build_index(data_dir, embedder)
    hits = Retriever(Index.load(data_dir), embedder).retrieve("tasks", k=50)
    assert len(hits) == 4


def test_stale_index_is_refused(data_dir: Path) -> None:
    build_index(data_dir, HashingEmbedder())
    with (data_dir / "chunks.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(make_chunk("new", "New", "new text"))) + "\n")
    with pytest.raises(StaleIndexError, match="changed since indexing"):
        Index.load(data_dir)


def test_embedder_must_match_index_model(data_dir: Path) -> None:
    build_index(data_dir, HashingEmbedder())
    with pytest.raises(ValueError, match="don't mix"):
        Retriever(Index.load(data_dir), HashingEmbedder(model="other-model"))


def test_empty_query_is_rejected(data_dir: Path) -> None:
    embedder = HashingEmbedder()
    build_index(data_dir, embedder)
    with pytest.raises(ValueError, match="empty"):
        Retriever(Index.load(data_dir), embedder).retrieve("   ")


def test_pickled_vectors_file_is_refused(data_dir: Path) -> None:
    build_index(data_dir, HashingEmbedder())
    np.save(data_dir / "embeddings.npy", np.array([object()]), allow_pickle=True)
    with pytest.raises(ValueError, match="allow_pickle"):
        Index.load(data_dir)


def test_build_index_refuses_empty_chunks(tmp_path: Path) -> None:
    (tmp_path / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no chunks"):
        build_index(tmp_path, HashingEmbedder())


def test_embed_text_without_heading_is_just_text() -> None:
    assert embed_text(make_chunk("x", "", "body")) == "body"
