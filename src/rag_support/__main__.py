"""Command line: ``python -I -m rag_support <command>`` from the project root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_support.embeddings import OpenAIEmbedder
from rag_support.index import build_index
from rag_support.ingest import pin, run_ingest
from rag_support.retrieve import Index, Retriever

LOCK_PATH = Path("sources.lock.json")
DATA_DIR = Path("data")
ALLOWLIST_PATH = Path("quarantine_allowlist.json")
PREVIEW_CHARS = 160


def cmd_pin(args: argparse.Namespace) -> None:
    source = pin(LOCK_PATH, DATA_DIR, args.name, args.repo, args.ref, args.commit)
    print(f"Pinned {source.repo}@{source.ref} ({source.commit})")
    print(f"archive sha256: {source.archive_sha256}")


def cmd_ingest(args: argparse.Namespace) -> None:
    report = run_ingest(LOCK_PATH, DATA_DIR, ALLOWLIST_PATH)
    print(f"files extracted:    {report['files_extracted']}")
    print(f"  docs / code:      {report['doc_files']} / {report['code_files']}")
    print(f"files rejected:     {len(report['files_rejected'])}")
    print(f"includes inlined:   {report['includes_resolved']}")
    print(f"includes left:      {len(report['includes_unresolved'])}")
    print(f"chunks indexed:     {report['chunks_indexed']}")
    print(f"chunks quarantined: {report['chunks_quarantined']}")
    print(f"sanitizer totals:   {report['sanitizer_totals']}")
    print(f"report: {DATA_DIR / 'ingest_report.json'}")


def cmd_index(args: argparse.Namespace) -> None:
    info = build_index(DATA_DIR, OpenAIEmbedder())
    print(f"indexed {info['count']} chunks with {info['model']}")
    print(f"dimensions: {info['dimensions']}, tokens billed: {info['tokens_used']}")


def cmd_search(args: argparse.Namespace) -> None:
    index = Index.load(DATA_DIR)
    retriever = Retriever(index, OpenAIEmbedder(model=index.info["model"]))
    for rank, hit in enumerate(retriever.retrieve(args.query, k=args.k), 1):
        preview = " ".join(hit.chunk.text.split())[:PREVIEW_CHARS]
        print(f"{rank}. [{hit.score:.3f}] {hit.chunk.heading}")
        print(f"   {hit.chunk.url}")
        print(f"   {preview}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(prog="rag_support")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pin = sub.add_parser("pin", help="pin a commit and record its archive hash")
    p_pin.add_argument("--name", required=True)
    p_pin.add_argument("--repo", required=True)
    p_pin.add_argument("--ref", required=True, help="release tag, for reference")
    p_pin.add_argument("--commit", required=True, help="full 40-character SHA")
    p_pin.set_defaults(func=cmd_pin)

    p_ingest = sub.add_parser("ingest", help="verify, extract, sanitize, chunk, screen")
    p_ingest.set_defaults(func=cmd_ingest)

    p_index = sub.add_parser("index", help="embed chunks and save the vector index")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="show the chunks closest to a question")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
