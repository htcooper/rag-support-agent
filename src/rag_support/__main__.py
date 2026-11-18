"""Command line: ``python -I -m rag_support <command>`` from the project root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_support.answer import DEFAULT_EFFORT, answer_question
from rag_support.embeddings import OpenAIEmbedder
from rag_support.index import build_index
from rag_support.ingest import pin, run_ingest
from rag_support.retrieve import Index, Retriever
from rag_support.tracing import NoopTracer, make_tracer

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


def cmd_ask(args: argparse.Namespace) -> None:
    import anthropic

    index = Index.load(DATA_DIR)
    retriever = Retriever(index, OpenAIEmbedder(model=index.info["model"]))
    tracer = make_tracer()
    if args.public and isinstance(tracer, NoopTracer):
        sys.exit("--public needs tracing: set the Langfuse keys in .env.")
    try:
        answer = answer_question(
            args.question,
            retriever,
            anthropic.Anthropic(),
            effort=args.effort,
            tracer=tracer,
            public=args.public,
            rewrite=args.rewrite,
            strict=args.strict,
        )
    except anthropic.AuthenticationError:
        sys.exit("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.PermissionDeniedError as e:
        sys.exit(f"The API key can't use this model or feature: {e.message}")
    except anthropic.RateLimitError:
        sys.exit("Rate limited or spend cap reached. Try again later.")
    except anthropic.APIStatusError as e:
        sys.exit(f"Anthropic API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Couldn't reach the Anthropic API. Check the network connection.")
    finally:
        tracer.flush()  # send the trace even when the call failed

    print(answer.text)
    print("\nSources:")
    for rank, hit in enumerate(answer.hits, 1):
        mark = "cited" if rank in answer.cited_ranks else "     "
        print(f"  [{rank}] {mark}  {hit.chunk.heading}  {hit.chunk.url}")
    if answer.rewrite is not None:
        print(f"\nSearched for: {answer.rewrite.text or '(rewrite failed)'}")
    notes = [f"model: {answer.model}", f"stop: {answer.stop_reason}"]
    notes.append(
        f"tokens in/out: {answer.usage.get('input_tokens')}"
        f"/{answer.usage.get('output_tokens')}"
    )
    if not answer.grounded:
        notes.append("not grounded (no valid citations)")
    if answer.fallback_used:
        notes.append("served by fallback model")
    if answer.invalid_citations:
        notes.append(f"invalid citations dropped: {answer.invalid_citations}")
    if answer.links_removed:
        notes.append(f"links removed: {answer.links_removed}")
    print("\n" + " | ".join(notes))
    if answer.trace_url:
        label = "public trace" if args.public else "trace (private)"
        print(f"{label}: {answer.trace_url}")


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

    p_ask = sub.add_parser("ask", help="answer a question with cited sources")
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high"]
    )
    p_ask.add_argument(
        "--public", action="store_true", help="make the trace viewable by link"
    )
    p_ask.add_argument(
        "--rewrite",
        action="store_true",
        help="restate the question in the docs' vocabulary before searching",
    )
    p_ask.add_argument(
        "--strict",
        action="store_true",
        help="forbid details the retrieved docs don't state",
    )
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
