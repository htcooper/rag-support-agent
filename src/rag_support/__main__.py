"""Command line: ``python -I -m rag_support <command>`` from the project root."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_support.ingest import pin, run_ingest

LOCK_PATH = Path("sources.lock.json")
DATA_DIR = Path("data")
ALLOWLIST_PATH = Path("quarantine_allowlist.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_support")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pin = sub.add_parser("pin", help="pin a commit and record its archive hash")
    p_pin.add_argument("--name", required=True)
    p_pin.add_argument("--repo", required=True)
    p_pin.add_argument("--ref", required=True, help="release tag, for reference")
    p_pin.add_argument("--commit", required=True, help="full 40-character SHA")

    sub.add_parser("ingest", help="verify, extract, sanitize, chunk, and screen")

    args = parser.parse_args(argv)
    if args.command == "pin":
        source = pin(LOCK_PATH, DATA_DIR, args.name, args.repo, args.ref, args.commit)
        print(f"Pinned {source.repo}@{source.ref} ({source.commit})")
        print(f"archive sha256: {source.archive_sha256}")
    elif args.command == "ingest":
        report = run_ingest(LOCK_PATH, DATA_DIR, ALLOWLIST_PATH)
        print(f"files extracted:   {report['files_extracted']}")
        print(f"  docs / code:     {report['doc_files']} / {report['code_files']}")
        print(f"files rejected:    {len(report['files_rejected'])}")
        print(f"includes inlined:  {report['includes_resolved']}")
        print(f"includes left:     {len(report['includes_unresolved'])}")
        print(f"chunks indexed:    {report['chunks_indexed']}")
        print(f"chunks quarantined: {report['chunks_quarantined']}")
        print(f"sanitizer totals:  {report['sanitizer_totals']}")
        print(f"report: {DATA_DIR / 'ingest_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
