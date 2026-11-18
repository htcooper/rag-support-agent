"""Harness gate: eval scripts refuse to run on code that hasn't been approved.

A fingerprint of every file that defines what the eval measures is compared
with the one recorded by ``--approve-harness``. Any change to those files, for
example a grader tweak that would flatter the scores, stops the eval until a
person reviews and approves it again.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FLOW_DIR = Path(".claude/hillclimb/support-answer")
APPROVAL_PATH = FLOW_DIR / "harness_approval.json"

HARNESS_PATHS = [
    "evals/run.py",
    "evals/retrieval_compare.py",
    "evals/cases.jsonl",
    "evals/wording.jsonl",
    "src/rag_support/evaluation.py",
    "src/rag_support/harness.py",
    "src/rag_support/answer.py",
    "src/rag_support/rewrite.py",
    "src/rag_support/retrieve.py",
    "src/rag_support/sanitize.py",
    "data/index_info.json",
]


def harness_sha(paths: list[str] = HARNESS_PATHS) -> str:
    digest = hashlib.sha256()
    for rel in paths:
        digest.update(rel.encode() + b"\0" + Path(rel).read_bytes() + b"\0")
    return digest.hexdigest()


def check_harness(approve: bool) -> None:
    """With approve, record the fingerprint and exit; otherwise exit 2 on mismatch."""
    sha = harness_sha()
    if approve:
        FLOW_DIR.mkdir(parents=True, exist_ok=True)
        APPROVAL_PATH.write_text(
            json.dumps({"sha256": sha}) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Harness approved: {sha[:16]}. Run again without --approve-harness.")
        sys.exit(0)
    approved = None
    if APPROVAL_PATH.exists():
        approved = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))["sha256"]
    if approved != sha:
        print(
            "The eval code changed since it was last approved (or was never\n"
            "approved). Review the change, then run once with --approve-harness.",
            file=sys.stderr,
        )
        sys.exit(2)
