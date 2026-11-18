"""Compare retrieval with and without query rewriting.

Run with python -I evals/retrieval_compare.py from the project root.

Scores every should-answer question in cases.jsonl plus the wording-variation
set in wording.jsonl. No answers are generated and nothing is judged, so the
only costs are one Haiku rewrite and two small embedding calls per query.
Writes evals/retrieval.md and a per-query file in the eval folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from rag_support.chunking import DOCS_BASE_URL
from rag_support.embeddings import OpenAIEmbedder
from rag_support.evaluation import (
    RetrievalResult,
    first_expected_rank,
    load_cases,
    retrieval_summary,
)
from rag_support.harness import FLOW_DIR, check_harness
from rag_support.retrieve import Index, Retriever
from rag_support.rewrite import rewrite_query, search_text
from rag_support.tracing import cost_details

K = 5
OUT_MD = Path("evals/retrieval.md")
OUT_JSONL = FLOW_DIR / "retrieval" / "results.jsonl"
LEVEL_LABELS = {"docs": "Docs' own words", "user": "User's words", "vague": "Vague"}


def load_queries() -> list[dict[str, Any]]:
    queries = [
        {
            "id": c.id,
            "group": c.split,
            "question": c.question,
            "expected": c.expected_urls,
        }
        for c in load_cases(Path("evals/cases.jsonl"))
        if c.kind == "in_scope"
    ]
    for line in Path("evals/wording.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            queries.append(
                {
                    "id": row["id"],
                    "group": row["level"],
                    "question": row["question"],
                    "expected": tuple(DOCS_BASE_URL + u for u in row["expected_urls"]),
                }
            )
    return queries


def fmt(summary: dict[str, float], key: str) -> str:
    n = int(summary["n"])
    return f"{int(summary[key])}/{n}"


def report(results: list[RetrievalResult], cost: float) -> str:
    s = retrieval_summary(results)
    lines = [
        "# Retrieval with and without query rewriting",
        "",
        "Before: the question is embedded as asked. After: Claude Haiku 5.5 first "
        "restates it in FastAPI's vocabulary, and the search uses the question "
        "plus the rewrite. Hit@5 counts queries with an expected page in the top "
        "5; MRR (0-1) rewards ranking it higher.",
        "",
        "| Queries | n | Hit@5 before | Hit@5 after | MRR before | MRR after |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "dev": "Should-answer, dev",
        "heldout": "Should-answer, held-out",
        **LEVEL_LABELS,
    }
    for group, label in labels.items():
        if group in s:
            g = s[group]
            lines.append(
                f"| {label} | {int(g['n'])} | {fmt(g, 'hits_before')} | "
                f"{fmt(g, 'hits_after')} | {g['mrr_before']:.2f} | "
                f"{g['mrr_after']:.2f} |"
            )
    for change in ("fixed", "broken"):
        changed = [r for r in results if r.change == change]
        lines += ["", f"## {change.capitalize()} by rewriting ({len(changed)})", ""]
        lines += [f"- `{r.id}`: {r.question} -> *{r.rewrite}*" for r in changed]
        if not changed:
            lines.append("None.")
    lines += ["", f"Cost of this comparison: ${cost:.4f}.", ""]
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-harness", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(".env"))
    check_harness(args.approve_harness)

    index = Index.load(Path("data"))
    embedder = OpenAIEmbedder(model=index.info["model"])
    retriever = Retriever(index, embedder)
    client = anthropic.Anthropic()

    results: list[RetrievalResult] = []
    cost = 0.0
    for q in load_queries():
        before = retriever.retrieve(q["question"], k=K)
        rewrite = rewrite_query(q["question"], client)
        after = retriever.retrieve(search_text(q["question"], rewrite), k=K)
        cost += cost_details(
            rewrite.model,
            rewrite.usage.get("input_tokens", 0),
            rewrite.usage.get("output_tokens", 0),
        ).get("total", 0.0)
        results.append(
            RetrievalResult(
                id=q["id"],
                group=q["group"],
                question=q["question"],
                rewrite=rewrite.text,
                rank_before=first_expected_rank(before, q["expected"]),
                rank_after=first_expected_rank(after, q["expected"]),
            )
        )
        print(f"  {q['id']:16} {results[-1].change}")
    cost += cost_details(embedder.model, embedder.tokens_used).get("total", 0.0)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps({**r.__dict__, "change": r.change}) + "\n")
    text = report(results, cost)
    OUT_MD.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
