"""Write evals/results.md from a run: python -I evals/summarize.py [variant].

Every number is recomputed from the variant's results.jsonl and errors.jsonl.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_support.evaluation import wilson_interval
from rag_support.tracing import cost_details

FLOW_DIR = Path(".claude/hillclimb/support-answer")
OUT = Path("evals/results.md")
KIND_LABELS = {
    "in_scope": "Should answer",
    "out_of_scope": "Should decline",
    "injection": "Prompt injection",
}


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def rate(values: list[float]) -> str:
    if not values:
        return "n/a"
    passes = int(sum(values))
    lo, hi = wilson_interval(passes, len(values))
    return f"{passes}/{len(values)} ({passes / len(values):.0%}; CI {lo:.0%}-{hi:.0%})"


def split_of(row: dict[str, Any]) -> str:
    """Rows recorded before splits existed carry two tags; they are dev cases."""
    tags = row["tags"]
    return str(tags[2]) if len(tags) > 2 else "dev"


def check_row(rows: list[dict[str, Any]], label: str, metric: str) -> str:
    return f"| {label} | {rate([r['grade'][metric] for r in rows])} |"


def row_cost(r: dict[str, Any]) -> tuple[float, float]:
    answer = cost_details(
        r["model"], r["usage"]["input_tokens"], r["usage"]["output_tokens"]
    )
    judge: dict[str, float] = {}
    if r.get("judge_usage"):
        u = r["judge_usage"]
        judge = cost_details(r["judge_model"], u["input_tokens"], u["output_tokens"])
    rewrite: dict[str, float] = {}
    if r.get("rewrite_usage"):  # part of answering, so counted with the answer
        u = r["rewrite_usage"]
        rewrite = cost_details(
            r["rewrite_model"], u["input_tokens"], u["output_tokens"]
        )
    pipeline = answer.get("total", 0.0) + rewrite.get("total", 0.0)
    return pipeline, judge.get("total", 0.0)


def failure_reason(r: dict[str, Any]) -> str:
    g = r["grade"]
    if r["tags"][0] == "in_scope":
        if not g.get("hit_at_5"):
            return "retrieval missed every expected page"
        if not g.get("cites_expected"):
            return "didn't cite an expected page"
        if not g.get("supported"):
            return "judge: claims not backed by the retrieved docs"
        return "judge: doesn't answer the question"
    if r["tags"][0] == "out_of_scope":
        return "answered instead of declining"
    if not g.get("resisted"):
        return "followed the planted instruction"
    return "didn't give a cited answer to the real question"


def summarize(variant: str) -> str:
    vdir = FLOW_DIR / variant
    rows = [r for r in load(vdir / "results.jsonl") if r["status"] == "ok"]
    errors = load(vdir / "errors.jsonl")
    truncated = len(load(vdir / "results.jsonl")) - len(rows)
    if not rows:
        raise SystemExit(f"no graded rows in {vdir}")

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_kind[r["tags"][0]].append(r)
    in_scope = by_kind["in_scope"]
    costs = [row_cost(r) for r in rows]
    answer_cost = sum(c[0] for c in costs)
    judge_cost = sum(c[1] for c in costs)
    latency = [r["latency_s"] for r in rows]
    model = sorted({r["model"] for r in rows})
    judge_model = sorted({r["judge_model"] for r in rows if r.get("judge_model")})

    lines = [
        f"# Eval results: `{variant}`",
        "",
        f"**{rate([r['grade']['pass'] for r in rows])} cases pass** "
        f"on {', '.join(model)}, judged by {', '.join(judge_model)}. "
        f"{len(errors)} errors, {truncated} truncated.",
        "",
        "With this many cases a pass rate is only accurate to roughly "
        "±18 points, so compare variants case by case, and treat "
        "small differences as noise.",
        "",
        "## By case type",
        "",
        "| Case type | Cases | Pass |",
        "|---|---|---|",
    ]
    for kind, label in KIND_LABELS.items():
        rs = by_kind.get(kind, [])
        lines.append(
            f"| {label} | {len(rs)} | {rate([r['grade']['pass'] for r in rs])} |"
        )

    if in_scope:
        mrr = statistics.mean(r["grade"]["mrr"] for r in in_scope)
        lines += [
            "",
            "## Should-answer questions, step by step",
            "",
            "| Check | Result |",
            "|---|---|",
            check_row(in_scope, "Expected page in top 5 (Hit@5)", "hit_at_5"),
            f"| Rank of first expected page (MRR, 0-1) | {mrr:.2f} |",
            check_row(in_scope, "Cites an expected page", "cites_expected"),
            check_row(in_scope, "Every claim backed by the docs (judge)", "supported"),
            check_row(in_scope, "Answers the question (judge)", "answers"),
        ]
        by_difficulty: dict[str, list[float]] = defaultdict(list)
        for r in in_scope:
            by_difficulty[r["tags"][1]].append(r["grade"]["pass"])
        lines += ["", "| Difficulty | Pass |", "|---|---|"]
        for level in ("easy", "medium", "hard"):
            lines.append(f"| {level} | {rate(by_difficulty.get(level, []))} |")

        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in in_scope:
            by_split[split_of(r)].append(r)
        if len(by_split) > 1:
            lines += [
                "",
                "Held-out questions were written before any fix was designed.",
                "",
                "| Split | Cases | Hit@5 | Pass |",
                "|---|---|---|---|",
            ]
            for split in ("dev", "heldout"):
                rs = by_split.get(split, [])
                lines.append(
                    f"| {split} | {len(rs)} | "
                    f"{rate([r['grade']['hit_at_5'] for r in rs])} | "
                    f"{rate([r['grade']['pass'] for r in rs])} |"
                )

    injection = by_kind.get("injection", [])
    if injection:
        lines += [
            "",
            "## Prompt injection",
            "",
            "| Check | Result |",
            "|---|---|",
            check_row(injection, "Model ignored the planted instruction", "resisted"),
            check_row(
                injection, "Attack absent from what the user saw", "output_clean"
            ),
        ]

    failures = [r for r in rows if not r["grade"]["pass"]]
    lines += ["", f"## Failures ({len(failures)})", ""]
    lines += [
        f"- `{r['prompt_id']}` ({r['tags'][1]}): {failure_reason(r)}" for r in failures
    ]
    if not failures:
        lines.append("None.")

    lines += [
        "",
        "## Cost and speed",
        "",
        f"- Total: ${answer_cost + judge_cost:.2f} "
        f"(answers ${answer_cost:.2f}, judge ${judge_cost:.2f})",
        f"- Per case: ${(answer_cost + judge_cost) / len(rows):.3f} on average",
        f"- Answer latency: median {statistics.median(latency):.1f} s, "
        f"max {max(latency):.1f} s",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    text = summarize(variant)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
