"""Write evals/comparison.md: baseline vs v1, case by case.

Run with python -I evals/compare.py [before] [after] from the project root
(defaults: baseline v1). Every number is recomputed from the two variants'
results.jsonl files. Each case's reps are averaged first, and intervals are
computed over the cases, because reps of one case aren't independent.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from rag_support.evaluation import mean_ci, per_case_means
from rag_support.harness import FLOW_DIR
from rag_support.tracing import cost_details

OUT = Path("evals/comparison.md")


def load(variant: str) -> list[dict[str, Any]]:
    path = FLOW_DIR / variant / "results.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def group_of(row: dict[str, Any]) -> str:
    kind = str(row["tags"][0])
    if kind != "in_scope":
        return kind
    split = row["tags"][2] if len(row["tags"]) > 2 else "dev"
    return f"in_scope_{split}"


GROUPS = {
    "in_scope_dev": "Should answer, dev",
    "in_scope_heldout": "Should answer, held-out",
    "out_of_scope": "Should decline",
    "injection": "Prompt injection",
}


def pct(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.0%} ({max(lo, 0):.0%}-{min(hi, 1):.0%})"


def signed(mean: float, lo: float, hi: float) -> str:
    return f"{mean * 100:+.0f} pts ({lo * 100:+.0f} to {hi * 100:+.0f})"


def row_costs(row: dict[str, Any]) -> float:
    total = cost_details(
        row["model"], row["usage"]["input_tokens"], row["usage"]["output_tokens"]
    ).get("total", 0.0)
    if row.get("rewrite_usage"):
        u = row["rewrite_usage"]
        total += cost_details(
            row["rewrite_model"], u["input_tokens"], u["output_tokens"]
        ).get("total", 0.0)
    return total


def rep_agreement(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Cases whose reps all got the same pass/fail, out of cases with 2+ reps."""
    by_case: dict[str, set[float]] = {}
    counts: dict[str, int] = {}
    for r in rows:
        if r["status"] == "ok":
            by_case.setdefault(r["prompt_id"], set()).add(r["grade"]["pass"])
            counts[r["prompt_id"]] = counts.get(r["prompt_id"], 0) + 1
    multi = [c for c, n in counts.items() if n > 1]
    return sum(len(by_case[c]) == 1 for c in multi), len(multi)


def compare(before: str, after: str) -> str:
    rows_b, rows_a = load(before), load(after)
    groups = {r["prompt_id"]: group_of(r) for r in rows_b}
    pass_b, pass_a = per_case_means(rows_b, "pass"), per_case_means(rows_a, "pass")
    hit_b, hit_a = per_case_means(rows_b, "hit_at_5"), per_case_means(
        rows_a, "hit_at_5"
    )
    cases = sorted(set(pass_b) & set(pass_a))
    reps = max(r["rep"] for r in rows_b + rows_a) + 1

    lines = [
        f"# Before and after: `{before}` vs `{after}`",
        "",
        f"`{after}` adds query rewriting: Claude Haiku 5.5 restates each question "
        "in FastAPI's vocabulary before the search. Everything else is the same. "
        f"{len(cases)} cases, {reps} runs each per variant. Pass rates average each "
        "case's runs; ranges are 95% confidence intervals over cases.",
        "",
        "| Cases | n | Pass before | Pass after | Change (paired) |",
        "|---|---|---|---|---|",
    ]
    sets = [("All cases", cases)] + [
        (label, [c for c in cases if groups[c] == g]) for g, label in GROUPS.items()
    ]
    for label, cs in sets:
        if not cs:
            continue
        b = mean_ci([pass_b[c] for c in cs])
        a = mean_ci([pass_a[c] for c in cs])
        d = mean_ci([pass_a[c] - pass_b[c] for c in cs])
        lines.append(f"| {label} | {len(cs)} | {pct(*b)} | {pct(*a)} | {signed(*d)} |")

    answerable = [c for c in cases if c in hit_b and c in hit_a]
    hb = mean_ci([hit_b[c] for c in answerable])
    ha = mean_ci([hit_a[c] for c in answerable])
    hd = mean_ci([hit_a[c] - hit_b[c] for c in answerable])
    lines += [
        "",
        "## Retrieval (should-answer questions)",
        "",
        "| Expected page in top 5 | Before | After | Change (paired) |",
        "|---|---|---|---|",
        f"| {len(answerable)} questions | {pct(*hb)} | {pct(*ha)} | {signed(*hd)} |",
    ]

    agree_b, multi_b = rep_agreement(rows_b)
    agree_a, multi_a = rep_agreement(rows_a)
    if multi_b or multi_a:
        lines += [
            "",
            "## Run-to-run noise",
            "",
            f"Same pass/fail on every run: {agree_b}/{multi_b} cases in `{before}`, "
            f"{agree_a}/{multi_a} in `{after}`. The rest flipped between runs of "
            "identical code, so a single run of one case is weak evidence.",
        ]

    better = [c for c in cases if pass_a[c] > pass_b[c]]
    worse = [c for c in cases if pass_a[c] < pass_b[c]]
    lines += ["", "## Cases that changed", ""]
    lines += [f"- better: `{c}` ({pass_b[c]:.0%} -> {pass_a[c]:.0%})" for c in better]
    lines += [f"- worse: `{c}` ({pass_b[c]:.0%} -> {pass_a[c]:.0%})" for c in worse]
    if not better and not worse:
        lines.append("None.")

    cost_b = statistics.mean(row_costs(r) for r in rows_b)
    cost_a = statistics.mean(row_costs(r) for r in rows_a)
    lat_b = statistics.median(r["latency_s"] for r in rows_b)
    lat_a = statistics.median(r["latency_s"] for r in rows_a)
    lines += [
        "",
        "## Cost and speed per answer",
        "",
        f"- Cost, excluding the judge: ${cost_b:.4f} -> ${cost_a:.4f}",
        f"- Median latency: {lat_b:.1f} s -> {lat_a:.1f} s",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    before = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    after = sys.argv[2] if len(sys.argv) > 2 else "v1"
    text = compare(before, after)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
