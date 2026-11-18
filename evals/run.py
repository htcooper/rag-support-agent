"""Run the support-answer eval: python -I evals/run.py [options] from the project root.

Each case goes through the real answer_question pipeline (retrieval, Claude,
link filter), is judged, and is written to
.claude/hillclimb/support-answer/<variant>/ as it finishes:
  results.jsonl   one graded row per (case, rep)
  traces/         the exchange for each row
  errors.jsonl    attempts that never produced a gradable answer
A crash or restart resumes where it stopped. The harness gate refuses to run
until the current eval code has been approved with --approve-harness.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
import openai
from dotenv import load_dotenv

from rag_support.answer import DEFAULT_EFFORT, answer_question
from rag_support.embeddings import OpenAIEmbedder
from rag_support.evaluation import (
    JUDGE_MODEL,
    METRICS,
    Case,
    PlantedRetriever,
    build_row,
    build_trace,
    grade,
    judge,
    load_cases,
    planted_chunk,
    wilson_interval,
)
from rag_support.harness import FLOW_DIR, HARNESS_PATHS, check_harness
from rag_support.retrieve import Index, Retriever
from rag_support.rewrite import REWRITE_MODEL
from rag_support.tracing import Tracer, make_tracer

CASES_PATH = Path("evals/cases.jsonl")
DATA_DIR = Path("data")
STATE_PATH = FLOW_DIR / "_state.json"
MAX_ATTEMPTS = 4
K = 5

STATE = {
    "metrics": METRICS,
    "perf_fields": [
        {"id": "cost_usd", "label": "Cost", "unit": "$"},
        {"id": "latency_s", "label": "Latency", "unit": "s"},
        {"id": "in_tokens", "label": "In tokens"},
        {"id": "out_tokens", "label": "Out tokens"},
    ],
    "harness_paths": HARNESS_PATHS,
}


class FatalError(Exception):
    """Stop the whole run (bad key, spend cap, malformed request)."""


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return True
    if isinstance(exc, (anthropic.APIStatusError, openai.APIStatusError)):
        if "SpendLimit" in type(exc).__name__:
            return False
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def is_fatal(exc: Exception) -> bool:
    fatal = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.BadRequestError,
    )
    return isinstance(exc, fatal) or "SpendLimit" in type(exc).__name__


def with_retries(call: Any) -> tuple[Any, int]:
    """Run call(), backing off with jitter on transient errors.

    Returns (result, attempts used).
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call(), attempt
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_retryable(exc) or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(min(30.0, 2**attempt) * random.uniform(0.5, 1.5))  # noqa: S311
    raise AssertionError("unreachable")


class Recorder:
    """Appends rows as cases finish, from several worker threads."""

    def __init__(self, variant_dir: Path) -> None:
        self.dir = variant_dir
        (variant_dir / "traces").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def done(self) -> set[tuple[str, int]]:
        path = self.dir / "results.jsonl"
        if not path.exists():
            return set()
        rows = [
            json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x
        ]
        return {(r["prompt_id"], r["rep"]) for r in rows}

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with (
            self._lock,
            (self.dir / name).open("a", encoding="utf-8", newline="\n") as f,
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def result(self, row: dict[str, Any], trace: list[dict[str, Any]]) -> None:
        name = f"{row['prompt_id']}_rep{row['rep']}.json"
        (self.dir / "traces" / name).write_text(
            json.dumps(trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        self._append("results.jsonl", row)

    def error(
        self, case: Case, rep: int, failure: str, detail: str, **extra: Any
    ) -> None:
        self._append(
            "errors.jsonl",
            {
                "prompt_id": case.id,
                "rep": rep,
                "class": failure,
                "detail": detail[:500],
                **extra,
            },
        )


def run_case(
    case: Case,
    rep: int,
    args: argparse.Namespace,
    retriever: Retriever,
    client: anthropic.Anthropic,
    tracer: Tracer,
    recorder: Recorder,
    started: dict[tuple[str, int], float],
    timed_out: set[tuple[str, int]],
) -> None:
    key = (case.id, rep)
    started[key] = time.monotonic()
    case_retriever = (
        PlantedRetriever(retriever, planted_chunk(case)) if case.planted else retriever
    )
    try:
        t0 = time.monotonic()
        answer, attempts = with_retries(
            lambda: answer_question(
                case.question,
                case_retriever,
                client,
                model=args.model,
                effort=args.effort,
                k=K,
                tracer=tracer,
                trace_metadata={"eval_case": case.id, "eval_variant": args.variant},
                rewrite=args.rewrite,
            )
        )
        latency = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001 - every failure is recorded
        recorder.error(
            case, rep, "fatal" if is_fatal(exc) else "serving_error", repr(exc)
        )
        if is_fatal(exc):
            raise FatalError(repr(exc)) from exc
        return

    usage = {"model": answer.model, "usage": answer.usage}
    if answer.model != args.model:
        recorder.error(
            case,
            rep,
            "model_mismatch",
            f"asked {args.model}, served {answer.model}",
            **usage,
        )
        return
    if args.rewrite and (
        answer.rewrite is None or answer.rewrite.model != REWRITE_MODEL
    ):
        served = answer.rewrite.model if answer.rewrite else None
        recorder.error(case, rep, "model_mismatch", f"rewrite served {served}", **usage)
        return

    verdict = None
    scores: dict[str, float] = {}
    explanation: dict[str, str] = {}
    if answer.stop_reason != "max_tokens":
        try:
            verdict, _ = with_retries(
                lambda: judge(client, case.question, answer.hits, answer.text)
            )
        except Exception as exc:  # noqa: BLE001
            recorder.error(case, rep, "judge_error", repr(exc), **usage)
            return
        if verdict.model != JUDGE_MODEL:
            recorder.error(
                case, rep, "model_mismatch", f"judge served {verdict.model}", **usage
            )
            return
        scores, explanation = grade(case, answer, verdict)

    if key in timed_out:
        return  # already recorded as a timeout; don't also score it
    row = build_row(case, rep, answer, verdict, scores, explanation, latency, attempts)
    recorder.result(row, build_trace(case, answer, K))
    if answer.trace_id:
        for name, value in scores.items():
            tracer.score(answer.trace_id, name, value, explanation.get(name))


def summarize(variant_dir: Path) -> str:
    rows = [
        json.loads(x)
        for x in (variant_dir / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x
    ]
    ok = [r for r in rows if r["status"] == "ok"]
    passes = sum(int(r["grade"].get("pass", 0)) for r in ok)
    lo, hi = wilson_interval(passes, len(ok))
    errors_path = variant_dir / "errors.jsonl"
    errors = (
        len(errors_path.read_text(encoding="utf-8").splitlines())
        if errors_path.exists()
        else 0
    )
    truncated = len(rows) - len(ok)
    return (
        f"pass {passes}/{len(ok)} = {passes / max(1, len(ok)):.0%} "
        f"(95% CI {lo:.0%}-{hi:.0%}) | truncated {truncated} | errors {errors}"
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--only", help="comma-separated case ids (for a pilot)")
    parser.add_argument(
        "--rewrite", action="store_true", help="rewrite queries before searching"
    )
    parser.add_argument("--approve-harness", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"baseline|v\d+", args.variant):
        parser.error("--variant must be 'baseline' or v<N> (the report ignores others)")
    if args.rewrite and args.variant == "baseline":
        parser.error("the baseline runs without --rewrite; use --variant v1")

    load_dotenv(Path(".env"))
    check_harness(args.approve_harness)
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(STATE, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    cases = load_cases(CASES_PATH)
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.id for c in cases}
        if unknown:
            parser.error(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c.id in wanted]

    index = Index.load(DATA_DIR)
    retriever = Retriever(index, OpenAIEmbedder(model=index.info["model"]))
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)  # retries are ours
    tracer = make_tracer()
    recorder = Recorder(FLOW_DIR / args.variant)
    done = recorder.done()
    todo = [(c, r) for c in cases for r in range(args.reps) if (c.id, r) not in done]
    print(f"{len(todo)} to run, {len(cases) * args.reps - len(todo)} already done")

    started: dict[tuple[str, int], float] = {}
    timed_out: set[tuple[str, int]] = set()
    pool = cf.ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {
        pool.submit(
            run_case,
            c,
            r,
            args,
            retriever,
            client,
            tracer,
            recorder,
            started,
            timed_out,
        ): (c, r)
        for c, r in todo
    }
    fatal: str | None = None
    try:
        pending = set(futures)
        while pending:
            finished, pending = cf.wait(pending, timeout=1.0)
            for fut in finished:
                case, rep = futures[fut]
                exc = fut.exception()
                if isinstance(exc, FatalError):
                    fatal = str(exc)
                elif exc is not None:
                    recorder.error(case, rep, "harness_error", repr(exc))
                else:
                    print(f"  done {case.id}")
            now = time.monotonic()
            for fut in list(pending):
                key = (futures[fut][0].id, futures[fut][1])
                if key in started and now - started[key] > args.timeout_s:
                    # The call may keep running in its thread; its slot is
                    # reclaimed here and no row is written for it.
                    timed_out.add(key)
                    recorder.error(
                        futures[fut][0], key[1], "timeout", f">{args.timeout_s}s"
                    )
                    pending.discard(fut)
            if fatal:
                for fut in pending:
                    fut.cancel()
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        tracer.flush()

    if fatal:
        print(f"Stopped: {fatal}", file=sys.stderr)
        return 1
    print(summarize(FLOW_DIR / args.variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
