"""Export one eval trace from Langfuse to docs/sample-trace.json.

Run with python -I evals/export_trace.py <case-id> [variant] [rep] from the
project root. Only the pipeline's own fields are kept; Langfuse's internal
attributes (SDK scope data, including the project's public key, and resource
attributes) are dropped, and the result is scanned for secrets and local paths
before it is written.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

FLOW_DIR = Path(".claude/hillclimb/support-answer")
OUT = Path("docs/sample-trace.json")
FIELDS = "core,basic,io,metadata,model,usage,metrics"
# Langfuse's own (camelCase) field names; anything else, such as projectId,
# is left out.
KEEP = (
    "id",
    "name",
    "type",
    "parentObservationId",
    "startTime",
    "endTime",
    "latency",
    "model",
    "input",
    "output",
    "usageDetails",
    "costDetails",
)
LEAK_PATTERNS = {
    "key-like string": r"\b(sk|pk)-[A-Za-z0-9-]{8,}",
    "windows path": r"[A-Za-z]:\\\\Users",
    "unix home path": r"/(home|Users)/[A-Za-z]",
    "account id": r"wrkspc_|organization\.id|workspace\.id",
}


def trace_id_for(case: str, variant: str, rep: int) -> str:
    path = FLOW_DIR / variant / "results.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["prompt_id"] == case and row["rep"] == rep:
            return str(row["meta"]["trace_url"]).rstrip("/").rsplit("/", 1)[-1]
    raise SystemExit(f"no row for {case} rep {rep} in {path}")


def clean(observation: dict[str, Any]) -> dict[str, Any]:
    kept = {k: observation.get(k) for k in KEEP}
    metadata = observation.get("metadata") or {}
    kept["metadata"] = {
        k: v
        for k, v in metadata.items()
        if not k.startswith(("scope.", "resourceAttributes.", "attributes."))
    }
    return kept


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(Path(".env"))
    from langfuse import Langfuse

    case = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "v1"
    rep = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    trace_id = trace_id_for(case, variant, rep)
    lf = Langfuse(timeout=60)
    observations: list[Any] = []
    for _ in range(6):
        observations = lf.api.observations.get_many(
            trace_id=trace_id, fields=FIELDS, expand_metadata="true", limit=50
        ).data
        if observations:
            break
        time.sleep(5)
    scores = lf.api.scores.get_many(trace_id=trace_id, limit=50).data
    trace: dict[str, Any] = {
        "case": case,
        "variant": variant,
        "rep": rep,
        "observations": sorted(
            (clean(o.dict()) for o in observations), key=lambda o: str(o["startTime"])
        ),
        "scores": sorted(
            ({"name": s.name, "value": s.value, "comment": s.comment} for s in scores),
            key=lambda s: str(s["name"]),
        ),
    }
    text = json.dumps(trace, indent=2, default=str, ensure_ascii=False)
    leaks = {
        n: re.findall(p, text) for n, p in LEAK_PATTERNS.items() if re.search(p, text)
    }
    if leaks:
        raise SystemExit(f"not written; possible leaks: {leaks}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8", newline="\n")
    names = [o["name"] for o in trace["observations"]]
    print(
        f"wrote {OUT}: {len(names)} observations {names}, {len(trace['scores'])} scores"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
