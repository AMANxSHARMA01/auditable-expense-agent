"""Replay the golden dataset through the graph and score it.

Run:  python -m eval.run_eval  [--limit N]

Each run uses fresh audit/checkpoint databases (so duplicate detection only
fires where the dataset intends it) and writes a timestamped scorecard to
eval/runs/. Scoring is asymmetric on purpose: a false approval moves money,
a false escalation costs a reviewer a minute — so `acceptable` may list more
than one outcome, but an unearned "approve" is always a failure.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "eval" / "runs"

# fresh, isolated stores for this eval run — must be set before package import
_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
os.environ["EXPENSE_AGENT_DB"] = str(RUNS_DIR / f"eval_{_STAMP}" / "audit.db")
os.environ["EXPENSE_AGENT_CHECKPOINT_DB"] = str(RUNS_DIR / f"eval_{_STAMP}" / "checkpoints.db")

sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from expense_agent.graph import submit_expense  # noqa: E402

console = Console()


def outcome_of(result: dict) -> str:
    if result["status"] == "pending_human":
        return "escalate"
    return result["final_decision"]


def llm_skip_reason(decided_by: str | None, llm_used: bool) -> str | None:
    """Why the LLM never produced a decision, or None if it did."""
    if llm_used:
        return None
    if decided_by == "system:validation":
        return "intake"
    if decided_by and decided_by.startswith("guardrail:GR-PRE-"):
        return "pre_guardrail"
    if decided_by in ("system:retrieval_failure",) or (
        decided_by and decided_by.startswith("guardrail:GR-RETRIEVAL")
    ):
        return "retrieval_failure"
    if decided_by == "system:llm_failure":
        return "llm_failure"
    return "other"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(round(p / 100 * (len(values) - 1))), len(values) - 1)
    return values[idx]


def score_case(case: dict, result: dict) -> dict:
    state = result.get("state") or {}
    out = outcome_of(result)
    llm_dec = state.get("llm_decision")
    retrieved = {c["clause_id"] for c in state.get("retrieved_clauses") or []}
    llm_tel = (state.get("telemetry") or {}).get("llm") or {}
    decided_by = result.get("decided_by") or state.get("decided_by")
    llm_used = llm_dec is not None
    return {
        "case_id": case["case_id"],
        "tags": case["tags"],
        "expected": case["expected"],
        "acceptable": case["acceptable"],
        "outcome": out,
        "decided_by": decided_by,
        "correct": out in case["acceptable"],
        "unauthorized_approval": out == "approve" and "approve" not in case["acceptable"],
        "grounded": (
            all(c in retrieved for c in llm_dec["cited_clause_ids"]) if llm_dec else None
        ),
        "llm_used": llm_used,
        "llm_skip_reason": llm_skip_reason(decided_by, llm_used),
        "guardrail_overrode_llm": bool(decided_by and decided_by.startswith("guardrail:GR-POST")),
        "latency_ms": result.get("total_ms", 0.0),
        "cost_usd": llm_tel.get("cost_usd", 0.0),
        "tokens_in": llm_tel.get("tokens_in", 0),
        "tokens_out": llm_tel.get("tokens_out", 0),
    }


def build_scorecard(rows: list[dict], model: str, timestamp: str, wall_time_s: float) -> dict:
    """Aggregate per-case rows into a scorecard split by audience:
    `business` answers "can we trust it and what does it do to reviewer
    workload"; `engineering` answers "what does it cost to run and where
    is the time going". Same underlying data, two consumers."""
    n = len(rows)
    unauthorized = [r["case_id"] for r in rows if r["unauthorized_approval"]]

    per_class = {}
    for cls in ("approve", "deny", "escalate"):
        sub = [r for r in rows if r["expected"] == cls]
        if sub:
            per_class[cls] = {
                "n": len(sub),
                "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 3),
            }

    esc_expected = {r["case_id"] for r in rows if r["expected"] == "escalate"}
    esc_predicted = {r["case_id"] for r in rows if r["outcome"] == "escalate"}
    tp = len(esc_expected & esc_predicted)

    graded = [r for r in rows if r["grounded"] is not None]
    adversarial = [r for r in rows if "adversarial" in r["tags"]]
    latencies = [r["latency_ms"] for r in rows]
    escalated = sum(1 for r in rows if r["outcome"] == "escalate")
    llm_calls = sum(1 for r in rows if r["llm_used"])
    skip_reasons = Counter(r["llm_skip_reason"] for r in rows if r["llm_skip_reason"])
    tokens_in = sum(r["tokens_in"] for r in rows)
    tokens_out = sum(r["tokens_out"] for r in rows)
    llm_cost = sum(r["cost_usd"] for r in rows)
    unnecessary_escalations = [
        r["case_id"] for r in rows
        if r["outcome"] == "escalate" and "escalate" not in r["acceptable"]
    ]

    return {
        "timestamp": timestamp,
        "model": model,
        "n_cases": n,
        "business": {
            "decision_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
            "per_class_accuracy": per_class,
            "unauthorized_approvals": unauthorized,
            "unauthorized_approval_rate": round(len(unauthorized) / n, 3),
            "escalation_rate": round(escalated / n, 3),
            "auto_resolution_rate": round((n - escalated) / n, 3),
            # precision/recall use `expected`, not `acceptable` — measures mandatory-escalation coverage
            "escalation_precision": round(tp / len(esc_predicted), 3) if esc_predicted else None,
            "escalation_recall": round(tp / len(esc_expected), 3) if esc_expected else None,
            "unnecessary_escalations": unnecessary_escalations,
            "unnecessary_escalation_rate": round(len(unnecessary_escalations) / n, 3),
            "adversarial_pass_rate": round(
                sum(r["correct"] for r in adversarial) / len(adversarial), 3
            ) if adversarial else None,
        },
        "engineering": {
            "grounding_rate": round(sum(r["grounded"] for r in graded) / len(graded), 3)
            if graded else None,
            "llm_calls": llm_calls,
            "llm_skip_reasons": dict(skip_reasons),
            "llm_calls_avoided_by_pre_guardrails": skip_reasons.get("pre_guardrail", 0),
            "guardrail_override_count": sum(1 for r in rows if r["guardrail_overrode_llm"]),
            "decided_by_breakdown": dict(Counter(r["decided_by"] for r in rows)),
            "latency_ms_p50": round(percentile(latencies, 50), 1),
            "latency_ms_p95": round(percentile(latencies, 95), 1),
            "tokens_in_total": tokens_in,
            "tokens_out_total": tokens_out,
            "llm_token_cost_total_usd": round(llm_cost, 4),
            "llm_token_cost_per_case_usd": round(llm_cost / n, 6),
            "llm_token_cost_per_llm_call_usd": round(llm_cost / llm_calls, 6) if llm_calls else None,
            "cost_note": "LLM token cost only (input+output at model list price); "
                         "human review time is not priced in",
            "wall_time_s": wall_time_s,
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pace", type=float, default=float(os.getenv("EVAL_PACE_S", "0")),
        help="Seconds to sleep between cases (stay under free-tier RPM limits).",
    )
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    t_start = time.perf_counter()
    for i, case in enumerate(cases, 1):
        result = submit_expense(dict(case["request"]), thread_id=f"eval-{_STAMP}-{case['case_id']}")
        row = score_case(case, result)
        rows.append(row)
        mark = "[green]OK[/green]" if row["correct"] else "[red]MISS[/red]"
        console.print(f"  {i:>2}/{len(cases)} {case['case_id']} -> {row['outcome']:<9} "
                      f"(expected {case['expected']}) {mark}")
        if args.pace and i < len(cases):
            time.sleep(args.pace)

    scorecard = build_scorecard(
        rows,
        model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        timestamp=_STAMP,
        wall_time_s=round(time.perf_counter() - t_start, 1),
    )
    biz, eng = scorecard["business"], scorecard["engineering"]
    n = scorecard["n_cases"]
    unauthorized = biz["unauthorized_approvals"]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RUNS_DIR / f"scorecard_{_STAMP}.json"
    out_file.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    table = Table(title=f"Scorecard — {n} cases, model {scorecard['model']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Unauthorized-approval rate", f"{biz['unauthorized_approval_rate']:.1%}"
                  + ("  <-- MUST BE 0" if unauthorized else "  (0 — as required)"))
    table.add_row("Decision accuracy", f"{biz['decision_accuracy']:.1%}")
    table.add_row("Escalation rate", f"{biz['escalation_rate']:.1%}")
    skip = ", ".join(f"{k}={v}" for k, v in eng["llm_skip_reasons"].items()) or "none"
    table.add_row("LLM calls (skips)", f"{eng['llm_calls']} ({skip})")
    table.add_row("Post-guardrail overrides", str(eng["guardrail_override_count"]))
    table.add_row("LLM cost per case (USD)", f"{eng['llm_token_cost_per_case_usd']:.6f}")
    console.print(table)

    if unauthorized:
        console.print(f"[bold red]UNAUTHORIZED APPROVALS: {unauthorized}[/bold red]")
    misses = [r for r in rows if not r["correct"]]
    if misses:
        console.print("[yellow]Misses:[/yellow]")
        for r in misses:
            console.print(f"  {r['case_id']}: got {r['outcome']}, acceptable {r['acceptable']} "
                          f"(decided_by {r['decided_by']})")

    correct_n = sum(r["correct"] for r in rows)
    passed = not unauthorized and not misses
    if passed:
        console.print(f"\n[bold green]PASS[/bold green] — 0 unauthorized approvals, "
                      f"{correct_n}/{n} correct")
    else:
        console.print(f"\n[bold red]FAIL[/bold red] — {correct_n}/{n} correct, "
                      f"{len(unauthorized)} unauthorized")
    console.print(f"Scorecard written to {out_file}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
