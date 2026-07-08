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

from eval.scoring import build_scorecard, score_case  # noqa: E402
from expense_agent.graph import submit_expense  # noqa: E402

console = Console()


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
        result = submit_expense(
            dict(case["request"]),
            thread_id=f"eval-{_STAMP}-{case['case_id']}",
            include_state=True,
        )
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
