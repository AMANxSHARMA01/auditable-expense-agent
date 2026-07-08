"""Eval scoring — pure functions, no API key."""

from eval.scoring import build_scorecard, llm_skip_reason, score_case


def test_llm_skip_reason_validate():
    assert llm_skip_reason("system:validation", False) == "validate"


def test_score_case_flags_unauthorized_approval():
    case = {"case_id": "D01", "expected": "deny", "acceptable": ["deny"], "tags": []}
    result = {
        "status": "completed",
        "final_decision": "approve",
        "decided_by": "llm",
        "total_ms": 100,
    }
    row = score_case(case, result)
    assert row["correct"] is False
    assert row["unauthorized_approval"] is True


def test_build_scorecard_aggregates_rows():
    rows = [
        {"case_id": "A01", "expected": "approve", "acceptable": ["approve"], "tags": [],
         "outcome": "approve", "correct": True, "unauthorized_approval": False,
         "grounded": True, "llm_used": True, "llm_skip_reason": None,
         "guardrail_overrode_llm": False, "decided_by": "llm",
         "latency_ms": 100, "cost_usd": 0.001, "tokens_in": 10, "tokens_out": 5},
        {"case_id": "D01", "expected": "deny", "acceptable": ["deny"], "tags": ["guardrail"],
         "outcome": "deny", "correct": True, "unauthorized_approval": False,
         "grounded": None, "llm_used": False, "llm_skip_reason": "pre_guardrail",
         "guardrail_overrode_llm": False, "decided_by": "guardrail:GR-PRE-BLACKLIST",
         "latency_ms": 50, "cost_usd": 0, "tokens_in": 0, "tokens_out": 0},
    ]
    sc = build_scorecard(rows, "test-model", "ts", 1.0)
    assert sc["n_cases"] == 2
    assert sc["business"]["decision_accuracy"] == 1.0
    assert sc["engineering"]["llm_calls"] == 1
    assert sc["engineering"]["llm_calls_avoided_by_pre_guardrails"] == 1
