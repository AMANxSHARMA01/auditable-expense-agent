"""Pipeline node tests — no API key, LLM/retrieval mocked."""

from unittest.mock import patch

from expense_agent.graph import ai_review, validate
from expense_agent.schemas import LLMDecision


def _request(**overrides) -> dict:
    base = {
        "request_id": "g-1", "employee_id": "E-1", "amount": 100.0,
        "vendor": "Good Vendor", "category": "meals",
        "description": "team lunch", "receipt_attached": True,
    }
    base.update(overrides)
    return base


def test_validate_denies_blacklisted_vendor():
    state = {"request": _request(vendor="QuickCash Services"), "telemetry": {}}
    with patch("expense_agent.graph.audit.duplicate_exists", return_value=False):
        out = validate(state)
    assert out["final_decision"] == "deny"
    assert out["decided_by"] == "guardrail:GR-PRE-BLACKLIST"


def test_validate_escalates_at_hard_ceiling():
    state = {"request": _request(amount=12000.0), "telemetry": {}}
    with patch("expense_agent.graph.audit.duplicate_exists", return_value=False):
        out = validate(state)
    assert out["final_decision"] == "escalate"
    assert out["decided_by"] == "guardrail:GR-PRE-CEILING"


def test_ai_review_approves_compliant_request():
    state = {
        "request": _request(amount=80.0),
        "guardrail_events": [],
        "telemetry": {},
    }
    clauses = [{"clause_id": "EXP-004", "title": "Meals", "text": "...", "score": 0.9}]
    llm_out = LLMDecision(
        decision="approve", justification="Within meal limit.",
        cited_clause_ids=["EXP-004"], confidence=0.9,
    )
    with patch("expense_agent.graph.retrieval.retrieve", return_value=clauses), patch(
        "expense_agent.graph.llm.decide", return_value=(llm_out, {"cost_usd": 0.001}),
    ):
        out = ai_review(state)
    assert out["final_decision"] == "approve"
    assert out["decided_by"] == "llm"


def test_ai_review_overrides_over_limit_approval():
    state = {
        "request": _request(amount=4200.0),
        "guardrail_events": [],
        "telemetry": {},
    }
    clauses = [{"clause_id": "EXP-004", "title": "Meals", "text": "...", "score": 0.9}]
    llm_out = LLMDecision(
        decision="approve", justification="Looks fine.",
        cited_clause_ids=["EXP-004"], confidence=0.9,
    )
    with patch("expense_agent.graph.retrieval.retrieve", return_value=clauses), patch(
        "expense_agent.graph.llm.decide", return_value=(llm_out, {"cost_usd": 0.001}),
    ):
        out = ai_review(state)
    assert out["final_decision"] == "escalate"
    assert out["decided_by"] == "guardrail:GR-POST-LIMIT"
