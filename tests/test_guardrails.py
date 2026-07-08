"""Guardrails are pure functions — the safety layer is testable with no
API key and no network."""

from expense_agent import guardrails
from expense_agent.schemas import ExpenseRequest, LLMDecision


def req(**overrides) -> ExpenseRequest:
    base = dict(
        request_id="t-1", employee_id="E-1", amount=100.0, vendor="Good Vendor",
        category="meals", description="team lunch", receipt_attached=True,
    )
    base.update(overrides)
    return ExpenseRequest.model_validate(base)


def dec(**overrides) -> LLMDecision:
    base = dict(decision="approve", justification="ok",
                cited_clause_ids=["EXP-004"], confidence=0.9)
    base.update(overrides)
    return LLMDecision.model_validate(base)


# --- pre_check ---

def test_blacklisted_vendor_denied_case_insensitive():
    events = guardrails.pre_check(req(vendor="ShAdOw Consulting LTD"), duplicate_exists=False)
    assert [e.action for e in events] == ["deny"]
    assert events[0].rule_id == "GR-PRE-BLACKLIST"


def test_blacklist_beats_hard_ceiling():
    events = guardrails.pre_check(
        req(vendor="Luxe Gifts Co", amount=9500.0), duplicate_exists=False)
    assert events[0].rule_id == "GR-PRE-BLACKLIST"
    assert events[0].action == "deny"


def test_missing_receipt_above_threshold_denied():
    events = guardrails.pre_check(req(receipt_attached=False, amount=85.0), duplicate_exists=False)
    assert events[0].rule_id == "GR-PRE-RECEIPT"


def test_missing_receipt_small_amount_passes():
    assert guardrails.pre_check(req(receipt_attached=False, amount=20.0), duplicate_exists=False) == []


def test_hard_ceiling_escalates_without_llm():
    events = guardrails.pre_check(req(amount=12000.0), duplicate_exists=False)
    assert events[0].rule_id == "GR-PRE-CEILING"
    assert events[0].action == "escalate"


def test_duplicate_escalates():
    events = guardrails.pre_check(req(), duplicate_exists=True)
    assert events[0].rule_id == "GR-PRE-DUPLICATE"


def test_clean_request_passes():
    assert guardrails.pre_check(req(), duplicate_exists=False) == []


def test_apply_pre_events_pass_is_empty():
    assert guardrails.apply_pre_events([]) == {}


def test_apply_pre_events_maps_deny_and_escalate():
    deny = guardrails.pre_check(req(vendor="QuickCash Services"), duplicate_exists=False)
    assert guardrails.apply_pre_events(deny) == {
        "final_decision": "deny",
        "decided_by": "guardrail:GR-PRE-BLACKLIST",
    }
    esc = guardrails.pre_check(req(amount=12000.0), duplicate_exists=False)
    out = guardrails.apply_pre_events(esc)
    assert out["final_decision"] == "escalate"
    assert out["decided_by"] == "guardrail:GR-PRE-CEILING"
    assert "CFO review" in out["pending_reason"]


# --- post_check ---

def test_llm_approve_over_limit_is_overridden():
    events = guardrails.post_check(req(amount=4200.0), dec(), ["EXP-004"])
    assert any(e.rule_id == "GR-POST-LIMIT" for e in events)
    final, decided_by = guardrails.resolve_final(dec(), events)
    assert final == "escalate"
    assert decided_by.startswith("guardrail:")


def test_hallucinated_citation_is_caught():
    events = guardrails.post_check(req(), dec(cited_clause_ids=["EXP-042"]), ["EXP-004", "EXP-001"])
    assert any(e.rule_id == "GR-POST-GROUNDING" for e in events)


def test_low_confidence_approval_escalates():
    events = guardrails.post_check(req(), dec(confidence=0.3), ["EXP-004"])
    assert any(e.rule_id == "GR-POST-CONFIDENCE" for e in events)


def test_compliant_approval_stands():
    events = guardrails.post_check(req(amount=100.0), dec(), ["EXP-004"])
    assert events == []
    final, decided_by = guardrails.resolve_final(dec(), events)
    assert (final, decided_by) == ("approve", "llm")


def test_llm_failure_fails_closed():
    final, decided_by = guardrails.resolve_final(None, [])
    assert (final, decided_by) == ("escalate", "system:llm_failure")


def test_apply_post_events_llm_failure():
    out = guardrails.apply_post_events(None, [])
    assert out["final_decision"] == "escalate"
    assert out["decided_by"] == "system:llm_failure"
    assert "failing closed" in out["pending_reason"]


def test_apply_post_events_over_limit():
    events = guardrails.post_check(req(amount=4200.0), dec(), ["EXP-004"])
    out = guardrails.apply_post_events(dec(), events)
    assert out["final_decision"] == "escalate"
    assert out["decided_by"] == "guardrail:GR-POST-LIMIT"
    assert "pending_reason" in out
