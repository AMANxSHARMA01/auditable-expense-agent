"""Deterministic guardrails. These run around the LLM and always win:
the model proposes, policy code disposes. Pure functions — unit-testable
without any API key."""

from . import config
from .schemas import ExpenseRequest, GuardrailEvent, LLMDecision


def pre_check(request: ExpenseRequest, duplicate_exists: bool) -> list[GuardrailEvent]:
    """Runs BEFORE retrieval/LLM. A deny/escalate here short-circuits the
    graph entirely — no tokens are spent on requests policy already settles."""
    events: list[GuardrailEvent] = []

    if request.vendor.strip().lower() in config.VENDOR_BLACKLIST:
        events.append(GuardrailEvent(
            rule_id="GR-PRE-BLACKLIST", stage="pre", action="deny",
            detail=f"Vendor '{request.vendor}' is on the restricted-vendor list (EXP-013).",
        ))
        return events  # blacklist is terminal; no other check can soften it

    if not request.receipt_attached and request.amount > config.RECEIPT_REQUIRED_ABOVE:
        events.append(GuardrailEvent(
            rule_id="GR-PRE-RECEIPT", stage="pre", action="deny",
            detail=(f"No receipt attached for {request.amount:.2f} {request.currency}; "
                    f"receipts are mandatory above {config.RECEIPT_REQUIRED_ABOVE:.0f} (EXP-001)."),
        ))
        return events

    if request.amount >= config.HARD_CEILING:
        events.append(GuardrailEvent(
            rule_id="GR-PRE-CEILING", stage="pre", action="escalate",
            detail=(f"Amount {request.amount:.2f} >= hard ceiling {config.HARD_CEILING:.0f}; "
                    "requires CFO review, LLM is not consulted (EXP-003)."),
        ))
        return events

    if duplicate_exists:
        events.append(GuardrailEvent(
            rule_id="GR-PRE-DUPLICATE", stage="pre", action="escalate",
            detail=("Same employee, vendor and amount seen within "
                    f"{config.DUPLICATE_WINDOW_HOURS:.0f}h; routed to manual review (EXP-014)."),
        ))

    return events


def apply_pre_events(events: list[GuardrailEvent]) -> dict:
    """Map pre-guardrail events to a terminal graph outcome, or {} to continue."""
    denies = [e for e in events if e.action == "deny"]
    if denies:
        e = denies[0]
        return {"final_decision": "deny", "decided_by": f"guardrail:{e.rule_id}"}
    escalates = [e for e in events if e.action == "escalate"]
    if escalates:
        e = escalates[0]
        return {
            "final_decision": "escalate",
            "decided_by": f"guardrail:{e.rule_id}",
            "pending_reason": e.detail,
        }
    return {}


def post_check(
    request: ExpenseRequest,
    decision: LLMDecision,
    retrieved_clause_ids: list[str],
) -> list[GuardrailEvent]:
    """Runs AFTER the LLM. Overrides the model whenever its output would
    exceed the authority policy grants an automated system."""
    events: list[GuardrailEvent] = []

    if decision.decision == "approve" and request.amount > config.AUTO_APPROVE_LIMIT:
        events.append(GuardrailEvent(
            rule_id="GR-POST-LIMIT", stage="post", action="escalate",
            detail=(f"LLM approved {request.amount:.2f} {request.currency} but auto-approval "
                    f"is capped at {config.AUTO_APPROVE_LIMIT:.0f} (EXP-002). Overridden to escalate."),
        ))

    hallucinated = [c for c in decision.cited_clause_ids if c not in retrieved_clause_ids]
    if hallucinated:
        events.append(GuardrailEvent(
            rule_id="GR-POST-GROUNDING", stage="post", action="escalate",
            detail=(f"Justification cites clause(s) {hallucinated} that were not in the "
                    "retrieved set — ungrounded reasoning is never executed."),
        ))

    if decision.decision == "approve" and decision.confidence < config.CONFIDENCE_FLOOR:
        events.append(GuardrailEvent(
            rule_id="GR-POST-CONFIDENCE", stage="post", action="escalate",
            detail=(f"Approval confidence {decision.confidence:.2f} below floor "
                    f"{config.CONFIDENCE_FLOOR:.2f}; routed to a human."),
        ))

    return events


def resolve_final(
    llm_decision: LLMDecision | None,
    post_events: list[GuardrailEvent],
) -> tuple[str, str]:
    """Combine LLM output and post-guardrail events into (final_decision, decided_by)."""
    if llm_decision is None:
        return "escalate", "system:llm_failure"
    if post_events:
        # any post event forces escalation and names the overriding rule
        return "escalate", f"guardrail:{post_events[0].rule_id}"
    return llm_decision.decision, "llm"


def apply_post_events(
    llm_decision: LLMDecision | None,
    post_events: list[GuardrailEvent],
) -> dict:
    """Map LLM output + post-guardrail events to a terminal graph outcome."""
    if llm_decision is None:
        return {
            "final_decision": "escalate",
            "decided_by": "system:llm_failure",
            "pending_reason": "LLM reasoning step failed schema validation twice; failing closed.",
        }
    final, decided_by = resolve_final(llm_decision, post_events)
    reason_txt = post_events[0].detail if post_events else (
        llm_decision.justification if final == "escalate" else None
    )
    out: dict = {"final_decision": final, "decided_by": decided_by}
    if final == "escalate" and reason_txt:
        out["pending_reason"] = reason_txt
    return out
