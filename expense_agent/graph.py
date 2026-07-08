"""The LangGraph state machine.

    validate -> ai_review -> finalize
         |         |
         `-(deny)  `-(escalate) -> hitl -> finalize

LangGraph is used for durable human-in-the-loop pauses (interrupt +
checkpointer) and replayable control flow.
"""

import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from . import audit, config, guardrails, llm, retrieval
from .schemas import ExpenseRequest, LLMDecision


class AgentState(TypedDict, total=False):
    run_id: str
    request: dict
    guardrail_events: list[dict]
    retrieved_clauses: list[dict]
    llm_decision: Optional[dict]
    final_decision: Optional[str]
    decided_by: Optional[str]
    pending_reason: Optional[str]
    human: Optional[dict]
    telemetry: dict


def _node_timing(state: AgentState, node: str, t0: float) -> dict:
    tel = dict(state.get("telemetry") or {})
    nodes = dict(tel.get("node_latency_ms") or {})
    nodes[node] = round((time.perf_counter() - t0) * 1000, 1)
    tel["node_latency_ms"] = nodes
    return tel


def _request(state: AgentState) -> ExpenseRequest:
    return ExpenseRequest.model_validate(state["request"])


def _audit_record(state: AgentState, *, status: str, total_ms: float | None = None) -> dict:
    llm_tel = (state.get("telemetry") or {}).get("llm") or {}
    telemetry = {
        **(state.get("telemetry") or {}),
        "cost_usd": llm_tel.get("cost_usd", 0.0),
        "tokens_in": llm_tel.get("tokens_in", 0),
        "tokens_out": llm_tel.get("tokens_out", 0),
    }
    if total_ms is not None:
        telemetry["total_ms"] = total_ms
    return {
        "run_id": state["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "request": state["request"],
        "retrieved_clauses": state.get("retrieved_clauses") or [],
        "llm_decision": state.get("llm_decision"),
        "guardrail_events": state.get("guardrail_events") or [],
        "human": state.get("human"),
        "final_decision": state.get("final_decision"),
        "decided_by": state.get("decided_by"),
        "telemetry": telemetry,
    }


# --- Nodes ---

def validate(state: AgentState) -> dict:
    t0 = time.perf_counter()
    try:
        req = ExpenseRequest.model_validate(state["request"]).normalized()
    except ValidationError as exc:
        return {
            "final_decision": "deny",
            "decided_by": "system:validation",
            "guardrail_events": [{
                "rule_id": "GR-INTAKE-SCHEMA", "stage": "pre", "action": "deny",
                "detail": f"Request failed schema validation: {exc.error_count()} error(s).",
            }],
            "telemetry": _node_timing(state, "validate", t0),
        }

    dup = audit.duplicate_exists(req.employee_id, req.vendor, req.amount, req.request_id)
    events = guardrails.pre_check(req, duplicate_exists=dup)
    update: dict = {
        "request": req.model_dump(),
        "guardrail_events": [e.model_dump() for e in events],
        "telemetry": _node_timing(state, "validate", t0),
    }
    update.update(guardrails.apply_pre_events(events))
    return update


def ai_review(state: AgentState) -> dict:
    """Retrieve policy clauses, run the LLM, apply post-guardrails."""
    t0 = time.perf_counter()
    req = _request(state)
    prior_events = list(state.get("guardrail_events") or [])

    try:
        clauses = retrieval.retrieve(req)
        if not clauses:
            detail = "Retrieval returned no policy clauses."
            raise ValueError(detail)
    except Exception as exc:
        detail = (
            str(exc)
            if isinstance(exc, ValueError)
            else f"Policy retrieval failed after retries: {type(exc).__name__}."
        )
        return {
            "final_decision": "escalate",
            "decided_by": "system:retrieval_failure",
            "pending_reason": detail + " Routed to manual review (no grounded decision possible).",
            "guardrail_events": prior_events + [{
                "rule_id": "GR-RETRIEVAL-FAILCLOSED", "stage": "pre",
                "action": "escalate", "detail": detail,
            }],
            "telemetry": _node_timing(state, "ai_review", t0),
        }

    decision, llm_tel = llm.decide(req, clauses)
    tel = _node_timing(state, "ai_review", t0)
    tel["llm"] = llm_tel
    raw = decision.model_dump() if decision else None
    retrieved_ids = [c["clause_id"] for c in clauses]

    if raw is None:
        llm_decision = None
        events = []
    else:
        llm_decision = LLMDecision.model_validate(raw)
        events = guardrails.post_check(req, llm_decision, retrieved_ids)

    return {
        "retrieved_clauses": clauses,
        "llm_decision": raw,
        "guardrail_events": prior_events + [e.model_dump() for e in events],
        **guardrails.apply_post_events(llm_decision, events),
        "telemetry": tel,
    }


def hitl(state: AgentState) -> dict:
    answer = interrupt({
        "request": state["request"],
        "reason": state.get("pending_reason"),
        "llm_decision": state.get("llm_decision"),
        "guardrail_events": state.get("guardrail_events") or [],
    })
    return {
        "final_decision": answer["decision"],
        "decided_by": f"human:{answer.get('reviewer', 'unknown')}",
        "human": answer,
    }


def finalize(state: AgentState) -> dict:
    audit.write_run(_audit_record(state, status="completed"))
    return {}


# --- Routing ---

_ROUTE: dict[str, Callable[[AgentState], str]] = {
    "after_validate": lambda s: (
        "finalize" if s.get("final_decision") == "deny"
        else "hitl" if s.get("final_decision") == "escalate"
        else "ai_review"
    ),
    "after_ai_review": lambda s: "hitl" if s.get("final_decision") == "escalate" else "finalize",
}


def route(state: AgentState, stage: str) -> str:
    return _ROUTE[stage](state)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        builder = StateGraph(AgentState)
        builder.add_node("validate", validate)
        builder.add_node("ai_review", ai_review)
        builder.add_node("hitl", hitl)
        builder.add_node("finalize", finalize)

        builder.add_edge(START, "validate")
        builder.add_conditional_edges(
            "validate", lambda s: route(s, "after_validate"), ["ai_review", "hitl", "finalize"]
        )
        builder.add_conditional_edges(
            "ai_review", lambda s: route(s, "after_ai_review"), ["hitl", "finalize"]
        )
        builder.add_edge("hitl", "finalize")
        builder.add_edge("finalize", END)

        config.CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
        _graph = builder.compile(checkpointer=SqliteSaver(conn))
    return _graph


# --- Service layer ---

def submit_expense(
    request: dict, thread_id: str | None = None, *, include_state: bool = False,
) -> dict:
    thread_id = thread_id or request.get("request_id") or uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()
    result = get_graph().invoke({"request": request, "run_id": run_id, "telemetry": {}}, cfg)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    if "__interrupt__" in result:
        reason_txt = result.get("pending_reason") or "Escalated for manual review."
        audit.add_pending(thread_id, run_id, result["request"], reason_txt)
        audit.write_run(_audit_record(result, status="pending_human", total_ms=total_ms))
        out = {"status": "pending_human", "thread_id": thread_id, "run_id": run_id,
               "reason": reason_txt, "total_ms": total_ms}
    else:
        out = {"status": "completed", "thread_id": thread_id, "run_id": run_id,
               "final_decision": result.get("final_decision"),
               "decided_by": result.get("decided_by"), "total_ms": total_ms}
    if include_state:
        out["state"] = result
    return out


def resume_expense(
    thread_id: str, decision: str, reviewer: str, *, include_state: bool = False,
) -> dict:
    if decision not in ("approve", "deny"):
        raise ValueError("Human decision must be 'approve' or 'deny'.")
    cfg = {"configurable": {"thread_id": thread_id}}
    result = get_graph().invoke(
        Command(resume={"decision": decision, "reviewer": reviewer}), cfg
    )
    audit.resolve_pending(thread_id, decision, reviewer)
    out = {"status": "completed", "thread_id": thread_id,
           "final_decision": result.get("final_decision"),
           "decided_by": result.get("decided_by")}
    if include_state:
        out["state"] = result
    return out
