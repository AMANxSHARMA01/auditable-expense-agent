"""Thin FastAPI layer over the graph: submit expenses, work the human
review queue, read audit records.

Run:  uvicorn expense_agent.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import audit
from .graph import resume_expense, submit_expense
from .schemas import ExpenseRequest

app = FastAPI(title="Auditable Expense-Approval Agent")


class HumanDecision(BaseModel):
    decision: str  # approve | deny
    reviewer: str


@app.post("/expenses")
def create_expense(request: ExpenseRequest):
    return submit_expense(request.model_dump(mode="json"))


@app.get("/queue")
def review_queue():
    return audit.list_pending()


@app.post("/queue/{thread_id}/decision")
def decide(thread_id: str, body: HumanDecision):
    pending = {p["thread_id"] for p in audit.list_pending()}
    if thread_id not in pending:
        raise HTTPException(404, "No pending review for this thread_id")
    try:
        result = resume_expense(thread_id, body.decision, body.reviewer)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return result


@app.get("/audit/{request_id}")
def audit_record(request_id: str):
    record = audit.get_run(request_id)
    if record is None:
        raise HTTPException(404, "No audit record for this request_id")
    return record
