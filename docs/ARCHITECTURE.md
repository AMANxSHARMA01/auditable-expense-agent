# Architecture

This document describes the system design of the **Auditable Expense-Approval Agent**: a finance-grade pipeline that decides expense requests as **approve**, **deny**, or **escalate**.

**Design thesis:** *The LLM is advisory; policy is law.* The model produces a reasoned recommendation with citations, but approval authority lives in deterministic code. Even a perfect prompt injection cannot move money.

---

## Decision criteria

Every expense ends as **approve**, **deny**, or **escalate** (human review). Four layers decide, in order:

| Layer | Who decides | Can approve money? |
|-------|-------------|-------------------|
| 1. Hard rules (code) | `guardrails.py` | Deny only — or escalate |
| 2. AI review | Gemini + policy clauses | Recommends approve/deny/escalate |
| 3. Safety check (code) | `guardrails.py` again | Can override AI approve → escalate |
| 4. Human | Finance reviewer | Final approve or deny on escalated cases |

### Deny — automatic, no human needed

| Condition | Example | Policy |
|-----------|---------|--------|
| Invalid request data | Missing vendor, zero amount | — |
| Restricted vendor | "Shadow Consulting Ltd" | EXP-013 |
| No receipt above €25 | €80 meal, no receipt attached | EXP-001 |

### Escalate — routed to human review

| Condition | Example | Policy |
|-----------|---------|--------|
| Amount ≥ €10,000 | €15k equipment purchase | EXP-003 |
| Duplicate submission | Same employee/vendor/amount within 24 h | EXP-014 |
| Amount > €500 and AI says approve | €800 travel — auto-cap enforced in code | EXP-002 |
| AI cites a clause it wasn't shown | Hallucinated policy reference | — |
| AI approves with low confidence | Confidence below 0.6 | — |
| AI or retrieval fails | API error, empty policy match | EXP-019 |
| AI says escalate | Unclear purpose, ambiguous category | EXP-019 |

### Approve — automatic, no human needed

All of the following must be true:

- Passes all hard rules (valid data, receipt if needed, vendor OK, amount < €10k, not a duplicate)
- AI recommends **approve**
- Amount ≤ **€500** (auto-approve limit)
- AI confidence ≥ **0.6**
- Every cited clause was actually retrieved (grounded)

Typical examples: €35 taxi with receipt, €120 software under team-lead rules, €40 business meal.

### Human decision (after escalate)

Reviewer sees the request, AI recommendation, and escalation reason. They choose **approve** or **deny** — that decision is final and attributed as `human:<reviewer>`.

---

## Flow (simplified)

```mermaid
flowchart LR
    A[Submit expense] --> B[Validate]
    B --> C{Hard rules}
    C -->|deny| F[Audit + done]
    C -->|escalate| E[Human review]
    C -->|pass| D[AI + policy]
    D --> G{Safety check}
    G -->|approve or deny| F
    G -->|escalate| E
    E --> F
```

**Five steps:**

1. **Validate** (`validate`) — schema check + hard rules before any AI call.
2. **AI review** (`ai_review`) — retrieve policy clauses, run Gemini, apply post-guardrails.
3. **Human review** (`hitl`) — only if escalated; pauses until a reviewer responds.
4. **Audit** (`finalize`) — every path writes one immutable record.

**Amount bands (defaults):**

| Amount | What happens |
|--------|--------------|
| ≤ €25 | Receipt optional |
| > €25 | Receipt required or **deny** |
| ≤ €500 | AI can **approve** (if policy-compliant) |
| > €500 | AI may recommend approve, but code **escalates** |
| ≥ €10,000 | **Escalate** immediately — AI never consulted |

---

```mermaid
flowchart TB
    subgraph Entry["Entry points"]
        API["FastAPI REST<br/>expense_agent/api.py"]
        Demo["CLI demo<br/>scripts/demo.py"]
        Eval["Eval harness<br/>eval/run_eval.py"]
    end

    subgraph Core["Core pipeline — expense_agent/"]
        Graph["graph.py<br/>LangGraph state machine"]
        Guard["guardrails.py<br/>Deterministic rules"]
        LLM["llm.py<br/>Gemini reasoning"]
        RAG["retrieval.py<br/>Policy RAG"]
        Audit["audit.py<br/>Audit + review queue"]
    end

    subgraph External["External services"]
        Gemini["Google Gemini<br/>2.5-flash + embeddings"]
    end

    subgraph Storage["Persistence (SQLite)"]
        AuditDB["data/expense_agent.db<br/>audit runs + review queue"]
        CheckpointDB["data/checkpoints.db<br/>LangGraph HITL checkpoints"]
    end

    subgraph Policy["Policy corpus"]
        PolicyMD["policy/expense_policy.md<br/>19 clauses EXP-001…019"]
    end

    API --> Graph
    Demo --> Graph
    Eval --> Graph
    Graph --> Guard
    Graph --> RAG
    Graph --> LLM
    Graph --> Audit
    RAG --> Gemini
    LLM --> Gemini
    RAG --> PolicyMD
    Audit --> AuditDB
    Graph --> CheckpointDB
```

## System overview

| Layer | Technology | Role |
|-------|------------|------|
| Orchestration | LangGraph | State machine with human-review pauses |
| LLM | Gemini 2.5-flash | One structured reasoning call per request |
| Embeddings | gemini-embedding-001 | Policy clause retrieval |
| API | FastAPI | REST endpoints |
| Storage | SQLite | Audit trail, review queue, checkpoints |

## Repository layout

```
auditable-expense-agent/
├── expense_agent/              # Application package
│   ├── graph.py                # Pipeline orchestration (main entry for logic)
│   ├── guardrails.py           # Deterministic pre/post rules
│   ├── llm.py                  # Prompts + Gemini structured output
│   ├── retrieval.py            # Policy RAG (embed + top-k)
│   ├── audit.py                # Immutable audit store + human queue
│   ├── schemas.py              # Pydantic contracts
│   ├── config.py               # Thresholds and environment variables
│   └── api.py                  # REST endpoints
├── policy/
│   └── expense_policy.md       # 19 retrievable clauses (EXP-001 … EXP-019)
├── scripts/
│   └── demo.py                 # Four end-to-end scenarios
├── eval/
│   ├── run_eval.py             # 50-case golden dataset replay
│   └── golden_cases.jsonl      # Evaluation cases
├── tests/
│   └── test_guardrails.py      # Guardrail unit tests (no API key)
└── docs/
    └── ARCHITECTURE.md           # This document
```

**Runtime-generated (gitignored):**

| Path | Purpose |
|------|---------|
| `data/expense_agent.db` | Audit runs and review queue |
| `data/checkpoints.db` | LangGraph HITL checkpoint state |
| `.cache/policy_embeddings_*.json` | Cached clause embeddings |
| `eval/runs/` | Eval scorecards |

---

## Graph nodes

Implementation detail for the flow above — each box maps to a node in `expense_agent/graph.py`:

| Node | What it does |
|------|--------------|
| `validate` | Schema validation + hard rules (deny / escalate / pass) |
| `ai_review` | Retrieve policy, LLM reasoning, post-guardrail overrides |
| `hitl` | Pause for human reviewer |
| `finalize` | Write audit record |

### Rule IDs (for audit logs)

| When | Rule ID |
|------|---------|
| Bad request data | `GR-INTAKE-SCHEMA` |
| Blacklisted vendor | `GR-PRE-BLACKLIST` |
| Missing receipt > €25 | `GR-PRE-RECEIPT` |
| Amount ≥ €10k | `GR-PRE-CEILING` |
| Duplicate within 24 h | `GR-PRE-DUPLICATE` |
| AI approve > €500 | `GR-POST-LIMIT` |
| AI cites wrong clause | `GR-POST-GROUNDING` |
| AI low-confidence approve | `GR-POST-CONFIDENCE` |
| Retrieval failed | `GR-RETRIEVAL-FAILCLOSED` |
| AI call failed | `system:llm_failure` |

Thresholds are configurable in `expense_agent/config.py`.

---

## Component details

### Policy retrieval (`retrieval.py`)

1. Parse `policy/expense_policy.md` into 19 clauses via regex (`### EXP-NNN — Title`).
2. Embed all clauses once; cache keyed by policy content hash and model.
3. On each request, embed a query built from category, amount, vendor, and description.
4. Rank clauses by cosine similarity; return top-k with scores.
5. Scores and clause IDs are stored in the audit record for full traceability.

No vector database — 19 clauses fit in memory. The retrieval **contract** (which clauses were shown, with what scores) matters more than retrieval infrastructure at this scale.

### LLM reasoning (`llm.py`)

One temperature-0 Gemini call per request that reaches the reasoning step:

- **System instruction:** ground decisions in provided clauses only; treat employee note as untrusted data; escalate when ambiguous (EXP-019).
- **Structured output:** Pydantic `LLMDecision` via `response_schema`.
- **Reliability:** schema retry (max 2), 429 backoff (max 5 waits), bounded `thinking_budget` and `max_output_tokens`.
- **Failure:** returns `None` → post-guardrails escalates as `system:llm_failure`.

### Human-in-the-loop (`graph.py`)

Escalations park at a LangGraph `interrupt()` backed by a SQLite checkpointer (`data/checkpoints.db`):

1. `submit_expense()` detects `__interrupt__` in the graph result.
2. A provisional audit record and review-queue entry are written.
3. API returns `{ status: "pending_human", thread_id, reason }`.
4. Reviewer calls `resume_expense(thread_id, decision, reviewer)`.
5. Graph resumes via `Command(resume=…)` → `hitl` node → `finalize`.
6. Final decision is attributed as `human:<reviewer>`.

The checkpoint survives process restarts.

### Audit trail (`audit.py`)

One immutable record per run in `data/expense_agent.db`:

- Request payload
- Retrieved clauses with similarity scores
- Full LLM output (if any)
- Every guardrail event
- Per-node latency, token counts, cost
- Final decision and `decided_by` attribution
- Human decision (if applicable)

Status is either `completed` or `pending_human`.

---

## External interfaces

### Service layer

| Function | File | Purpose |
|----------|------|---------|
| `submit_expense(request, thread_id?)` | `graph.py` | Run a request through the pipeline |
| `resume_expense(thread_id, decision, reviewer)` | `graph.py` | Resume a parked HITL run |
| `get_graph()` | `graph.py` | Compiled LangGraph singleton with SQLite checkpointer |

### REST API (`api.py`)

| Method | Path | Action |
|--------|------|--------|
| `POST` | `/expenses` | Submit expense → decision or `pending_human` |
| `GET` | `/queue` | List pending human reviews |
| `POST` | `/queue/{thread_id}/decision` | Resume parked run with reviewer decision |
| `GET` | `/audit/{request_id}` | Full audit JSON for any past run |

Run the server:

```bash
uvicorn expense_agent.api:app --reload
```

### Other entry points

| Entry | Command | Purpose |
|-------|---------|---------|
| CLI demo | `python -m scripts.demo` | Four scenarios end to end |
| Eval harness | `python -m eval.run_eval` | 50-case regression scorecard |
| Unit tests | `pytest tests/ -q` | Guardrail tests (no API key) |

---

## End-to-end flows

**Automatic:** `POST /expenses` → validate → rules → AI → rules → audit → `{ status: "completed" }`

**Human review:** same, but escalated cases return `{ status: "pending_human", thread_id }` → reviewer calls `POST /queue/{thread_id}/decision` → audit → done.

---

## Design principles

1. **Fail-closed everywhere** — LLM failure, empty retrieval, low confidence, and ungrounded citations all resolve to escalate, never to approve.
2. **Pre-guardrails save cost** — Settled cases skip the LLM entirely (~14% of eval cases, zero tokens).
3. **Post-guardrails override the LLM** — Code enforces the auto-approve cap and grounding check regardless of model output.
4. **Durable HITL** — `interrupt()` plus SQLite checkpointing survives restarts.
5. **Full auditability** — Every run stores enough data to reconstruct the decision and attribute it.
6. **Untrusted input handling** — Employee free-text is delimited and declared as data, not instructions.
7. **Thresholds in code** — Policy limits live in `config.py`; prompts are advisory only.

---

## Evaluation

`python -m eval.run_eval` replays 50 golden cases from `eval/golden_cases.jsonl` against fresh databases and writes a scorecard to `eval/runs/`.

Scoring is asymmetric: a false approval moves money; a false escalation costs a reviewer a minute. The headline metric is **unauthorized-approval rate**, which must be 0 — guaranteed for over-limit amounts because the cap is enforced in code, not in the prompt.

See [README.md](../README.md#evaluation) for latest benchmark results.

---

## Key files by concept

| Concept | File |
|---------|------|
| Pipeline orchestration | `expense_agent/graph.py` |
| HTTP entry point | `expense_agent/api.py` |
| Deterministic safety | `expense_agent/guardrails.py` |
| LLM reasoning and prompts | `expense_agent/llm.py` |
| Policy RAG | `expense_agent/retrieval.py` |
| Audit and queue | `expense_agent/audit.py` |
| Data models | `expense_agent/schemas.py` |
| Configuration | `expense_agent/config.py` |
| Policy source | `policy/expense_policy.md` |
| Eval harness | `eval/run_eval.py` |
