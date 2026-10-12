"""Conversation orchestration: one message in, one ChatResponse out.

Flow
  1. A pending write proposal + a clear confirmation ("yes") -> execute it, no LLM call.
  2. Otherwise LLM call #1 (planner, no corpus text) -> Plan.
  3. Deterministic routing on the plan: refusals, permission checks, tools, clarifications
     and write proposals are all decided in code.
  4. If documents are needed: retrieval (eligible docs only, <= budget) and LLM call #2.
  5. Final PII sanitising of the answer; history stored without personal data.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date

from app.data.db import ACTIVE_STATUSES, Database, Employee
from app.llm.client import LLMClient, LLMError
from app.logging_setup import log_event
from app.orchestration.composer import compose_answer
from app.orchestration.dates import describe
from app.orchestration.pii import PiiGuard
from app.orchestration.planner import Plan, looks_like_manipulation, plan_message
from app.orchestration.session import Proposal, Session
from app.retrieval.eligibility import holiday_dates_for
from app.retrieval.retriever import KnowledgeBase
from app.tools import leave_tools as T
from app.tools.permissions import BULK_REFUSAL, MANAGERS_ONLY, NOT_PERMITTED_EMPLOYEE, can_view_leave_of, visible_employees

log = logging.getLogger(__name__)

CAPABILITIES = ("I can answer questions about Veltrix HR policies and holidays, show your leave balance and "
                "requests, and apply for or cancel leave for you.")
OFF_TOPIC = "Sorry, I can only help with HR topics. " + CAPABILITIES
MANIPULATION = ("I can't do that. I follow fixed rules and can't change them, reveal my instructions or grant "
                "extra access. " + CAPABILITIES)
CONTACT_DETAILS = "I can't share personal contact details of employees. " + CAPABILITIES
NO_ANSWER = ("The HR documents available to you don't cover this question, so I can't answer it reliably. "
             "Please contact the HR team for help.")
LLM_UNAVAILABLE = "Sorry, I couldn't process that request just now. Could you try again in a moment?"

_CONFIRM_WORD = r"(?:yes|yeah|yep|yup|y|sure|ok|okay|confirm(?:ed)?|go ahead|do it|please do|proceed|book it|submit it|cancel it|correct|that'?s right|sounds good)"
_CONFIRM = re.compile(
    rf"^\s*(?:please[\s,]*)?{_CONFIRM_WORD}(?:[\s,.!]+(?:{_CONFIRM_WORD}|please|thanks|thank you))*[\s,.!]*$", re.I
)
_DECLINE = re.compile(r"^\s*(?:no|nope|nah|don'?t|do not|stop|never ?mind|forget it)[\s,.!]*(?:thanks|thank you)?[\s.!]*$", re.I)
_EMPLOYEE_ID = re.compile(r"^[A-Za-z]\d{3,}$")


@dataclass
class ChatResponse:
    request_id: str
    answer: str
    route: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    pending_confirmation: bool = False
    history_text: str | None = None  # answer as stored in session history (personal data removed)

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "route": self.route,
            "citations": self.citations,
            "tool_calls": self.tool_calls,
            "pending_confirmation": self.pending_confirmation,
        }


class Engine:
    def __init__(self, db: Database, kb: KnowledgeBase, llm: LLMClient, today_fn, budget: int, hard_limit: int = 3000):
        self.db, self.kb, self.llm = db, kb, llm
        self.today_fn = today_fn
        self.budget, self.hard_limit = budget, hard_limit
        emps = db.all_employees()
        self.pii = PiiGuard([e.email for e in emps], [e.phone for e in emps], kb.corpus_emails)

    # ------------------------------------------------------------------ entry
    def handle(self, requester: Employee, session: Session, message: str) -> ChatResponse:
        rid = str(uuid.uuid4())
        t0 = time.time()
        today = self.today_fn()
        ctx = T.ToolContext(requester, self.db, today, self._holidays_for(today))
        intent = "-"
        try:
            resp, intent = self._dispatch(ctx, session, message, rid)
        except LLMError:
            resp = ChatResponse(rid, LLM_UNAVAILABLE, "clarify")
            intent = "llm_error"
        resp.answer = self.pii.sanitize(resp.answer)
        session.add_turn(message, self.pii.sanitize(resp.history_text or resp.answer))
        log_event(log, "chat_handled", request_id=rid, employee_id=requester.employee_id, role=requester.role,
                  region=requester.region, intent=intent, route=resp.route,
                  tools=[f"{t['name']}:{t['status']}" for t in resp.tool_calls],
                  citations=[c["doc_id"] for c in resp.citations], pending_confirmation=resp.pending_confirmation,
                  latency_ms=int((time.time() - t0) * 1000))
        return resp

    def _holidays_for(self, today: date):
        def fn(emp: Employee, year: int):
            return holiday_dates_for(emp.region, emp.role, year, today, self.kb.documents, self.kb.holidays)
        return fn

    # --------------------------------------------------------------- dispatch
    def _dispatch(self, ctx: T.ToolContext, session: Session, message: str, rid: str) -> tuple[ChatResponse, str]:
        # 1. Confirmation of a pending write: deterministic, no LLM involved.
        if session.pending and _CONFIRM.match(message):
            return self._execute_pending(ctx, session, rid), "confirm"
        if session.pending and _DECLINE.match(message):
            session.pending = None
            return ChatResponse(rid, "Okay, I haven't made any changes. Is there anything else I can help with?", "clarify"), "decline"

        if looks_like_manipulation(message):
            session.pending = None
            return ChatResponse(rid, MANIPULATION, "refuse"), "manipulation"

        # 2. Plan.
        plan = plan_message(self.llm, ctx.requester, session, message, ctx.today, rid)
        intent = plan.intent

        # 3. A pending proposal survives only a confirmation or a changed version of itself.
        if session.pending:
            if intent == "confirm":
                return self._execute_pending(ctx, session, rid), intent
            if intent not in ("apply_leave", "cancel_leave"):
                session.pending = None
        elif intent == "confirm":
            return ChatResponse(rid, "There's nothing waiting for your confirmation right now. " + CAPABILITIES, "clarify"), intent
        if intent not in ("apply_leave", "cancel_leave"):
            session.draft = None

        if intent == "greeting":
            return ChatResponse(rid, "Hello! " + CAPABILITIES + " What would you like to do?", "clarify"), intent
        if intent == "off_topic":
            return ChatResponse(rid, OFF_TOPIC, "refuse"), intent
        if intent == "manipulation":
            return ChatResponse(rid, MANIPULATION, "refuse"), intent
        if intent == "bulk_request":
            return ChatResponse(rid, BULK_REFUSAL, "refuse"), intent
        if intent == "contact_details":
            return ChatResponse(rid, CONTACT_DETAILS, "refuse"), intent
        if intent == "apply_leave":
            return self._apply(ctx, session, plan, rid), intent
        if intent == "cancel_leave":
            return self._cancel(ctx, session, plan, rid), intent
        if intent == "team_leave":
            return self._team_leave(ctx, plan, message, rid), intent
        if intent == "leave_data":
            return self._leave_data(ctx, plan, message, rid), intent
        return self._rag(ctx, plan, message, rid, tool_results=[], tool_calls=[], tool_text=""), intent

    # ------------------------------------------------------------- documents
    def _rag(self, ctx: T.ToolContext, plan: Plan, message: str, rid: str, tool_results: list[dict],
             tool_calls: list[dict], tool_text: str) -> ChatResponse:
        reference = ctx.today
        past = False
        if plan.reference_date:
            ref = date.fromisoformat(plan.reference_date)
            if ref < ctx.today:
                reference, past = ref, True
        query = f"{plan.search_query} {plan.standalone_question or message}".strip()
        retrieval = self.kb.retrieve(query, ctx.requester.role, ctx.requester.region, reference,
                                     include_upcoming=not past, budget=self.budget, hard_limit=self.hard_limit)
        log_event(log, "retrieval", request_id=rid, employee_id=ctx.requester.employee_id,
                  reference_date=reference.isoformat(), corpus_tokens=retrieval.corpus_tokens,
                  sources=[f"{s.doc['doc_id']}|{s.chunk['locator']}" for s in retrieval.sources])
        question = message
        if plan.standalone_question and plan.standalone_question.strip().lower() != message.strip().lower():
            question = f"{message}\n(In context, the user is asking: {plan.standalone_question})"
        composed = compose_answer(self.llm, ctx.requester, question, retrieval, tool_results, reference, ctx.today, rid)

        if composed.status == "no_answer":
            if tool_calls:
                text = (tool_text + "\n\n" if tool_text else "") + (
                    "The HR documents don't cover the rest of your question; please contact the HR team for that part.")
                return ChatResponse(rid, text.strip(), "tool", tool_calls=tool_calls)
            return ChatResponse(rid, NO_ANSWER, "no_answer")
        route = "hybrid" if tool_calls else "rag"
        return ChatResponse(rid, composed.answer, route, citations=composed.citations, tool_calls=tool_calls)

    # ------------------------------------------------------------ read tools
    def _resolve_employee(self, ctx: T.ToolContext, ref: str | None) -> tuple[str | None, ChatResponse | None, list[Employee]]:
        """Map 'self' / an ID / a typed name to an employee ID the requester may see.

        Returns (employee_id, None, []) on success, (None, None, candidates) if ambiguous,
        and (None, refusal, []) otherwise. Unknown and not-permitted targets get the same
        refusal so the answer never reveals whether an employee exists.
        """
        me = ctx.requester
        ref = (ref or "self").strip()
        if ref.lower() in ("self", "me", "myself", "i", "my", me.employee_id.lower()):
            return me.employee_id, None, []
        if _EMPLOYEE_ID.match(ref):
            eid = ref.upper()
            return (eid, None, []) if can_view_leave_of(me, eid, self.db) else (None, None, [])
        tokens = [t for t in re.findall(r"[a-z]+", ref.lower().replace("'s", "")) if t]
        if not tokens:
            return None, None, []
        matches = [e for e in visible_employees(me, self.db)
                   if all(t in re.findall(r"[a-z]+", e.full_name.lower()) for t in tokens)]
        if len(matches) == 1:
            return matches[0].employee_id, None, []
        if len(matches) > 1:
            return None, None, matches
        return None, None, []

    def _leave_data(self, ctx: T.ToolContext, plan: Plan, message: str, rid: str) -> ChatResponse:
        requests_ = plan.data_requests or [{"tool": "get_leave_balance", "employee": "self"}]
        resolved: list[tuple[dict, str]] = []
        for dr in requests_:
            eid, _, candidates = self._resolve_employee(ctx, dr.get("employee"))
            if candidates:
                names = "; ".join(f"{e.employee_id} ({e.full_name})" for e in candidates)
                return ChatResponse(rid, f"I found more than one employee matching that name: {names}. Which one do you mean?",
                                    "clarify", history_text="I found more than one employee matching that name: "
                                    + ", ".join(e.employee_id for e in candidates) + ". Which one do you mean?")
            if eid is None:
                return ChatResponse(rid, NOT_PERMITTED_EMPLOYEE, "refuse")
            resolved.append((dr, eid))
        others = {eid for _, eid in resolved if eid != ctx.requester.employee_id}
        if len(others) > 1:
            return ChatResponse(rid, BULK_REFUSAL, "refuse")

        results: list[T.ToolResult] = []
        for dr, eid in resolved:
            lt = dr.get("leave_type") if dr.get("leave_type") in T.LEAVE_TYPES else None
            if dr.get("tool") == "list_leave_requests":
                st = dr.get("status") if dr.get("status") in T.REQUEST_STATUSES else None
                results.append(T.list_leave_requests(ctx, eid, st))
            else:
                results.append(T.get_leave_balance(ctx, eid, lt))
        if any(r.status == "error" for r in results):
            # Defence in depth: the tool's own check disagreed with the resolver.
            return ChatResponse(rid, NOT_PERMITTED_EMPLOYEE, "refuse")

        text = "\n".join(self._render_read(ctx, r) for r in results)
        calls = [r.public() for r in results]
        if plan.needs_documents:
            llm_results = [{"tool": r.name, "args": r.args, "status": r.status, "data": r.data} for r in results]
            return self._rag(ctx, plan, message, rid, llm_results, calls, text)
        return ChatResponse(rid, text, "tool", tool_calls=calls)

    def _render_read(self, ctx: T.ToolContext, r: T.ToolResult) -> str:
        eid = r.data.get("employee_id")
        who_self = eid == ctx.requester.employee_id
        if r.name == "get_leave_balance":
            parts = [f"{lt} leave: {T.fmt_days(v)} day(s)" for lt, v in r.data["balances"].items()]
            subject = "Your current balance" if who_self else f"Current balance for {eid}"
            return f"{subject} — " + ("; ".join(parts) if parts else "no balance on record") + "."
        reqs = r.data["requests"]
        subject = "You have" if who_self else f"{eid} has"
        st = r.args.get("status")
        if not reqs:
            return f"{subject} no{' ' + st if st else ''} leave requests."
        lines = [f"- {q['request_id']}: {q['leave_type']} leave {q['start_date']} to {q['end_date']}, "
                 f"{q['working_days']} working day(s), {q['status']}" for q in reqs]
        return f"{subject} {len(reqs)}{' ' + st if st else ''} leave request(s):\n" + "\n".join(lines)

    def _team_leave(self, ctx: T.ToolContext, plan: Plan, message: str, rid: str) -> ChatResponse:
        if ctx.requester.role != "manager":
            return ChatResponse(rid, MANAGERS_ONLY, "refuse")
        if plan.date_issue:
            return ChatResponse(rid, plan.date_issue, "clarify")
        s, e = plan.team_range.get("start_date"), plan.team_range.get("end_date")
        if not s and not e:
            return ChatResponse(rid, "Which dates should I check your team's leave for?", "clarify")
        s, e = s or e, e or s
        r = T.get_team_leave(ctx, s, e)
        if r.status == "error":
            return ChatResponse(rid, r.error or MANAGERS_ONLY, "tool", tool_calls=[r.public()])
        names = {x.employee_id: x.full_name for x in self.db.direct_reports(ctx.requester.employee_id)}
        reqs = r.data["requests"]
        if not reqs:
            text = f"None of your direct reports have pending or approved leave between {s} and {e}."
            hist = text
        else:
            def line(q, with_names: bool) -> str:
                who = f"{q['employee_id']} ({names.get(q['employee_id'], '')})" if with_names else q["employee_id"]
                return (f"- {who}: {q['leave_type']} leave {q['start_date']} to {q['end_date']} "
                        f"({q['working_days']} working day(s), {q['status']})")
            head = f"Your direct reports with pending or approved leave between {s} and {e}:\n"
            text = head + "\n".join(line(q, True) for q in reqs)
            hist = head + "\n".join(line(q, False) for q in reqs)
        calls = [r.public()]
        if plan.needs_documents:
            llm_results = [{"tool": r.name, "args": r.args, "status": r.status, "data": r.data}]
            resp = self._rag(ctx, plan, message, rid, llm_results, calls, hist)
            return resp
        return ChatResponse(rid, text, "tool", tool_calls=calls, history_text=hist)

    # ----------------------------------------------------------- write tools
    def _apply(self, ctx: T.ToolContext, session: Session, plan: Plan, rid: str) -> ChatResponse:
        draft = (session.draft or {}).get("leave", {}) if (session.draft or {}).get("intent") == "apply_leave" else {}
        base = session.pending.args if session.pending and session.pending.name == "apply_leave" else {}
        leave = {k: plan.leave.get(k) or draft.get(k) or base.get(k) for k in ("leave_type", "start_date", "end_date")}
        if leave["leave_type"] not in T.LEAVE_TYPES:
            leave["leave_type"] = None
        session.pending = None

        if plan.date_issue:
            session.draft = {"intent": "apply_leave", "leave": leave}
            return ChatResponse(rid, plan.date_issue, "clarify")
        missing = []
        if not leave["leave_type"]:
            missing.append("the leave type (annual or sick)")
        if not leave["start_date"] and not leave["end_date"]:
            missing.append("the start and end dates")
        elif not leave["start_date"]:
            missing.append("the start date")
        elif not leave["end_date"]:
            missing.append("the end date (or tell me if it's a single day)")
        if missing:
            session.draft = {"intent": "apply_leave", "leave": leave}
            return ChatResponse(rid, "To apply for leave I need " + " and ".join(missing) + ".", "clarify")

        session.draft = None
        r = T.apply_leave(ctx, leave["leave_type"], leave["start_date"], leave["end_date"], execute=False)
        if r.status == "error":
            return ChatResponse(rid, f"I can't submit this leave request: {r.error}", "tool", tool_calls=[r.public()])
        d = r.data
        hol = f" Public holidays in that range are not counted ({', '.join(d['holidays_in_range'])})." if d["holidays_in_range"] else ""
        text = (f"Please confirm this leave request: {d['leave_type']} leave from {describe(d['start_date'])} to "
                f"{describe(d['end_date'])}, {d['working_days']} working day(s).{hol} Your {d['leave_type']} balance "
                f"would go from {T.fmt_days(d['balance_before'])} to {T.fmt_days(d['balance_before'] - d['working_days'])} "
                f"day(s). Reply \"yes\" to submit it.")
        session.pending = Proposal("apply_leave", r.args)
        return ChatResponse(rid, text, "tool", tool_calls=[r.public()], pending_confirmation=True)

    def _cancel(self, ctx: T.ToolContext, session: Session, plan: Plan, rid: str) -> ChatResponse:
        draft = (session.draft or {}).get("cancel", {}) if (session.draft or {}).get("intent") == "cancel_leave" else {}
        c = {k: plan.cancel.get(k) or draft.get(k) for k in ("request_id", "start_date", "end_date", "leave_type")}
        session.pending = None
        if plan.date_issue:
            session.draft = {"intent": "cancel_leave", "cancel": c}
            return ChatResponse(rid, plan.date_issue, "clarify")

        request_id = c["request_id"]
        if not request_id:
            # Internal lookup over the requester's OWN requests only (not a tool call, and
            # never reads anyone else's data).
            own = [r for r in self.db.list_requests(ctx.requester.employee_id) if r.status in ACTIVE_STATUSES]
            s, e = c["start_date"], c["end_date"]
            hinted = bool(s or e or c["leave_type"])
            if s or e:
                s, e = s or e, e or s
                own = [r for r in own if r.start_date <= e and r.end_date >= s]
            if c["leave_type"] in T.LEAVE_TYPES:
                own = [r for r in own if r.leave_type == c["leave_type"]]
            if not hinted:
                own = [r for r in own if r.start_date > ctx.today.isoformat()]
            if len(own) == 1:
                request_id = own[0].request_id
            else:
                session.draft = {"intent": "cancel_leave", "cancel": c}
                if not own:
                    msg = ("I couldn't find a pending or approved leave request of yours matching that. "
                           "Which request do you want to cancel? Please give its dates or request ID.")
                else:
                    lines = "\n".join(f"- {r.request_id}: {r.leave_type} leave {r.start_date} to {r.end_date} ({r.status})" for r in own)
                    msg = f"You have more than one request that could match:\n{lines}\nWhich one do you want to cancel?"
                return ChatResponse(rid, msg, "clarify")

        session.draft = None
        r = T.cancel_leave(ctx, request_id, execute=False)
        if r.status == "error":
            return ChatResponse(rid, f"I can't cancel that request: {r.error}", "tool", tool_calls=[r.public()])
        d = r.data
        text = (f"Please confirm you want to cancel request {d['request_id']}: {d['leave_type']} leave from "
                f"{describe(d['start_date'])} to {describe(d['end_date'])} ({d['working_days']} working day(s), "
                f"{d['status']}). {d['working_days']} day(s) will be returned to your {d['leave_type']} balance. "
                f"Reply \"yes\" to cancel it.")
        session.pending = Proposal("cancel_leave", r.args)
        return ChatResponse(rid, text, "tool", tool_calls=[r.public()], pending_confirmation=True)

    def _execute_pending(self, ctx: T.ToolContext, session: Session, rid: str) -> ChatResponse:
        p = session.pending
        session.pending = None
        session.draft = None
        if p.name == "apply_leave":
            r = T.apply_leave(ctx, p.args["leave_type"], p.args["start_date"], p.args["end_date"], execute=True)
            if r.status == "error":
                return ChatResponse(rid, f"I couldn't submit the leave request: {r.error}", "tool", tool_calls=[r.public()])
            d = r.data
            text = (f"Done. Your {d['leave_type']} leave request {d['request_id']} for {d['start_date']} to {d['end_date']} "
                    f"({d['working_days']} working day(s)) has been submitted with status pending. Your remaining "
                    f"{d['leave_type']} balance is {T.fmt_days(d['balance_after'])} day(s).")
            return ChatResponse(rid, text, "tool", tool_calls=[r.public()])
        r = T.cancel_leave(ctx, p.args["request_id"], execute=True)
        if r.status == "error":
            return ChatResponse(rid, f"I couldn't cancel the request: {r.error}", "tool", tool_calls=[r.public()])
        d = r.data
        text = (f"Done. Request {d['request_id']} ({d['leave_type']} leave {d['start_date']} to {d['end_date']}) is now "
                f"cancelled and {d['working_days']} day(s) were returned. Your {d['leave_type']} balance is now "
                f"{T.fmt_days(d['balance_after'])} day(s).")
        return ChatResponse(rid, text, "tool", tool_calls=[r.public()])
