"""LLM call #1: turn the message into a structured Plan. No corpus text is sent here."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from app.data.db import Employee
from app.llm.client import LLMClient
from app.llm.prompts import PLANNER_SYSTEM
from app.orchestration.dates import helper_calendar
from app.orchestration.session import Session

INTENTS = {"hr_question", "leave_data", "team_leave", "apply_leave", "cancel_leave", "confirm", "contact_details",
           "bulk_request", "manipulation", "greeting", "off_topic"}

# Cheap deterministic net for the most common override / extraction attempts. The planner
# also classifies these; this just makes the refusal independent of the model.
_MANIPULATION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+|above\s+)*"
    r"(instructions|rules|guidelines|prompt|restrictions)"
    r"|(system|developer|hidden)\s+(prompt|instructions|message)"
    r"|\b(jailbreak|DAN mode|developer mode)\b"
    r"|(reveal|show|print|dump|repeat)\s+(me\s+)?(your|the)\s+(prompt|instructions|configuration|rules)",
    re.I,
)


@dataclass
class Plan:
    intent: str = "hr_question"
    needs_documents: bool = False
    search_query: str = ""
    standalone_question: str = ""
    reference_date: str | None = None
    data_requests: list[dict] = field(default_factory=list)
    team_range: dict = field(default_factory=dict)
    leave: dict = field(default_factory=dict)
    cancel: dict = field(default_factory=dict)
    date_issue: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> "Plan":
        intent = d.get("intent") if d.get("intent") in INTENTS else "hr_question"
        ref = d.get("reference_date")
        try:
            ref = date.fromisoformat(ref).isoformat() if ref else None
        except (TypeError, ValueError):
            ref = None
        return cls(
            intent=intent,
            needs_documents=bool(d.get("needs_documents")) or intent == "hr_question",
            search_query=str(d.get("search_query") or ""),
            standalone_question=str(d.get("standalone_question") or ""),
            reference_date=ref,
            data_requests=[r for r in (d.get("data_requests") or []) if isinstance(r, dict)],
            team_range=d.get("team_range") or {},
            leave=d.get("leave") or {},
            cancel=d.get("cancel") or {},
            date_issue=d.get("date_issue") or None,
        )


def looks_like_manipulation(message: str) -> bool:
    return bool(_MANIPULATION.search(message))


def plan_message(llm: LLMClient, requester: Employee, session: Session, message: str, today: date,
                 request_id: str) -> Plan:
    payload = {
        "requester": {"employee_id": requester.employee_id, "role": requester.role, "region": requester.region},
        "dates": helper_calendar(today),
        "conversation_so_far": [{"role": r, "text": t} for r, t in session.history],
        "pending_proposal": (
            {"action": session.pending.name, **session.pending.args} if session.pending else None
        ),
        "draft_from_previous_clarifying_question": session.draft,
        "latest_user_message": message,
    }
    out = llm.complete_json(PLANNER_SYSTEM, json.dumps(payload, ensure_ascii=False), corpus_tokens=0,
                            purpose="plan", request_id=request_id, max_tokens=700)
    return Plan.from_json(out)
