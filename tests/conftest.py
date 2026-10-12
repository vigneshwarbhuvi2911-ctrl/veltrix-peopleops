"""Shared fixtures.

The suite builds a small synthetic company (tests/fixtures/builder.py) so it is
self-contained and gives the same result whatever is in ./data. The LLM is replaced by
FakeLLM: scripted plans and answers, plus a record of every prompt so tests can assert
exactly what would have been sent to a real model.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import date

import pytest

from app.data.db import Database
from app.ingestion.pipeline import build_index
from app.llm.client import LLMClient
from app.llm.prompts import ANSWER_SYSTEM, PLANNER_SYSTEM
from app.orchestration.engine import Engine
from app.orchestration.session import SessionStore
from app.retrieval.retriever import KnowledgeBase
from tests.fixtures.builder import build_company

TODAY = date(2026, 10, 12)  # a Monday

# Synthetic employee IDs (see tests/fixtures/builder.py)
HR, MGR_IN, EMP_IN, EMP_IN2, MGR_US, EMP_US, EMP_DE, EMP_IN3 = (
    "E9001", "E9002", "E9003", "E9004", "E9005", "E9006", "E9007", "E9008")


def default_plan(**overrides) -> dict:
    plan = {
        "intent": "hr_question", "needs_documents": True, "search_query": "", "reference_date": None,
        "data_requests": [], "team_range": {"start_date": None, "end_date": None},
        "leave": {"leave_type": None, "start_date": None, "end_date": None},
        "cancel": {"request_id": None, "start_date": None, "end_date": None, "leave_type": None},
        "date_issue": None,
    }
    plan.update(overrides)
    return plan


class FakeLLM:
    """Stands in for the provider. `plans` maps a user message to the planner's JSON."""

    def __init__(self):
        self.plans: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.answer_fn = None

    def transport(self, system: str, user: str, max_tokens: int) -> str:
        if system == PLANNER_SYSTEM:
            msg = json.loads(user)["latest_user_message"]
            self.calls.append({"kind": "plan", "user": user})
            return json.dumps(self.plans.get(msg, default_plan(search_query=msg)))
        assert system == ANSWER_SYSTEM
        self.calls.append({"kind": "answer", "user": user})
        if self.answer_fn:
            return json.dumps(self.answer_fn(user))
        labels = re.findall(r"^\[(S\d+)\] doc_id=", user, re.M)
        if not labels:
            return json.dumps({"status": "no_answer", "answer": "", "citations": []})
        return json.dumps({"status": "answered", "answer": f"Answer [{labels[0]}].", "citations": [labels[0]]})

    def answer_prompts(self) -> list[str]:
        return [c["user"] for c in self.calls if c["kind"] == "answer"]

    def sources_sent(self) -> list[str]:
        out = []
        for u in self.answer_prompts():
            out += re.findall(r"^\[S\d+\] doc_id=([A-Za-z0-9-]+)", u, re.M)
        return out


@pytest.fixture(scope="session")
def company(tmp_path_factory) -> dict:
    return build_company(tmp_path_factory.mktemp("company"))


@pytest.fixture(scope="session")
def index(company) -> dict:
    return build_index(company["corpus"], company["manifest"])


@pytest.fixture(scope="session")
def kb(index) -> KnowledgeBase:
    return KnowledgeBase(index)


@pytest.fixture()
def db(company, tmp_path) -> Database:
    dst = tmp_path / "veltrix.db"
    shutil.copy(company["db"], dst)
    return Database(str(dst))


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def engine(db, kb, fake_llm) -> Engine:
    llm = LLMClient("fake", transport=fake_llm.transport)
    return Engine(db, kb, llm, lambda: TODAY, budget=2800)


@pytest.fixture()
def sessions() -> SessionStore:
    return SessionStore()


@pytest.fixture()
def chat(engine, db, sessions):
    """chat(employee_id, message, session='s1') -> response dict (engine level, no HTTP)."""
    def _chat(employee_id: str, message: str, session: str = "s1") -> dict:
        emp = db.get_employee(employee_id)
        s = sessions.get_or_create(f"{employee_id}:{session}", employee_id)
        return engine.handle(emp, s, message).to_json()
    return _chat


@pytest.fixture()
def tool_ctx(db, kb):
    from app.retrieval.eligibility import holiday_dates_for
    from app.tools.leave_tools import ToolContext

    def make(employee_id: str, today: date = TODAY) -> ToolContext:
        emp = db.get_employee(employee_id)
        return ToolContext(emp, db, today, lambda e, y: holiday_dates_for(e.region, e.role, y, today, kb.documents, kb.holidays))
    return make
