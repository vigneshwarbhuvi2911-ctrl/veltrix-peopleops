"""Checks against whatever is in ./data (run with `make test-corpus`).

These assert properties, not specific corpus values, so they hold for any corpus that
follows the brief: restricted documents never retrieved for the wrong role, the token
budget always respected, every format ingested, and our working-day calculation in
agreement with the working_days the HR system stored for existing requests.
"""
import json
import os
from datetime import date
from pathlib import Path

import pytest

from app.data.db import Database
from app.ingestion.pipeline import build_index
from app.retrieval.eligibility import can_read, holiday_dates_for
from app.retrieval.retriever import KnowledgeBase
from app.tools.working_days import count_working_days

DATA = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
pytestmark = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="no ./data")
TODAY = date.fromisoformat(os.environ.get("APP_TODAY") or date.today().isoformat())

QUERIES = [
    "sick leave days per year", "annual leave carry forward", "parental leave weeks", "work from home days",
    "salary bands compensation", "performance improvement plan", "approving leave managers", "expense claims",
    "public holidays", "notice period", "bereavement leave", "accrual rate per month", "learning budget",
]


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(build_index(str(DATA / "corpus"), str(DATA / "manifest.json")))


def test_every_manifest_document_ingested(kb):
    manifest = json.loads((DATA / "manifest.json").read_text())
    missing = [d["doc_id"] for d in manifest if not kb.documents.get(d["doc_id"], {}).get("chunk_count")]
    assert missing == []


@pytest.mark.parametrize("role", ["employee", "manager", "hr"])
@pytest.mark.parametrize("region", ["IN", "US", "DE"])
def test_no_restricted_document_and_budget(kb, role, region):
    for q in QUERIES:
        r = kb.retrieve(q, role, region, TODAY, True, 2800)
        assert r.corpus_tokens <= 3000
        for s in r.sources:
            assert can_read(role, s.doc["access_level"]), (q, s.doc["doc_id"])
            assert region in s.doc["regions"] or "GLOBAL" in s.doc["regions"]


def test_working_days_agree_with_existing_requests(kb):
    db = Database(str(DATA / "veltrix.db"))
    mismatches = []
    for emp in db.all_employees():
        hol, _ = holiday_dates_for(emp.region, emp.role, 2026, TODAY, kb.documents, kb.holidays)
        for r in db.list_requests(emp.employee_id):
            if not r.start_date.startswith("2026"):
                continue
            wd = count_working_days(date.fromisoformat(r.start_date), date.fromisoformat(r.end_date), hol)
            if wd != r.working_days:
                mismatches.append((r.request_id, r.working_days, wd))
    assert mismatches == []
