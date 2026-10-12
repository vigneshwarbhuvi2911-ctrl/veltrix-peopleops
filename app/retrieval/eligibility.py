"""Section 5: which documents apply to a requester, and in what order of precedence.

Everything here is deterministic and unit-tested (tests/test_precedence.py). The LLM is
never asked to decide eligibility; it only ever sees chunks of eligible documents,
already labelled with their precedence rank.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Role -> access levels that role may read (Section 3.1). Unknown levels fail closed.
READABLE_LEVELS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "hr": frozenset({"all", "managers", "hr"}),
}

# Authority tiers (Section 5.1 rule 2). Unknown types rank with faq/wiki: an
# unrecognised document type should never silently override an official policy.
HIGH_AUTHORITY_TYPES = frozenset({"policy", "addendum", "memo", "handbook", "calendar"})


def can_read(role: str, access_level: str) -> bool:
    return access_level in READABLE_LEVELS.get(role, frozenset())


def covers_region(doc: dict, region: str) -> bool:
    regions = doc.get("regions") or []
    return region in regions or "GLOBAL" in regions


def precedence_key(doc: dict, region: str) -> tuple[int, int, str]:
    """Higher tuple wins: (authority, scope, recency)."""
    authority = 1 if doc.get("doc_type") in HIGH_AUTHORITY_TYPES else 0
    scope = 1 if region in (doc.get("regions") or []) else 0  # region-specific beats GLOBAL
    return authority, scope, doc.get("effective_date", "")


@dataclass(frozen=True)
class DocSets:
    eligible: list[dict]   # in force on the reference date
    upcoming: list[dict]   # readable, for the region, effective after the reference date


def partition_documents(documents: list[dict], role: str, region: str, reference: date,
                        include_upcoming: bool = True) -> DocSets:
    ref = reference.isoformat()
    eligible, upcoming = [], []
    for d in documents:
        if not can_read(role, d.get("access_level", "")):
            continue  # restricted documents are dropped before retrieval ever runs
        if not covers_region(d, region):
            continue
        if d.get("effective_date", "9999-12-31") <= ref:
            eligible.append(d)
        elif include_upcoming:
            upcoming.append(d)
    return DocSets(eligible, upcoming)


def rank_documents(docs: list[dict], region: str) -> dict[str, int]:
    """Dense rank by precedence key: 1 = strongest. Equal keys share a rank, which is
    exactly the 'tests do not decide' case of Section 5.1 rule 3."""
    keys = sorted({precedence_key(d, region) for d in docs}, reverse=True)
    rank_of = {k: i + 1 for i, k in enumerate(keys)}
    return {d["doc_id"]: rank_of[precedence_key(d, region)] for d in docs}


def holiday_dates_for(region: str, role: str, year: int, today: date,
                      documents: dict[str, dict], holidays: dict[str, list[dict]]) -> tuple[set[str], list[str]]:
    """Public holidays in `year` for the requester's region, taken from the calendar
    document(s) that win under Section 5 on `today`.

    Returns (set of ISO dates, doc_ids used). If no calendar is yet in force on `today`
    for that year (e.g. today is in December and next year's calendar starts on 1
    January), calendars effective by the end of `year` are considered instead.
    """
    def candidates(cutoff: str) -> list[dict]:
        out = []
        for doc_id, rows in holidays.items():
            d = documents.get(doc_id)
            if not d or d.get("doc_type") != "calendar":
                continue
            if not can_read(role, d.get("access_level", "")) or not covers_region(d, region):
                continue
            if d.get("effective_date", "9999") > cutoff:
                continue
            if any(r["date"].startswith(f"{year}-") for r in rows):
                out.append(d)
        return out

    cands = candidates(today.isoformat()) or candidates(f"{year}-12-31")
    if not cands:
        return set(), []
    best = max(precedence_key(d, region) for d in cands)
    winners = [d for d in cands if precedence_key(d, region) == best]
    dates = {r["date"] for d in winners for r in holidays[d["doc_id"]] if r["date"].startswith(f"{year}-")}
    return dates, sorted(d["doc_id"] for d in winners)
