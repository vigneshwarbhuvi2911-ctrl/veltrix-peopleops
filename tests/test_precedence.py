"""Section 5: eligibility, precedence, unresolved conflicts and upcoming changes.

Uses the worked example from Section 5.2 of the brief (volunteer days), reproduced in
the synthetic corpus.
"""
import re
from datetime import date

from app.retrieval.eligibility import (can_read, partition_documents, precedence_key, rank_documents)
from tests.conftest import EMP_DE, EMP_IN, default_plan

DOCS = [
    {"doc_id": "V1", "doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2025-01-01", "access_level": "all"},
    {"doc_id": "V2", "doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2026-04-01", "access_level": "all"},
    {"doc_id": "FAQ", "doc_type": "faq", "regions": ["GLOBAL"], "effective_date": "2026-05-01", "access_level": "all"},
    {"doc_id": "DE", "doc_type": "addendum", "regions": ["DE"], "effective_date": "2025-06-01", "access_level": "all"},
    {"doc_id": "V3", "doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2027-01-01", "access_level": "all"},
]
REF = date(2026, 9, 15)


def ids(docs):
    return sorted(d["doc_id"] for d in docs)


def test_eligibility_india_employee():
    sets = partition_documents(DOCS, "employee", "IN", REF)
    assert ids(sets.eligible) == ["FAQ", "V1", "V2"]   # DE not their region, V3 not yet effective
    assert ids(sets.upcoming) == ["V3"]


def test_india_v2_wins_and_v3_is_an_upcoming_change():
    sets = partition_documents(DOCS, "employee", "IN", REF)
    ranks = rank_documents(sets.eligible + sets.upcoming, "IN")
    eligible_best = min(sets.eligible, key=lambda d: ranks[d["doc_id"]])
    assert eligible_best["doc_id"] == "V2"            # FAQ loses on authority, V1 on recency
    assert ranks["V3"] < ranks["V2"]                  # V3 would win on its date -> mention it


def test_germany_addendum_wins_on_scope_and_v3_not_mentioned():
    sets = partition_documents(DOCS, "employee", "DE", REF)
    ranks = rank_documents(sets.eligible + sets.upcoming, "DE")
    assert min(ranks, key=ranks.get) == "DE"
    assert ranks["V3"] > ranks["DE"]                  # addendum would still win in 2027


def test_past_reference_date_only_v1():
    sets = partition_documents(DOCS, "employee", "IN", date(2025, 8, 1), include_upcoming=False)
    assert ids(sets.eligible) == ["V1"]


def test_authority_beats_scope_and_recency():
    faq_regional_new = {"doc_type": "faq", "regions": ["IN"], "effective_date": "2026-09-01"}
    policy_global_old = {"doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2024-01-01"}
    assert precedence_key(policy_global_old, "IN") > precedence_key(faq_regional_new, "IN")


def test_scope_beats_recency():
    addendum_old = {"doc_type": "addendum", "regions": ["IN"], "effective_date": "2025-01-01"}
    policy_new = {"doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2026-06-01"}
    assert precedence_key(addendum_old, "IN") > precedence_key(policy_new, "IN")


def test_identical_keys_share_a_rank_unresolved_conflict():
    a = {"doc_id": "A", "doc_type": "policy", "regions": ["GLOBAL"], "effective_date": "2026-01-01"}
    b = {"doc_id": "B", "doc_type": "handbook", "regions": ["GLOBAL"], "effective_date": "2026-01-01"}
    r = rank_documents([a, b], "IN")
    assert r["A"] == r["B"] == 1


def test_unknown_access_level_fails_closed():
    assert not can_read("hr", "board-only")
    assert not can_read("employee", "managers") and can_read("manager", "managers") and can_read("hr", "hr")


# ---------------------------------------------------------- through retrieval
def _ranks_in_prompt(prompt: str) -> dict[str, tuple[int, str]]:
    out = {}
    for doc_id, status, rank in re.findall(r"doc_id=([A-Z0-9-]+) .*?status=([A-Z ]+?) \| precedence_rank=(\d+)", prompt):
        out[doc_id] = (int(rank), status)
    return out


def test_prompt_labels_india(chat, fake_llm):
    fake_llm.plans["How many volunteer days do I get?"] = default_plan(search_query="volunteer days per year")
    r = chat(EMP_IN, "How many volunteer days do I get?")
    ranks = _ranks_in_prompt(fake_llm.answer_prompts()[-1])
    assert "ADD-VOL-DE" not in ranks
    assert ranks["POL-VOL-V3"] == (1, "NOT YET IN FORCE")
    assert ranks["POL-VOL-V2"][0] == 2 and ranks["POL-VOL-V1"][0] == 3 and ranks["FAQ-VOL"][0] == 4
    assert r["route"] == "rag" and r["citations"]


def test_prompt_labels_germany(chat, fake_llm):
    fake_llm.plans["How many volunteer days do I get?"] = default_plan(search_query="volunteer days per year")
    chat(EMP_DE, "How many volunteer days do I get?")
    ranks = _ranks_in_prompt(fake_llm.answer_prompts()[-1])
    de = ranks["ADD-VOL-DE"][0]
    assert all(de < ranks[d][0] for d in ("POL-VOL-V1", "POL-VOL-V2", "FAQ-VOL", "POL-VOL-V3") if d in ranks)
    assert ranks["POL-VOL-V3"][0] > de  # V3 would not win in 2027 either -> no upcoming mention


def test_past_question_uses_reference_date(chat, fake_llm):
    q = "How many volunteer days did we get in 2025?"
    fake_llm.plans[q] = default_plan(search_query="volunteer days", reference_date="2025-12-31")
    chat(EMP_IN, q)
    prompt = fake_llm.answer_prompts()[-1]
    ranks = _ranks_in_prompt(prompt)
    assert set(ranks) & {"POL-VOL-V2", "POL-VOL-V3", "FAQ-VOL"} == set()
    assert "POL-VOL-V1" in ranks and "REFERENCE DATE for deciding which rules apply: 2025-12-31" in prompt


def test_conflicting_same_rank_documents_both_reach_the_llm(chat, fake_llm):
    q = "What is the deadline for expense claims?"
    fake_llm.plans[q] = default_plan(search_query="expense claims submitted deadline days")
    fake_llm.answer_fn = lambda u: {"status": "answered", "citations": ["S1", "S2"],
                                   "answer": "The documents conflict: 30 days [S1] or 45 days [S2]."}
    r = chat(EMP_IN, q)
    ranks = _ranks_in_prompt(fake_llm.answer_prompts()[-1])
    assert ranks["POL-EXP"][0] == ranks["HB-FIN"][0]
    assert {c["doc_id"] for c in r["citations"]} == {"POL-EXP", "HB-FIN"}
    assert "[S" not in r["answer"]  # labels are rewritten to doc IDs for the user


def test_citations_only_from_retrieved_sources(chat, fake_llm):
    q = "How many volunteer days do I get?"
    fake_llm.plans[q] = default_plan(search_query="volunteer days")
    fake_llm.answer_fn = lambda u: {"status": "answered", "answer": "3 days [S2] [S99]", "citations": ["S2", "S99", "HR-SAL"]}
    r = chat(EMP_IN, q)
    assert all(c["doc_id"] != "HR-SAL" for c in r["citations"]) and len(r["citations"]) == 1


def test_no_answer_route(chat, fake_llm):
    q = "What is the pet insurance allowance?"
    fake_llm.plans[q] = default_plan(search_query="pet insurance allowance")
    r = chat(EMP_IN, q)
    assert r["route"] == "no_answer" and r["citations"] == [] and r["tool_calls"] == []
    assert "HR team" in r["answer"]
