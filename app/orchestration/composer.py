"""LLM call #2: answer from retrieved sources (and tool results, for hybrid answers).

The LLM cites sources by label ([S3]); labels are mapped back to (doc_id, locator) in
code, so a citation can only ever point at a chunk that was actually retrieved for this
requester, with the locator recorded at ingestion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from app.data.db import Employee
from app.llm.client import LLMClient
from app.llm.prompts import ANSWER_SYSTEM
from app.retrieval.retriever import RetrievalResult

_LABEL = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


@dataclass
class Composed:
    status: str                      # answered | partial | no_answer
    answer: str
    citations: list[dict] = field(default_factory=list)


def compose_answer(llm: LLMClient, requester: Employee, question: str, retrieval: RetrievalResult,
                   tool_results: list[dict], reference: date, today: date, request_id: str) -> Composed:
    if not retrieval.sources:
        return Composed("no_answer", "")
    payload = (
        "REQUESTER (from the HR database; not personal data):\n"
        + json.dumps(requester.safe_profile())
        + f"\n\nTODAY: {today.isoformat()}\nREFERENCE DATE for deciding which rules apply: {reference.isoformat()}\n"
        + ("\nTOOL RESULTS:\n" + json.dumps(tool_results, ensure_ascii=False) + "\n" if tool_results else "")
        + "\nSOURCES:\n<<<SOURCES\n" + retrieval.context + "\nSOURCES>>>\n"
        + "\nQUESTION:\n" + question
    )
    out = llm.complete_json(ANSWER_SYSTEM, payload, corpus_tokens=retrieval.corpus_tokens,
                            purpose="answer", request_id=request_id, max_tokens=1000)

    status = out.get("status") if out.get("status") in ("answered", "partial", "no_answer") else "answered"
    answer = str(out.get("answer") or "").strip()
    by_label = retrieval.by_label()

    cited_labels: list[str] = []
    for lab in list(out.get("citations") or []) + [l.strip() for m in _LABEL.findall(answer) for l in m.split(",")]:
        lab = str(lab).strip().strip("[]")
        if lab in by_label and lab not in cited_labels:
            cited_labels.append(lab)

    citations: list[dict] = []
    for lab in cited_labels:
        src = by_label[lab]
        c = {"doc_id": src.doc["doc_id"], "locator": src.chunk["locator"]}
        if c not in citations:
            citations.append(c)

    # Replace [S#] markers with readable document IDs in the text shown to the user.
    def _sub(m: re.Match) -> str:
        ids = []
        for lab in m.group(1).split(","):
            src = by_label.get(lab.strip())
            if src and src.doc["doc_id"] not in ids:
                ids.append(src.doc["doc_id"])
        return f"[{', '.join(ids)}]" if ids else ""

    answer = re.sub(r"\s+([.,;])", r"\1", _LABEL.sub(_sub, answer)).strip()

    if status == "no_answer" or not citations:
        return Composed("no_answer", answer)
    return Composed(status, answer, citations)
