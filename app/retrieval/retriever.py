"""Retrieval: eligibility filter -> BM25 over the remaining chunks -> diversified,
budget-aware packing -> source block with precedence labels.

Order of operations matters for Section 7.2 / must-pass check 2: documents the
requester cannot read are removed in `partition_documents` *before* scoring, so they
can never be selected, packed or sent to the LLM.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from app.retrieval.bm25 import BM25, tokenize
from app.retrieval.eligibility import READABLE_LEVELS, can_read, partition_documents, rank_documents
from app.retrieval.tokens import count_tokens

log = logging.getLogger(__name__)

MIN_RELATIVE_SCORE = 0.2    # a document is relevant if its best chunk scores >= 20% of the top chunk
FILL_RELATIVE_SCORE = 0.35  # extra (non-best) chunks must score >= 35% of the top chunk
MAX_CHUNKS = 16
DIVERSITY_DECAY = 0.7       # each extra chunk from the same document is discounted


@dataclass
class Source:
    label: str
    chunk: dict
    doc: dict
    status: str  # "IN FORCE" | "NOT YET IN FORCE"
    rank: int
    score: float

    def render(self) -> str:
        d = self.doc
        head = (
            f"[{self.label}] doc_id={d['doc_id']} | title={d['title']} | type={d['doc_type']} | "
            f"regions={','.join(d['regions'])} | effective={d['effective_date']} | status={self.status} | "
            f"precedence_rank={self.rank} | locator={self.chunk['locator']}"
        )
        return f"{head}\n{self.chunk['text']}"


@dataclass
class RetrievalResult:
    sources: list[Source] = field(default_factory=list)
    context: str = ""
    corpus_tokens: int = 0
    eligible_doc_ids: list[str] = field(default_factory=list)

    def by_label(self) -> dict[str, Source]:
        return {s.label: s for s in self.sources}


class KnowledgeBase:
    def __init__(self, index: dict):
        self.documents: dict[str, dict] = index["documents"]
        self.chunks: list[dict] = index["chunks"]
        self.holidays: dict[str, list[dict]] = index.get("holidays", {})
        self.corpus_emails: set[str] = set(index.get("corpus_emails", []))
        self._tokens = [
            tokenize(f"{self.documents[c['doc_id']]['title']} {c['text']}") for c in self.chunks
        ]
        # One BM25 model per role, built only over documents that role may read, so term
        # statistics are stable (whole readable corpus, not the per-query candidate set) and
        # never derived from documents the requester cannot access.
        self._bm25: dict[str, tuple[BM25, dict[int, int]]] = {}
        for role in READABLE_LEVELS:
            positions = [i for i, c in enumerate(self.chunks)
                         if can_read(role, self.documents[c["doc_id"]].get("access_level", ""))]
            self._bm25[role] = (BM25([self._tokens[i] for i in positions]), {i: k for k, i in enumerate(positions)})

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def retrieve(self, query: str, role: str, region: str, reference: date, include_upcoming: bool,
                 budget: int, hard_limit: int = 3000) -> RetrievalResult:
        sets = partition_documents(list(self.documents.values()), role, region, reference, include_upcoming)
        status = {d["doc_id"]: "IN FORCE" for d in sets.eligible}
        status.update({d["doc_id"]: "NOT YET IN FORCE" for d in sets.upcoming})
        idx = [i for i, c in enumerate(self.chunks) if c["doc_id"] in status]
        result = RetrievalResult(eligible_doc_ids=sorted(d["doc_id"] for d in sets.eligible))
        q = tokenize(query)
        if not idx or not q:
            return result

        model, local = self._bm25.get(role, (None, {}))
        idx = [i for i in idx if i in local]
        if model is None or not idx:
            return result
        scores = model.scores(q, [local[i] for i in idx])
        scored = {i: s for s, i in zip(scores, idx) if s > 0}
        if not scored:
            return result
        top = max(scored.values())
        pool = {i: s for i, s in scored.items() if s >= top * MIN_RELATIVE_SCORE}

        # Precedence-aware selection. FAQs and wikis are often phrased exactly like user
        # questions, so pure lexical ranking would crowd out the policy that actually
        # governs the answer. Phase 1 therefore guarantees every relevant document its
        # best chunk, strongest precedence first; phase 2 fills the remaining budget by
        # score, discounting repeated chunks from the same document.
        best_of: dict[str, int] = {}
        for i, s in pool.items():
            d = self.chunks[i]["doc_id"]
            if d not in best_of or s > pool[best_of[d]]:
                best_of[d] = i
        all_ranks = rank_documents([self.documents[d] for d in best_of], region)
        phase1 = sorted(best_of.values(), key=lambda i: (all_ranks[self.chunks[i]["doc_id"]], -pool[i], i))

        chosen: list[tuple[int, float]] = []
        per_doc: dict[str, int] = {}
        used_tokens = 0

        def take(i: int) -> bool:
            nonlocal used_tokens
            c = self.chunks[i]
            est = c["tokens"] + 70  # header line; exact count is taken below
            if used_tokens + est > budget or len(chosen) >= MAX_CHUNKS:
                return False
            chosen.append((i, pool[i]))
            used_tokens += est
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            return True

        for i in phase1:
            take(i)
        rest = {i: s for i, s in pool.items() if i not in {j for j, _ in chosen} and s >= top * FILL_RELATIVE_SCORE}
        while rest and len(chosen) < MAX_CHUNKS:
            nxt = max(rest, key=lambda i: (rest[i] * DIVERSITY_DECAY ** per_doc.get(self.chunks[i]["doc_id"], 0), -i))
            rest.pop(nxt)
            take(nxt)

        docs = [self.documents[d] for d in {self.chunks[i]["doc_id"] for i, _ in chosen}]
        ranks = rank_documents(docs, region)
        # Present strongest documents first, then by document order within a document.
        chosen.sort(key=lambda t: (ranks[self.chunks[t[0]]["doc_id"]], self.chunks[t[0]]["doc_id"], t[0]))

        sources: list[Source] = []
        for n, (i, score) in enumerate(chosen, start=1):
            c = self.chunks[i]
            sources.append(Source(f"S{n}", c, self.documents[c["doc_id"]], status[c["doc_id"]], ranks[c["doc_id"]], score))

        # Exact cl100k count of everything that will be sent; trim if ever over budget.
        while sources:
            context = "\n\n".join(s.render() for s in sources)
            tokens = count_tokens(context)
            if tokens <= min(budget, hard_limit):
                break
            sources.pop(min(range(len(sources)), key=lambda k: sources[k].score))  # drop weakest
        else:
            context, tokens = "", 0
        result.sources, result.context, result.corpus_tokens = sources, context, tokens
        return result
