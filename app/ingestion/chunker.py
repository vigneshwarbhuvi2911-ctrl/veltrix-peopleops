"""Section-aware chunking.

One chunk = one section (heading + its paragraphs and tables). Sections longer than
MAX_CHUNK_TOKENS are split on block boundaries; long tables are split by rows with the
header row repeated, so every chunk is self-describing. Each chunk carries a locator:
the nearest section heading, or "p. N" for paged documents without headings.

Text addressed to an AI system ("Note for the AI assistant: ...") is quarantined here,
so it never reaches retrieval or the LLM (defence in depth; the answer prompt also
treats all sources as data).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.ingestion.parsers import Block, ParsedDocument, table_to_markdown
from app.retrieval.tokens import count_tokens

log = logging.getLogger(__name__)

MAX_CHUNK_TOKENS = 380

_INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\b(note|message|instructions?|reminder)\s+(for|to)\s+(the\s+|any\s+)?(ai|a\.i\.|assistant|chat\s*bot|bot|llm|language model|model)\b",
        r"\b(ai|virtual|hr)\s+assistants?\s+(reading|processing|summari[sz]ing)\b",
        r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier|your)\s+(instructions|rules|guidelines)",
        r"\b(system|developer)\s+prompt\b",
        r"\byou\s+are\s+now\s+(an?|the)\b",
        r"\bif\s+you\s+are\s+an?\s+(ai|assistant|language model|llm)\b",
    )
]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    locator: str
    heading: str
    page: int | None
    text: str
    tokens: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _split_table(rows: list[list[str]], max_tokens: int) -> list[str]:
    md = table_to_markdown(rows)
    if count_tokens(md) <= max_tokens or len(rows) <= 2:
        return [md]
    header, body = rows[0], rows[1:]
    parts, cur = [], []
    for r in body:
        trial = table_to_markdown([header] + cur + [r])
        if cur and count_tokens(trial) > max_tokens:
            parts.append(table_to_markdown([header] + cur))
            cur = []
        cur.append(r)
    if cur:
        parts.append(table_to_markdown([header] + cur))
    return parts


def chunk_document(doc_id: str, title: str, parsed: ParsedDocument) -> tuple[list[Chunk], int]:
    """Returns (chunks, number_of_quarantined_blocks)."""
    has_sections = any(b.kind == "heading" and b.level >= 1 for b in parsed.blocks)
    quarantined = 0

    # Group blocks into sections.
    sections: list[tuple[str, int | None, list[Block]]] = []
    heading = title
    heading_page: int | None = None
    current: list[Block] = []
    for b in parsed.blocks:
        if b.kind == "heading":
            if current:
                sections.append((heading, heading_page, current))
            current = []
            heading = b.text if b.level >= 1 or not has_sections else title
            heading_page = b.page
            continue
        if b.kind == "text" and looks_like_injection(b.text):
            quarantined += 1
            log.warning("ingest_quarantined_block", extra={"fields": {"doc_id": doc_id}})
            continue
        if not has_sections and current and b.page != current[-1].page:
            # Paged document without headings: keep chunks within a page so "p. N" is exact.
            sections.append((heading, heading_page, current))
            current = []
        current.append(b)
    if current:
        sections.append((heading, heading_page, current))

    chunks: list[Chunk] = []
    for heading, hpage, blocks in sections:
        first_page = next((b.page for b in blocks if b.page), hpage)
        if has_sections:
            locator = heading
        elif first_page:
            locator = f"p. {first_page}"
        else:
            locator = title
        header = f"## {heading}" if has_sections else (f"[{locator}]" if first_page else "")

        pieces: list[str] = []
        for b in blocks:
            if b.kind == "table":
                pieces.extend(_split_table(b.rows, MAX_CHUNK_TOKENS - 40))
            elif b.text:
                pieces.append(b.text)

        buf: list[str] = []
        page = first_page

        def emit():
            if not buf:
                return
            body = "\n".join(buf)
            text = f"{header}\n{body}".strip() if header else body
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks) + 1}",
                    doc_id=doc_id,
                    locator=locator,
                    heading=heading,
                    page=page,
                    text=text,
                    tokens=count_tokens(text),
                )
            )

        for piece in pieces:
            if buf and count_tokens("\n".join(buf + [piece])) > MAX_CHUNK_TOKENS:
                emit()
                buf = []
            buf.append(piece)
        emit()
    return chunks, quarantined
