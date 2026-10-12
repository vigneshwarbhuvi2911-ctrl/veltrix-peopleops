"""Extract public-holiday dates from calendar documents at ingestion time.

Reads table rows first (any column whose cells parse as dates). If a calendar has no
usable table (for example an OCR'd scan), falls back to scanning text lines. Tables or
sections labelled restricted / optional / floating are skipped, because the office is
open on those days and they are therefore not public holidays for working-day counts.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from app.ingestion.parsers import Block

_SKIP_SECTION = re.compile(r"\b(restricted|optional|floating|elective)\b", re.I)
_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%Y-%m-%d", "%d.%m.%Y", "%A, %d %B %Y", "%A %d %B %Y")
_DATE_IN_TEXT = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2})\b"
)


def parse_date(text: str) -> date | None:
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip()).rstrip(".")
    t = re.sub(r"\bSept\b", "Sep", t)
    t = re.sub(r"\s+", " ", t.replace(".,", ","))
    for fmt in _FORMATS:
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def extract_holidays(blocks: list[Block]) -> list[dict]:
    found: dict[str, str] = {}
    heading = ""
    for b in blocks:
        if b.kind == "heading":
            heading = b.text
            continue
        if b.kind != "table" or not b.rows:
            continue
        header = " ".join(b.rows[0])
        if _SKIP_SECTION.search(heading) or _SKIP_SECTION.search(header):
            continue
        for row in b.rows:
            dates = [(i, parse_date(c)) for i, c in enumerate(row)]
            dates = [(i, d) for i, d in dates if d]
            if not dates:
                continue
            idx, d = dates[0]
            names = [c for i, c in enumerate(row) if i != idx and c and not parse_date(c)
                     and c.lower() not in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")]
            found.setdefault(d.isoformat(), names[-1] if names else "")

    if not found:  # text fallback, e.g. a scanned calendar
        heading = ""
        for b in blocks:
            if b.kind == "heading":
                heading = b.text
                continue
            if b.kind != "text":
                continue
            for line in b.text.splitlines():
                if _SKIP_SECTION.search(line):
                    heading = line  # a line introducing a restricted/optional list
                    continue
                if _SKIP_SECTION.search(heading):
                    continue
                for m in _DATE_IN_TEXT.finditer(line):
                    d = parse_date(m.group(1))
                    if d:
                        rest = line[m.end():].strip(" -:|")
                        found.setdefault(d.isoformat(), rest[:60])
    return [{"date": k, "name": v} for k, v in sorted(found.items())]
