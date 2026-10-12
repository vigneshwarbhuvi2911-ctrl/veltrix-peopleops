"""Ingestion pipeline: manifest + corpus -> index (chunks, document metadata, holidays).

Runs automatically at service start-up (app/main.py) and is idempotent: the index is
rebuilt from whatever is in CORPUS_DIR / MANIFEST_PATH, so a changed corpus is always
reflected. Nothing from the corpus is hard-coded anywhere in the code base.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.holidays import extract_holidays
from app.ingestion.parsers import parse_file

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("doc_id", "file", "title", "doc_type", "regions", "effective_date", "access_level")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for entry in data:
        missing = [k for k in REQUIRED_FIELDS if k not in entry]
        if missing:
            log.error("manifest_entry_invalid", extra={"fields": {"doc_id": entry.get("doc_id"), "missing": missing}})
            continue
        entry = dict(entry)
        entry["regions"] = [r.upper() for r in entry["regions"]]
        entry["doc_type"] = entry["doc_type"].lower()
        entry["access_level"] = entry["access_level"].lower()
        docs.append(entry)
    return docs


def build_index(corpus_dir: str, manifest_path: str, ocr_lang: str = "eng") -> dict:
    t0 = time.time()
    docs = load_manifest(manifest_path)
    corpus = Path(corpus_dir)
    listed = {d["file"] for d in docs}
    for f in sorted(p.name for p in corpus.iterdir() if p.is_file() and not p.name.startswith(".")):
        if f not in listed:
            log.warning("corpus_file_not_in_manifest_skipped", extra={"fields": {"file": f}})

    index = {"documents": {}, "chunks": [], "holidays": {}, "corpus_emails": []}
    emails: set[str] = set()
    for meta in docs:
        path = corpus / meta["file"]
        if not path.exists():
            log.error("corpus_file_missing", extra={"fields": {"doc_id": meta["doc_id"], "file": meta["file"]}})
            continue
        try:
            parsed = parse_file(path, ocr_lang=ocr_lang)
        except Exception as exc:  # one bad file must not stop ingestion
            log.error("parse_failed", extra={"fields": {"doc_id": meta["doc_id"], "error": type(exc).__name__}})
            continue
        chunks, quarantined = chunk_document(meta["doc_id"], meta["title"], parsed)
        holidays = extract_holidays(parsed.blocks) if meta["doc_type"] == "calendar" else []
        for c in chunks:
            emails.update(e.lower() for e in _EMAIL_RE.findall(c.text))
        index["documents"][meta["doc_id"]] = {
            **meta,
            "chunk_count": len(chunks),
            "ocr_pages": parsed.ocr_pages,
            "quarantined_blocks": quarantined,
        }
        index["chunks"].extend(c.to_dict() for c in chunks)
        if holidays:
            index["holidays"][meta["doc_id"]] = holidays
        log.info(
            "ingested_document",
            extra={"fields": {
                "doc_id": meta["doc_id"], "chunks": len(chunks), "ocr_pages": parsed.ocr_pages,
                "tables": sum(1 for b in parsed.blocks if b.kind == "table"),
                "holidays": len(holidays), "quarantined_blocks": quarantined,
            }},
        )
    index["corpus_emails"] = sorted(emails)
    log.info("ingestion_complete", extra={"fields": {
        "documents": len(index["documents"]), "chunks": len(index["chunks"]),
        "seconds": round(time.time() - t0, 2)}})
    return index


def save_index(index: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp, path)


if __name__ == "__main__":  # `python -m app.ingestion.pipeline` for manual inspection
    from app.config import get_settings
    from app.logging_setup import configure_logging

    s = get_settings()
    configure_logging(s.log_level)
    save_index(build_index(s.corpus_dir, s.manifest_path, s.ocr_lang), s.index_path)
