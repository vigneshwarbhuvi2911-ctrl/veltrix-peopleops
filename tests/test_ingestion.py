"""Ingestion of every format: headings/locators, tables, OCR, metadata cards, injections."""
import shutil

import pytest


def chunks_of(index, doc_id):
    return [c for c in index["chunks"] if c["doc_id"] == doc_id]


def test_every_document_ingested(index):
    assert all(d["chunk_count"] > 0 for d in index["documents"].values())
    assert len(index["documents"]) == 14


@pytest.mark.parametrize("doc_id,locator", [
    ("POL-VOL-V1", "1. Volunteer days"),          # Markdown heading
    ("POL-VOL-V2", "1. Volunteer days"),          # HTML <h2>
    ("ADD-VOL-DE", "1. Volunteer days"),          # DOCX Heading 1
    ("HB-MGR", "1. Approving leave"),             # PDF bold, larger font
    ("FAQ-VOL", "How many volunteer days do I get?"),
])
def test_locator_is_nearest_section_heading(index, doc_id, locator):
    assert locator in {c["locator"] for c in chunks_of(index, doc_id)}


def test_tables_become_markdown_with_header_row(index):
    for doc_id in ("CAL-IN-2026", "CAL-US-2026", "CAL-DE-2026"):  # HTML, PDF, DOCX tables
        text = "\n".join(c["text"] for c in chunks_of(index, doc_id))
        assert "| Date | Day | Holiday |" in text and "|---|---|---|" in text


def test_metadata_card_is_not_indexed(index):
    for c in index["chunks"]:
        assert "Document ID" not in c["text"], c["chunk_id"]


def test_boilerplate_removed(index):
    text = " ".join(c["text"] for c in index["chunks"])
    assert "Intranet" not in text and "color:red" not in text


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_scanned_pdf_is_read_with_ocr(index):
    doc = index["documents"]["MEMO-WFH"]
    assert doc["ocr_pages"] == [1]
    text = " ".join(c["text"] for c in chunks_of(index, "MEMO-WFH")).lower()
    assert "3 days per week" in text
    assert {c["locator"] for c in chunks_of(index, "MEMO-WFH")} == {"p. 1"}  # no headings -> page


def test_holidays_extracted_and_restricted_skipped(index):
    assert [h["date"] for h in index["holidays"]["CAL-IN-2026"]] == ["2026-01-26", "2026-11-09", "2026-12-25"]
    assert "CAL-IN-2026" in index["holidays"] and "POL-VOL-V1" not in index["holidays"]


def test_team_mailboxes_collected_for_pii_allowlist(index):
    assert all("@" in e for e in index["corpus_emails"])


def test_missing_file_does_not_stop_ingestion(company, tmp_path):
    import json
    from app.ingestion.pipeline import build_index
    manifest = json.load(open(company["manifest"]))
    manifest.append({"doc_id": "GHOST", "file": "ghost.pdf", "title": "Ghost", "doc_type": "policy",
                     "regions": ["GLOBAL"], "effective_date": "2026-01-01", "access_level": "all"})
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    idx = build_index(company["corpus"], str(p))
    assert "GHOST" not in idx["documents"] and len(idx["documents"]) == 14
