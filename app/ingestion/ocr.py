"""OCR for PDF pages without a text layer (Tesseract via pytesseract)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def ocr_pdf_page(page, lang: str = "eng", resolution: int = 300) -> str:
    """Render a pdfplumber page to an image and run Tesseract on it."""
    try:
        import pytesseract
    except ImportError:  # pragma: no cover - the Docker image always has it
        log.warning("ocr_unavailable")
        return ""
    image = page.to_image(resolution=resolution).original.convert("L")
    try:
        # psm 4: a single column of text of variable sizes, which suits memos and letters.
        return pytesseract.image_to_string(image, lang=lang, config="--psm 4")
    except pytesseract.TesseractNotFoundError:
        log.error("ocr_tesseract_missing")
        return ""
