"""Structured, PII-safe logging.

Policy (DESIGN.md §5): log records carry employee IDs, routes, tool names and token
counts only. Message text, answers, names, emails and phone numbers are never passed
to the logger. As a second line of defence, a filter scrubs anything that looks like an
email address or phone number from every record before it is written.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.orchestration.pii import EMAIL_RE, PHONE_RES


def scrub(text: str) -> str:
    text = EMAIL_RE.sub("[email]", text)
    for rx in PHONE_RES:
        text = rx.sub("[phone]", text)
    return text


class PiiScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg, record.args = scrub(message), None
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            record.fields = {k: (scrub(v) if isinstance(v, str) else v) for k, v in fields.items()}
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            # Exception text can contain data; keep the type only.
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: str = "INFO", log_file: str = "") -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(level.upper())
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        h.addFilter(PiiScrubFilter())
        root.addHandler(h)
    # Third-party HTTP loggers can echo request bodies at DEBUG; keep them quiet.
    for noisy in ("urllib3", "httpx", "httpcore", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(event, extra={"fields": fields})
