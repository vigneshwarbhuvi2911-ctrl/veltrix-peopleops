"""Personal-data protection for answers (Section 7.4).

Answers are built so that personal data rarely gets near them (the LLM never receives
names, emails or phones from the database). This module is the last line of defence on
every outgoing answer: it masks email addresses, removes phone numbers and blanks the
exact phone numbers / emails of every employee in the database.
"""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
# Phone shapes: international (+CC ...), US (512) 555-0182 / 512-555-0182, Indian 98221 51875.
# Deliberately does not match ISO dates (2026-11-09) or UUIDs.
PHONE_RES = [
    re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.-]?\(?\d{1,5}\)?){2,5}(?![\w-])"),
    re.compile(r"(?<![\w-])\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w-])"),
    re.compile(r"(?<![\w-])\d{5}\s\d{5}(?![\w-])"),
    re.compile(r"(?<![\w-])[6-9]\d{9}(?![\w-])"),
]
PHONE_PLACEHOLDER = "[phone number withheld]"


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}" if domain else email


def remove_phones(text: str) -> str:
    for rx in PHONE_RES:
        text = rx.sub(PHONE_PLACEHOLDER, text)
    return text


class PiiGuard:
    def __init__(self, employee_emails: list[str], employee_phones: list[str], public_emails: set[str]):
        self.public_emails = {e.lower() for e in public_emails}  # team mailboxes found in the corpus
        self.employee_emails = {e.lower() for e in employee_emails}
        self._phone_res = []
        for p in employee_phones:
            digits = re.sub(r"\D", "", p)
            if len(digits) >= 7:
                self._phone_res.append(re.compile(r"[\s().+-]*".join(re.escape(d) for d in digits)))
                # also the national number without the country code
                if p.strip().startswith("+") and len(digits) > 10:
                    tail = digits[-10:]
                    self._phone_res.append(re.compile(r"(?<!\d)" + r"[\s().-]*".join(re.escape(d) for d in tail) + r"(?!\d)"))

    def sanitize(self, text: str) -> str:
        if not text:
            return text
        for rx in self._phone_res:
            text = rx.sub(PHONE_PLACEHOLDER, text)
        text = remove_phones(text)

        def _mask(m: re.Match) -> str:
            email = m.group(0)
            if email.lower() in self.public_emails and email.lower() not in self.employee_emails:
                return email  # e.g. a team mailbox printed in a policy, not personal data
            return mask_email(email)

        return EMAIL_RE.sub(_mask, text)
