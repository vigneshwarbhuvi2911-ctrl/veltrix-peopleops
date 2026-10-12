"""Working-day arithmetic (Section 6.2): Monday-Friday, minus public holidays."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def working_dates(start: date, end: date, holidays: set[str]) -> list[date]:
    return [d for d in daterange(start, end) if d.weekday() < 5 and d.isoformat() not in holidays]


def count_working_days(start: date, end: date, holidays: set[str]) -> int:
    if end < start:
        return 0
    return len(working_dates(start, end, holidays))
