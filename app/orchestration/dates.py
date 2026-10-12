"""Date helpers given to the planner so relative phrases resolve consistently."""
from __future__ import annotations

from datetime import date, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def helper_calendar(today: date, days: int = 45) -> dict:
    monday = today - timedelta(days=today.weekday())
    next_monday = monday + timedelta(days=7)
    nxt = {}
    for i in range(1, 8):
        d = today + timedelta(days=i)
        nxt[f"next {WEEKDAYS[d.weekday()]}"] = d.isoformat()
    return {
        "today": f"{today.isoformat()} ({WEEKDAYS[today.weekday()]})",
        "tomorrow": (today + timedelta(days=1)).isoformat(),
        "this_week_mon_fri": [monday.isoformat(), (monday + timedelta(days=4)).isoformat()],
        "next_week_mon_fri": [next_monday.isoformat(), (next_monday + timedelta(days=4)).isoformat()],
        "first_occurrence_after_today": nxt,
        "upcoming_days": [f"{(today + timedelta(days=i)).isoformat()} {WEEKDAYS[(today + timedelta(days=i)).weekday()][:3]}"
                          for i in range(0, days)],
    }


def describe(d: str) -> str:
    x = date.fromisoformat(d)
    return f"{WEEKDAYS[x.weekday()][:3]} {x.isoformat()}"
