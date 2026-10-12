"""The five leave tools (Section 6.1).

Every tool takes a ToolContext built from the X-Employee-Id header, and enforces its own
permission rules before touching data. Tool *arguments* may come from the LLM's plan,
but the identity used for authorisation never does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from app.data.db import ACTIVE_STATUSES, Database, Employee
from app.tools.permissions import PermissionDenied, require_manager, require_view
from app.tools.working_days import count_working_days

LEAVE_TYPES = ("annual", "sick")
REQUEST_STATUSES = ("pending", "approved", "cancelled")
LEAVE_YEAR = 2026

TOOL_NAMES = ("get_leave_balance", "list_leave_requests", "get_team_leave", "apply_leave", "cancel_leave")


@dataclass
class ToolContext:
    requester: Employee
    db: Database
    today: date
    holidays_for: Callable[[Employee, int], tuple[set[str], list[str]]]


@dataclass
class ToolResult:
    name: str
    args: dict
    status: str  # "success" | "error" | "pending_confirmation"
    data: dict = field(default_factory=dict)
    error: str | None = None

    def public(self) -> dict:
        """Shape used in the API response's tool_calls list."""
        return {"name": self.name, "args": self.args, "status": self.status}


class ToolError(Exception):
    pass


def fmt_days(x: float) -> str:
    return f"{x:g}"


def _parse(d: str | None, label: str) -> date:
    try:
        return date.fromisoformat(str(d))
    except (TypeError, ValueError):
        raise ToolError(f"The {label} '{d}' is not a valid date (expected YYYY-MM-DD).")


# ------------------------------------------------------------------ read tools
def get_leave_balance(ctx: ToolContext, employee_id: str, leave_type: str | None = None) -> ToolResult:
    args = {"employee_id": employee_id} | ({"leave_type": leave_type} if leave_type else {})
    try:
        require_view(ctx.requester, employee_id, ctx.db)
        if leave_type and leave_type not in LEAVE_TYPES:
            raise ToolError("Leave type must be 'annual' or 'sick'.")
        balances = ctx.db.get_balances(employee_id, leave_type)
        return ToolResult("get_leave_balance", args, "success", {"employee_id": employee_id, "balances": balances})
    except (PermissionDenied, ToolError) as e:
        return ToolResult("get_leave_balance", args, "error", error=getattr(e, "message", str(e)))


def list_leave_requests(ctx: ToolContext, employee_id: str, status: str | None = None) -> ToolResult:
    args = {"employee_id": employee_id} | ({"status": status} if status else {})
    try:
        require_view(ctx.requester, employee_id, ctx.db)
        if status and status not in REQUEST_STATUSES:
            raise ToolError("Status must be 'pending', 'approved' or 'cancelled'.")
        reqs = [r.as_dict() for r in ctx.db.list_requests(employee_id, status)]
        return ToolResult("list_leave_requests", args, "success", {"employee_id": employee_id, "requests": reqs})
    except (PermissionDenied, ToolError) as e:
        return ToolResult("list_leave_requests", args, "error", error=getattr(e, "message", str(e)))


def get_team_leave(ctx: ToolContext, start_date: str, end_date: str) -> ToolResult:
    args = {"start_date": start_date, "end_date": end_date}
    try:
        require_manager(ctx.requester)
        s, e = _parse(start_date, "start date"), _parse(end_date, "end date")
        if e < s:
            raise ToolError("The end date must be on or after the start date.")
        reqs = [r.as_dict() for r in ctx.db.team_leave(ctx.requester.employee_id, s.isoformat(), e.isoformat())]
        reports = [r.employee_id for r in ctx.db.direct_reports(ctx.requester.employee_id)]
        return ToolResult("get_team_leave", args, "success", {"requests": reqs, "direct_reports": reports})
    except (PermissionDenied, ToolError) as e:
        return ToolResult("get_team_leave", args, "error", error=getattr(e, "message", str(e)))


# ------------------------------------------------------------------ write tools
def validate_apply(ctx: ToolContext, leave_type: str | None, start_date: str | None, end_date: str | None) -> dict:
    """Checks every rule in Section 6.2 without writing. Returns details or raises ToolError."""
    if leave_type not in LEAVE_TYPES:
        raise ToolError("Leave type must be 'annual' or 'sick'.")
    s, e = _parse(start_date, "start date"), _parse(end_date, "end date")
    if s.year != LEAVE_YEAR or e.year != LEAVE_YEAR:
        raise ToolError(f"Leave can only be booked within the {LEAVE_YEAR} calendar year.")
    if s < ctx.today:
        raise ToolError(f"The start date {s.isoformat()} is in the past. Leave must start on or after today ({ctx.today.isoformat()}).")
    if e < s:
        raise ToolError("The end date must be on or after the start date.")
    holidays, calendars = ctx.holidays_for(ctx.requester, LEAVE_YEAR)
    wd = count_working_days(s, e, holidays)
    if wd == 0:
        raise ToolError(
            f"{s.isoformat()} to {e.isoformat()} contains no working days (only weekends or public holidays), so there is nothing to book."
        )
    balance = ctx.db.get_balances(ctx.requester.employee_id, leave_type).get(leave_type, 0.0)
    if wd > balance:
        raise ToolError(
            f"This request needs {wd} working day(s) of {leave_type} leave, but your {leave_type} balance is only {fmt_days(balance)} day(s)."
        )
    for r in ctx.db.list_requests(ctx.requester.employee_id):
        if r.status in ACTIVE_STATUSES and r.start_date <= e.isoformat() and r.end_date >= s.isoformat():
            raise ToolError(
                f"These dates overlap your existing {r.status} {r.leave_type} leave request {r.request_id} "
                f"({r.start_date} to {r.end_date})."
            )
    return {
        "leave_type": leave_type, "start_date": s.isoformat(), "end_date": e.isoformat(),
        "working_days": wd, "balance_before": balance, "holiday_calendars": calendars,
        "holidays_in_range": sorted(h for h in holidays if s.isoformat() <= h <= e.isoformat()),
    }


def apply_leave(ctx: ToolContext, leave_type: str, start_date: str, end_date: str, execute: bool) -> ToolResult:
    """execute=False -> proposal (status pending_confirmation, nothing written)."""
    args = {"leave_type": leave_type, "start_date": start_date, "end_date": end_date}
    try:
        details = validate_apply(ctx, leave_type, start_date, end_date)
    except ToolError as e:
        return ToolResult("apply_leave", args, "error", error=str(e))
    if not execute:
        return ToolResult("apply_leave", args, "pending_confirmation", details)
    req, reason = ctx.db.create_request(
        ctx.requester.employee_id, leave_type, details["start_date"], details["end_date"], details["working_days"]
    )
    if req is None:
        msg = {
            "overlap": "These dates now overlap another of your pending or approved requests.",
            "insufficient_balance": f"Your {leave_type} balance is no longer enough for {details['working_days']} working day(s).",
        }.get(reason or "", "The request could not be created.")
        return ToolResult("apply_leave", args, "error", error=msg)
    new_balance = ctx.db.get_balances(ctx.requester.employee_id, leave_type).get(leave_type, 0.0)
    return ToolResult("apply_leave", args, "success", {**details, "request_id": req.request_id,
                                                      "status": req.status, "balance_after": new_balance})


def validate_cancel(ctx: ToolContext, request_id: str | None) -> dict:
    req = ctx.db.get_request(request_id) if request_id else None
    if req is None or req.employee_id != ctx.requester.employee_id:
        # Same message for "doesn't exist" and "belongs to someone else".
        raise ToolError("I couldn't find a leave request with that ID among your own requests.")
    if req.status not in ACTIVE_STATUSES:
        raise ToolError(f"Request {req.request_id} is already {req.status}, so it can't be cancelled.")
    if not req.start_date > ctx.today.isoformat():
        raise ToolError(
            f"Request {req.request_id} starts on {req.start_date}. Only requests that start after today "
            f"({ctx.today.isoformat()}) can be cancelled."
        )
    return req.as_dict()


def cancel_leave(ctx: ToolContext, request_id: str, execute: bool) -> ToolResult:
    args = {"request_id": request_id}
    try:
        details = validate_cancel(ctx, request_id)
    except ToolError as e:
        return ToolResult("cancel_leave", args, "error", error=str(e))
    if not execute:
        return ToolResult("cancel_leave", args, "pending_confirmation", details)
    req, reason = ctx.db.cancel_request(request_id, ctx.requester.employee_id, ctx.today.isoformat())
    if reason:
        msg = {
            "not_found": "I couldn't find a leave request with that ID among your own requests.",
            "bad_status": f"Request {request_id} is no longer pending or approved.",
            "already_started": f"Request {request_id} now starts on or before today and can't be cancelled.",
        }[reason]
        return ToolResult("cancel_leave", args, "error", error=msg)
    balance = ctx.db.get_balances(ctx.requester.employee_id, req.leave_type).get(req.leave_type, 0.0)
    return ToolResult("cancel_leave", args, "success", {**req.as_dict(), "balance_after": balance})
