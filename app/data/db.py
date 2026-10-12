"""Data access layer for veltrix.db.

Only this module issues SQL against the HR database. It knows nothing about roles or
permissions; those live in app/tools/permissions.py. The existing tables and columns
are never altered.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

ACTIVE_STATUSES = ("pending", "approved")


@dataclass(frozen=True)
class Employee:
    employee_id: str
    full_name: str
    email: str
    phone: str
    region: str
    join_date: str
    level: str
    role: str
    manager_id: str | None

    def safe_profile(self) -> dict:
        """Fields that are not personal data (Section 7.4) and may go to the LLM."""
        return {
            "employee_id": self.employee_id,
            "region": self.region,
            "role": self.role,
            "level": self.level,
            "join_date": self.join_date,
            "manager_id": self.manager_id,
        }


@dataclass(frozen=True)
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str
    start_date: str
    end_date: str
    working_days: int
    status: str
    created_at: str

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "employee_id": self.employee_id,
            "leave_type": self.leave_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "working_days": self.working_days,
            "status": self.status,
        }


class Database:
    def __init__(self, path: str):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------- employees
    def get_employee(self, employee_id: str) -> Employee | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
        return Employee(**dict(row)) if row else None

    def direct_reports(self, manager_id: str) -> list[Employee]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT * FROM employees WHERE manager_id = ? ORDER BY employee_id", (manager_id,)
            ).fetchall()
        return [Employee(**dict(r)) for r in rows]

    def all_employees(self) -> list[Employee]:
        with self.connect() as c:
            rows = c.execute("SELECT * FROM employees ORDER BY employee_id").fetchall()
        return [Employee(**dict(r)) for r in rows]

    # ------------------------------------------------------------------- leave
    def get_balances(self, employee_id: str, leave_type: str | None = None) -> dict[str, float]:
        sql = "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ?"
        args: list = [employee_id]
        if leave_type:
            sql += " AND leave_type = ?"
            args.append(leave_type)
        with self.connect() as c:
            rows = c.execute(sql + " ORDER BY leave_type", args).fetchall()
        return {r["leave_type"]: float(r["balance_days"]) for r in rows}

    def list_requests(self, employee_id: str, status: str | None = None) -> list[LeaveRequest]:
        sql = "SELECT * FROM leave_requests WHERE employee_id = ?"
        args: list = [employee_id]
        if status:
            sql += " AND status = ?"
            args.append(status)
        with self.connect() as c:
            rows = c.execute(sql + " ORDER BY start_date, request_id", args).fetchall()
        return [LeaveRequest(**dict(r)) for r in rows]

    def get_request(self, request_id: str) -> LeaveRequest | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM leave_requests WHERE request_id = ?", (request_id,)).fetchone()
        return LeaveRequest(**dict(row)) if row else None

    def team_leave(self, manager_id: str, start: str, end: str) -> list[LeaveRequest]:
        with self.connect() as c:
            rows = c.execute(
                """
                SELECT r.* FROM leave_requests r
                JOIN employees e ON e.employee_id = r.employee_id
                WHERE e.manager_id = ?
                  AND r.status IN ('pending', 'approved')
                  AND r.start_date <= ? AND r.end_date >= ?
                ORDER BY r.start_date, r.employee_id
                """,
                (manager_id, end, start),
            ).fetchall()
        return [LeaveRequest(**dict(r)) for r in rows]

    # ------------------------------------------------------------------ writes
    def create_request(
        self, employee_id: str, leave_type: str, start: str, end: str, working_days: int
    ) -> tuple[LeaveRequest | None, str | None]:
        """Re-checks overlap and balance and writes atomically.

        Returns (request, None) on success or (None, reason_code) if the data changed
        since validation. BEGIN IMMEDIATE takes the write lock up front, so two
        concurrent requests cannot both pass the checks.
        """
        request_id = str(uuid.uuid4())
        created_at = datetime.now().replace(microsecond=0).isoformat()
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                overlap = c.execute(
                    """SELECT 1 FROM leave_requests WHERE employee_id = ?
                       AND status IN ('pending','approved') AND start_date <= ? AND end_date >= ?""",
                    (employee_id, end, start),
                ).fetchone()
                if overlap:
                    c.execute("ROLLBACK")
                    return None, "overlap"
                bal = c.execute(
                    "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
                    (employee_id, leave_type),
                ).fetchone()
                if bal is None or float(bal["balance_days"]) < working_days:
                    c.execute("ROLLBACK")
                    return None, "insufficient_balance"
                c.execute(
                    """INSERT INTO leave_requests
                       (request_id, employee_id, leave_type, start_date, end_date, working_days, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                    (request_id, employee_id, leave_type, start, end, working_days, created_at),
                )
                c.execute(
                    """UPDATE leave_balances SET balance_days = balance_days - ?
                       WHERE employee_id = ? AND leave_type = ?""",
                    (working_days, employee_id, leave_type),
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return (
            LeaveRequest(request_id, employee_id, leave_type, start, end, working_days, "pending", created_at),
            None,
        )

    def cancel_request(self, request_id: str, employee_id: str, today_iso: str) -> tuple[LeaveRequest | None, str | None]:
        """Atomically cancels and refunds. Re-checks ownership, status and start date."""
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute(
                    "SELECT * FROM leave_requests WHERE request_id = ? AND employee_id = ?",
                    (request_id, employee_id),
                ).fetchone()
                if row is None:
                    c.execute("ROLLBACK")
                    return None, "not_found"
                req = LeaveRequest(**dict(row))
                if req.status not in ACTIVE_STATUSES:
                    c.execute("ROLLBACK")
                    return req, "bad_status"
                if not req.start_date > today_iso:
                    c.execute("ROLLBACK")
                    return req, "already_started"
                c.execute("UPDATE leave_requests SET status = 'cancelled' WHERE request_id = ?", (request_id,))
                c.execute(
                    """UPDATE leave_balances SET balance_days = balance_days + ?
                       WHERE employee_id = ? AND leave_type = ?""",
                    (req.working_days, employee_id, req.leave_type),
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return LeaveRequest(**{**req.__dict__, "status": "cancelled"}), None
