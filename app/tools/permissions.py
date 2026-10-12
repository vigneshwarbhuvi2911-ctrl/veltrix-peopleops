"""Access control for leave data (Sections 3.1, 6.1 and 7.3).

These checks take the requester as loaded from the database using the X-Employee-Id
header. They never look at the prompt or at anything the LLM produced, other than the
target employee ID being asked about.
"""
from __future__ import annotations

from app.data.db import Database, Employee

# Same message whether the target exists or not, so access checks cannot be used to
# discover which employee IDs exist (Section 7.3).
NOT_PERMITTED_EMPLOYEE = "I can't share information about that employee."
MANAGERS_ONLY = "Only managers can view team leave, and only for their own direct reports."
BULK_REFUSAL = (
    "I can't share leave data for many employees at once. I can help with one employee you are "
    "allowed to see at a time, or, if you are a manager, show your direct reports' leave for a date range."
)


class PermissionDenied(Exception):
    def __init__(self, message: str = NOT_PERMITTED_EMPLOYEE):
        super().__init__(message)
        self.message = message


def can_view_leave_of(requester: Employee, target_id: str, db: Database) -> bool:
    """employee: self only; manager: self + direct reports; hr: any one employee."""
    if not target_id:
        return False
    if target_id == requester.employee_id:
        return True
    target = db.get_employee(target_id)
    if target is None:
        return False
    if requester.role == "hr":
        return True
    if requester.role == "manager" and target.manager_id == requester.employee_id:
        return True
    return False


def require_view(requester: Employee, target_id: str, db: Database) -> None:
    if not can_view_leave_of(requester, target_id, db):
        raise PermissionDenied(NOT_PERMITTED_EMPLOYEE)


def require_manager(requester: Employee) -> None:
    if requester.role != "manager":
        raise PermissionDenied(MANAGERS_ONLY)


def visible_employees(requester: Employee, db: Database) -> list[Employee]:
    """Employees whose names the requester may see (used to resolve 'Priya's balance')."""
    if requester.role == "hr":
        return db.all_employees()
    people = [requester]
    if requester.role == "manager":
        people += db.direct_reports(requester.employee_id)
    return people
