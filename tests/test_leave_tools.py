"""apply_leave / cancel_leave rules (Sections 6.2, 6.3) and their transactions."""
import re
import threading

from app.tools import leave_tools as T
from tests.conftest import EMP_DE, EMP_IN, EMP_IN2, EMP_US, TODAY

UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


# ----------------------------------------------------------------- apply_leave
def test_propose_does_not_write(tool_ctx, db):
    before = db.get_balances(EMP_IN)["annual"]
    n = len(db.list_requests(EMP_IN))
    r = T.apply_leave(tool_ctx(EMP_IN), "annual", "2026-10-26", "2026-10-28", execute=False)
    assert r.status == "pending_confirmation" and r.data["working_days"] == 3
    assert db.get_balances(EMP_IN)["annual"] == before and len(db.list_requests(EMP_IN)) == n


def test_execute_creates_pending_request_and_subtracts_balance(tool_ctx, db):
    before = db.get_balances(EMP_IN)["annual"]
    r = T.apply_leave(tool_ctx(EMP_IN), "annual", "2026-10-26", "2026-10-28", execute=True)
    assert r.status == "success"
    req = db.get_request(r.data["request_id"])
    assert UUID4.match(req.request_id)
    assert (req.status, req.working_days, req.employee_id) == ("pending", 3, EMP_IN)
    assert db.get_balances(EMP_IN)["annual"] == before - 3


def test_holidays_from_region_calendar_are_excluded(tool_ctx):
    # IN: Mon 9 Nov is a public holiday, Tue 10 Nov only a restricted holiday.
    r = T.apply_leave(tool_ctx(EMP_IN), "sick", "2026-11-09", "2026-11-13", execute=False)
    assert r.data["working_days"] == 4 and r.data["holidays_in_range"] == ["2026-11-09"]
    # DE: 24, 25 and 31 Dec are closures.
    r = T.apply_leave(tool_ctx(EMP_DE), "annual", "2026-12-21", "2026-12-31", execute=False)
    assert r.data["working_days"] == 6


def test_leave_type_must_be_annual_or_sick(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "vacation", "2026-10-26", "2026-10-26", execute=False)
    assert r.status == "error" and "annual" in r.error


def test_dates_must_be_in_2026(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "annual", "2026-12-30", "2027-01-04", execute=False)
    assert r.status == "error" and "2026" in r.error


def test_start_must_not_be_in_the_past(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "annual", "2026-10-09", "2026-10-14", execute=False)
    assert r.status == "error" and "past" in r.error


def test_today_is_allowed_as_start(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_US), "sick", TODAY.isoformat(), TODAY.isoformat(), execute=False)
    assert r.status == "pending_confirmation"


def test_end_before_start_rejected(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "annual", "2026-10-28", "2026-10-26", execute=False)
    assert r.status == "error" and "end date" in r.error


def test_zero_working_days_rejected(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "annual", "2026-10-24", "2026-10-25", execute=False)  # Sat-Sun
    assert r.status == "error" and "no working days" in r.error


def test_only_holiday_is_zero_working_days(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "annual", "2026-11-09", "2026-11-09", execute=False)
    assert r.status == "error" and "no working days" in r.error


def test_balance_exceeded_rejected(tool_ctx):
    # EMP_IN2 has 2 annual days left.
    r = T.apply_leave(tool_ctx(EMP_IN2), "annual", "2026-10-26", "2026-10-28", execute=False)
    assert r.status == "error" and "balance is only 2" in r.error


def test_exactly_the_balance_is_allowed(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN2), "annual", "2026-10-26", "2026-10-27", execute=False)
    assert r.status == "pending_confirmation"


def test_overlap_with_pending_or_approved_rejected(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "sick", "2026-11-20", "2026-11-23", execute=False)  # pending 16-20 Nov
    assert r.status == "error" and "overlap" in r.error and "11111111" in r.error


def test_overlap_with_cancelled_request_is_fine(tool_ctx):
    r = T.apply_leave(tool_ctx(EMP_IN), "sick", "2026-12-01", "2026-12-01", execute=False)  # cancelled one
    assert r.status == "pending_confirmation"


def test_execute_rechecks_inside_transaction(tool_ctx, db):
    """Data can change between proposal and confirmation; execution re-validates."""
    ctx = tool_ctx(EMP_IN2)
    assert T.apply_leave(ctx, "annual", "2026-10-26", "2026-10-27", execute=False).status == "pending_confirmation"
    assert T.apply_leave(ctx, "annual", "2026-10-29", "2026-10-30", execute=True).status == "success"
    r = T.apply_leave(ctx, "annual", "2026-10-26", "2026-10-27", execute=True)
    assert r.status == "error" and "balance" in r.error
    assert db.get_balances(EMP_IN2)["annual"] == 0


def test_concurrent_requests_cannot_double_spend(tool_ctx, db):
    ctx = tool_ctx(EMP_IN2)  # 2 days of balance
    results = []

    def go(day):
        results.append(T.apply_leave(ctx, "annual", day, day, execute=True).status)

    # Four single-day requests on different days racing for a 2-day balance.
    threads = [threading.Thread(target=go, args=(d,)) for d in ("2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("success") == 2 and db.get_balances(EMP_IN2)["annual"] == 0


# ---------------------------------------------------------------- cancel_leave
def test_cancel_refunds_balance(tool_ctx, db):
    before = db.get_balances(EMP_IN)["annual"]
    r = T.cancel_leave(tool_ctx(EMP_IN), "11111111-1111-4111-8111-111111111111", execute=True)
    assert r.status == "success"
    assert db.get_request("11111111-1111-4111-8111-111111111111").status == "cancelled"
    assert db.get_balances(EMP_IN)["annual"] == before + 5


def test_cancel_proposal_does_not_write(tool_ctx, db):
    r = T.cancel_leave(tool_ctx(EMP_IN), "11111111-1111-4111-8111-111111111111", execute=False)
    assert r.status == "pending_confirmation"
    assert db.get_request("11111111-1111-4111-8111-111111111111").status == "pending"


def test_cannot_cancel_someone_elses_request(tool_ctx, db):
    own = T.cancel_leave(tool_ctx(EMP_IN), "55555555-5555-4555-8555-555555555555", execute=True)  # EMP_IN2's
    missing = T.cancel_leave(tool_ctx(EMP_IN), "00000000-0000-4000-8000-000000000000", execute=True)
    assert own.status == missing.status == "error"
    assert own.error == missing.error  # does not reveal that the other request exists
    assert db.get_request("55555555-5555-4555-8555-555555555555").status == "approved"


def test_cannot_cancel_already_cancelled(tool_ctx):
    r = T.cancel_leave(tool_ctx(EMP_IN), "33333333-3333-4333-8333-333333333333", execute=False)
    assert r.status == "error" and "already cancelled" in r.error


def test_cannot_cancel_past_or_started_requests(tool_ctx):
    past = T.cancel_leave(tool_ctx(EMP_IN), "22222222-2222-4222-8222-222222222222", execute=False)
    today = T.cancel_leave(tool_ctx(EMP_IN), "44444444-4444-4444-8444-444444444444", execute=False)  # starts today
    assert past.status == today.status == "error"
    assert "after today" in today.error


def test_cancel_allowed_for_approved_future_request(tool_ctx):
    r = T.cancel_leave(tool_ctx(EMP_IN), "77777777-7777-4777-8777-777777777777", execute=False)
    assert r.status == "pending_confirmation"
