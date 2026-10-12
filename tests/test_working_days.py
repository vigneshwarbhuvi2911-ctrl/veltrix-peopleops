"""Working-day calculation: weekends, holidays, spans, and calendars read at ingestion."""
from datetime import date

from app.retrieval.eligibility import holiday_dates_for
from app.tools.working_days import count_working_days
from tests.conftest import TODAY


def test_full_week_has_five_working_days():
    assert count_working_days(date(2026, 11, 16), date(2026, 11, 22), set()) == 5


def test_weekend_only_range_is_zero():
    assert count_working_days(date(2026, 11, 21), date(2026, 11, 22), set()) == 0


def test_single_day():
    assert count_working_days(date(2026, 11, 18), date(2026, 11, 18), set()) == 1


def test_holiday_inside_range_not_counted():
    assert count_working_days(date(2026, 11, 9), date(2026, 11, 13), {"2026-11-09"}) == 4


def test_holiday_on_weekend_not_subtracted_twice():
    # Sun 2026-11-01 is a holiday but already a weekend day.
    assert count_working_days(date(2026, 10, 31), date(2026, 11, 6), {"2026-11-01"}) == 5


def test_range_spanning_weekends_and_several_holidays():
    hol = {"2026-12-24", "2026-12-25", "2026-12-31"}
    # Thu 17 .. Thu 31 Dec: 11 weekdays minus 3 holidays
    assert count_working_days(date(2026, 12, 17), date(2026, 12, 31), hol) == 8


def test_end_before_start_is_zero():
    assert count_working_days(date(2026, 11, 20), date(2026, 11, 16), set()) == 0


def test_region_calendars_come_from_ingested_documents(kb):
    hol_in, docs_in = holiday_dates_for("IN", "employee", 2026, TODAY, kb.documents, kb.holidays)
    hol_us, docs_us = holiday_dates_for("US", "employee", 2026, TODAY, kb.documents, kb.holidays)
    hol_de, docs_de = holiday_dates_for("DE", "employee", 2026, TODAY, kb.documents, kb.holidays)
    assert docs_in == ["CAL-IN-2026"] and docs_us == ["CAL-US-2026"] and docs_de == ["CAL-DE-2026"]
    assert hol_in == {"2026-01-26", "2026-11-09", "2026-12-25"}                      # HTML table
    assert hol_us == {"2026-11-26", "2026-12-24", "2026-12-25"}                      # PDF table
    assert hol_de == {"2026-10-03", "2026-12-24", "2026-12-25", "2026-12-31"}        # DOCX table


def test_restricted_holidays_are_not_office_closures(kb):
    hol_in, _ = holiday_dates_for("IN", "employee", 2026, TODAY, kb.documents, kb.holidays)
    assert "2026-11-10" not in hol_in


def test_same_range_differs_by_region(kb):
    s, e = date(2026, 12, 21), date(2026, 12, 31)
    days = {r: count_working_days(s, e, holiday_dates_for(r, "employee", 2026, TODAY, kb.documents, kb.holidays)[0])
            for r in ("IN", "US", "DE")}
    assert days == {"IN": 8, "US": 7, "DE": 6}
