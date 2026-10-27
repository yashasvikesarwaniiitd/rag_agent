from datetime import date

import pytest

from rag.domain.budget import DailyBudget


class Clock:
    def __init__(self, d):
        self.d = d

    def __call__(self):
        return self.d


def test_allows_until_the_limit_is_reached():
    b = DailyBudget(0.50, Clock(date(2026, 10, 28)))
    b.record(0.30)
    assert b.allows()
    b.record(0.25)  # overshoot of one question is allowed, then it stops
    assert not b.allows() and b.spent == pytest.approx(0.55)


def test_new_utc_day_resets_the_tally():
    clock = Clock(date(2026, 10, 28))
    b = DailyBudget(0.50, clock)
    b.record(0.60)
    assert not b.allows()
    clock.d = date(2026, 10, 29)
    assert b.allows() and b.spent == 0.0


def test_unknown_cost_exhausts_the_budget():
    b = DailyBudget(0.50, Clock(date(2026, 10, 28)))
    b.record(None)
    assert not b.allows()


def test_zero_limit_blocks_everything_and_negative_is_rejected():
    assert not DailyBudget(0.0).allows()
    with pytest.raises(ValueError):
        DailyBudget(-1)
