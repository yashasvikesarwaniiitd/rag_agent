"""Daily spending cap for a public deployment. Pure logic; the clock is injected.

The cap is checked BEFORE a paid question and the question's real cost is recorded AFTER,
so a day can overshoot by at most one question (~$0.001). Days roll over at midnight UTC.
The tally lives in memory: an app restart resets it, which is why the Anthropic Console's
own monthly limit is the backstop, not this.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date, datetime, timezone


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


class DailyBudget:
    def __init__(self, limit_usd: float, today: Callable[[], date] = _utc_today):
        if limit_usd < 0:
            raise ValueError("limit must be >= 0")
        self.limit = limit_usd
        self._today = today
        self._day = today()
        self._spent = 0.0
        self._lock = threading.Lock()  # Streamlit serves sessions from several threads

    def _roll(self) -> None:
        now = self._today()
        if now != self._day:
            self._day, self._spent = now, 0.0

    @property
    def spent(self) -> float:
        with self._lock:
            self._roll()
            return self._spent

    def allows(self) -> bool:
        with self._lock:
            self._roll()
            return self._spent < self.limit

    def record(self, cost_usd: float | None) -> None:
        """An unknown cost (None) counts as the whole remaining budget: if spend can't be
        measured, it must not be allowed to continue unmeasured."""
        with self._lock:
            self._roll()
            self._spent = self.limit if cost_usd is None else self._spent + max(cost_usd, 0.0)
