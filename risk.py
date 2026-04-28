"""
risk
~~~~
Capital-protection firewall. Pure functions over state — no I/O beyond the
kill-switch file existence check, no mutation of broker state. Callers
(``run_once``) decide what to do with the result.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def kill_switch_active(path: str | Path | None) -> bool:
    """Return True when the operator-controlled kill-switch file exists.

    Empty / None disables the check. Any I/O error is treated as not active
    so a transient filesystem problem can't accidentally halt trading.
    """
    if not path:
        return False
    try:
        return Path(path).exists()
    except OSError:
        return False


def week_anchor(today: date) -> date:
    """Monday of the ISO week containing ``today``.

    All week-based loss tracking pivots on Monday so the lock resets cleanly
    on a known boundary regardless of when the bot was started.
    """
    return today - timedelta(days=today.weekday())


def weekly_loss_exceeded(
    equity: float,
    week_baseline_equity: float,
    threshold_pct: float,
) -> bool:
    """Return True when drawdown vs the start-of-week baseline >= threshold."""
    if threshold_pct <= 0 or week_baseline_equity <= 0:
        return False
    drawdown = (week_baseline_equity - equity) / week_baseline_equity
    return drawdown >= threshold_pct


def slippage_halt_until(now: datetime, cooldown_seconds: int) -> datetime:
    """Return the wall-clock instant at which a slippage halt expires."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now + timedelta(seconds=max(0, cooldown_seconds))


def is_slippage_halted(halt_until: datetime | None, now: datetime | None = None) -> bool:
    """Return True while a slippage cooldown is active."""
    if halt_until is None:
        return False
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    if halt_until.tzinfo is None:
        halt_until = halt_until.replace(tzinfo=timezone.utc)
    return reference < halt_until
