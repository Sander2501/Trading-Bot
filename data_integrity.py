"""
data_integrity
~~~~~~~~~~~~~~
Validation for historical and live OHLC bar data.

Two entry points:

* :func:`check_csv` — static validation of a historical-bar CSV. Used by
  ``scripts/validate_data.py`` to vet a dataset before backtesting.
* :func:`check_bars` — runtime validation of a bar batch returned by a broker.
  Called from ``run_once`` so the bot refuses to trade on obviously bad data
  (NaNs, duplicate timestamps, gaps, stale feed, broken OHLC invariants).

A check returns a :class:`CheckResult` with two lists: ``issues`` (everything
flagged) and ``fatal`` (the subset severe enough to block trading). Callers
decide how to react; this module never logs or raises on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("h", "l", "c")


@dataclass
class CheckResult:
    """Outcome of a data-integrity check."""

    issues: list[str] = field(default_factory=list)
    fatal: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal

    def add(self, msg: str, *, fatal: bool = False) -> None:
        self.issues.append(msg)
        if fatal:
            self.fatal.append(msg)

    def merge(self, other: "CheckResult") -> None:
        self.issues.extend(other.issues)
        self.fatal.extend(other.fatal)


def _check_columns(df: pd.DataFrame, result: CheckResult) -> bool:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.add(f"missing required columns: {missing}", fatal=True)
        return False
    return True


def _check_numeric(df: pd.DataFrame, result: CheckResult) -> None:
    """NaN, inf, and OHLC invariants."""
    for col in REQUIRED_COLUMNS:
        s = df[col]
        nan_count = int(s.isna().sum())
        if nan_count:
            result.add(f"column {col!r} has {nan_count} NaN value(s)", fatal=True)
        inf_count = int((~s.isna() & ~s.replace([float("inf"), float("-inf")], pd.NA).notna()).sum())
        if inf_count:
            result.add(f"column {col!r} has {inf_count} inf value(s)", fatal=True)
        nonpos = int((s <= 0).sum()) if s.dtype.kind in "fc" else 0
        if nonpos:
            result.add(f"column {col!r} has {nonpos} non-positive value(s)", fatal=True)

    if all(c in df.columns for c in REQUIRED_COLUMNS):
        bad_hl = int((df["h"] < df["l"]).sum())
        if bad_hl:
            result.add(f"{bad_hl} row(s) where high < low", fatal=True)
        bad_c = int(((df["c"] > df["h"]) | (df["c"] < df["l"])).sum())
        if bad_c:
            result.add(f"{bad_c} row(s) where close is outside [low, high]", fatal=True)


def _parse_timestamps(df: pd.DataFrame, result: CheckResult) -> pd.Series | None:
    if "t" not in df.columns:
        return None
    raw = df["t"]
    try:
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
    except (ValueError, TypeError) as exc:
        result.add(f"could not parse 't' column as timestamps: {exc}", fatal=True)
        return None
    bad = int(ts.isna().sum())
    if bad:
        result.add(f"{bad} unparseable timestamp(s) in 't' column", fatal=True)
    return ts


def _check_timestamps(
    ts: pd.Series,
    result: CheckResult,
    *,
    expected_interval_seconds: int | None,
) -> None:
    if ts.empty:
        return

    # Already coerced to UTC by _parse_timestamps; warn if the source dtype was naive.
    if not getattr(ts.dt, "tz", None):
        result.add("timestamps were tz-naive; coerced to UTC for checks")

    diffs = ts.diff().dropna()
    if (diffs < pd.Timedelta(0)).any():
        result.add("timestamps are not monotonically non-decreasing", fatal=True)

    dupes = int(ts.duplicated().sum())
    if dupes:
        result.add(f"{dupes} duplicate timestamp(s)", fatal=True)

    if expected_interval_seconds is not None and not diffs.empty:
        expected = pd.Timedelta(seconds=expected_interval_seconds)
        gap_threshold = expected * 1.5
        gaps = diffs[diffs > gap_threshold]
        if not gaps.empty:
            biggest = gaps.max()
            result.add(
                f"{len(gaps)} gap(s) exceeding {gap_threshold} "
                f"(largest: {biggest})"
            )
        # Catch the operator-error case where the configured timeframe is
        # coarser/finer than the actual data — the median diff should be
        # close to the expected interval.
        median = diffs.median()
        if median < expected * 0.5 or median > expected * 2:
            result.add(
                f"median bar interval ({median}) differs sharply from "
                f"expected ({expected}); wrong TIMEFRAME?"
            )


def _check_freshness(
    ts: pd.Series,
    result: CheckResult,
    *,
    max_age_seconds: int | None,
    now: datetime | None,
) -> None:
    if max_age_seconds is None or ts.empty:
        return
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    last = ts.iloc[-1]
    if pd.isna(last):
        return
    age = (reference - last.to_pydatetime()).total_seconds()
    if age > max_age_seconds:
        result.add(
            f"latest bar is {age:.0f}s old (> {max_age_seconds}s threshold)",
            fatal=True,
        )


def check_bars(
    bars: pd.DataFrame,
    *,
    expected_interval_seconds: int | None = None,
    max_age_seconds: int | None = None,
    now: datetime | None = None,
) -> CheckResult:
    """Validate a bar batch returned by a broker.

    ``expected_interval_seconds`` enables gap detection; ``max_age_seconds``
    enables stale-feed detection (the latest bar must be at least this fresh
    relative to ``now``). Both are no-ops when ``t`` is absent from the frame.
    """
    result = CheckResult()
    if not _check_columns(bars, result):
        return result
    if len(bars) == 0:
        result.add("empty bar batch", fatal=True)
        return result
    _check_numeric(bars, result)
    ts = _parse_timestamps(bars, result)
    if ts is not None:
        _check_timestamps(ts, result, expected_interval_seconds=expected_interval_seconds)
        _check_freshness(ts, result, max_age_seconds=max_age_seconds, now=now)
    return result


def check_csv(
    path: str | Path,
    *,
    expected_interval_seconds: int | None = None,
) -> CheckResult:
    """Validate a historical-bar CSV file. Used offline before backtesting."""
    p = Path(path)
    result = CheckResult()
    if not p.exists():
        result.add(f"file not found: {p}", fatal=True)
        return result
    try:
        df = pd.read_csv(p)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        result.add(f"could not read CSV: {exc}", fatal=True)
        return result
    if "t" not in df.columns:
        result.add("CSV has no 't' column; timestamp checks skipped")
    result.merge(check_bars(df, expected_interval_seconds=expected_interval_seconds))
    return result


def interval_seconds_for(timeframe: str) -> int | None:
    """Map a config TIMEFRAME string to its bar interval in seconds."""
    mapping = {
        "1Min": 60,
        "5Min": 300,
        "15Min": 900,
        "30Min": 1800,
        "1H": 3600,
        "4H": 14400,
        "1D": 86400,
    }
    return mapping.get(timeframe.strip())
