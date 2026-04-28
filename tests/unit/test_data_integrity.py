from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data_integrity import check_bars, check_csv, interval_seconds_for


def _frame(rows: int = 5, *, with_t: bool = True) -> pd.DataFrame:
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    data = {
        "h": [101.0] * rows,
        "l": [99.0] * rows,
        "c": [100.0] * rows,
    }
    if with_t:
        data["t"] = [base + timedelta(minutes=15 * i) for i in range(rows)]
    return pd.DataFrame(data)


def test_clean_frame_passes():
    result = check_bars(_frame(), expected_interval_seconds=900)
    assert result.ok
    assert result.issues == []


def test_missing_columns_is_fatal():
    df = pd.DataFrame({"h": [1.0], "l": [1.0]})
    result = check_bars(df)
    assert not result.ok
    assert any("missing required" in m for m in result.fatal)


def test_empty_batch_is_fatal():
    result = check_bars(_frame(0))
    assert not result.ok
    assert any("empty bar batch" in m for m in result.fatal)


def test_nan_in_close_is_fatal():
    df = _frame()
    df.loc[2, "c"] = float("nan")
    result = check_bars(df)
    assert not result.ok
    assert any("NaN" in m for m in result.fatal)


def test_high_below_low_is_fatal():
    df = _frame()
    df.loc[1, "h"] = 50.0
    result = check_bars(df)
    assert not result.ok
    assert any("high < low" in m for m in result.fatal)


def test_close_outside_high_low_is_fatal():
    df = _frame()
    df.loc[1, "c"] = 1000.0
    result = check_bars(df)
    assert not result.ok
    assert any("close is outside" in m for m in result.fatal)


def test_duplicate_timestamps_are_fatal():
    df = _frame()
    df.loc[3, "t"] = df.loc[2, "t"]
    result = check_bars(df, expected_interval_seconds=900)
    assert not result.ok
    assert any("duplicate timestamp" in m for m in result.fatal)


def test_non_monotonic_timestamps_are_fatal():
    df = _frame()
    df.loc[3, "t"], df.loc[4, "t"] = df.loc[4, "t"], df.loc[3, "t"]
    result = check_bars(df, expected_interval_seconds=900)
    assert not result.ok
    assert any("monotonic" in m for m in result.fatal)


def test_gap_is_warning_not_fatal():
    df = _frame()
    df.loc[3, "t"] = df.loc[2, "t"] + timedelta(minutes=60)
    df.loc[4, "t"] = df.loc[3, "t"] + timedelta(minutes=15)
    result = check_bars(df, expected_interval_seconds=900)
    assert result.ok  # gaps are issues, not fatal
    assert any("gap" in m for m in result.issues)


def test_stale_feed_is_fatal():
    df = _frame()
    now = df["t"].iloc[-1].to_pydatetime() + timedelta(hours=2)
    result = check_bars(df, max_age_seconds=900, now=now)
    assert not result.ok
    assert any("old" in m for m in result.fatal)


def test_fresh_feed_passes():
    df = _frame()
    now = df["t"].iloc[-1].to_pydatetime() + timedelta(seconds=30)
    result = check_bars(df, max_age_seconds=900, now=now)
    assert result.ok


def test_check_csv_missing_file():
    result = check_csv("/nonexistent/path.csv")
    assert not result.ok
    assert any("file not found" in m for m in result.fatal)


def test_check_csv_clean_file(tmp_path):
    df = _frame(rows=10)
    path = tmp_path / "bars.csv"
    df.to_csv(path, index=False)
    result = check_csv(path, expected_interval_seconds=900)
    assert result.ok


def test_check_csv_with_dupes(tmp_path):
    df = _frame(rows=5)
    df.loc[2, "t"] = df.loc[1, "t"]
    path = tmp_path / "bars.csv"
    df.to_csv(path, index=False)
    result = check_csv(path, expected_interval_seconds=900)
    assert not result.ok


@pytest.mark.parametrize(
    "tf,expected",
    [("1Min", 60), ("15Min", 900), ("1H", 3600), ("4H", 14400), ("1D", 86400), ("nope", None)],
)
def test_interval_seconds_for(tf, expected):
    assert interval_seconds_for(tf) == expected
