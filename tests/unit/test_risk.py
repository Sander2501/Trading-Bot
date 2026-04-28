from datetime import date, datetime, timedelta, timezone

import pytest

import risk


def test_kill_switch_disabled_for_empty_path():
    assert risk.kill_switch_active("") is False
    assert risk.kill_switch_active(None) is False


def test_kill_switch_inactive_when_file_missing(tmp_path):
    assert risk.kill_switch_active(tmp_path / "nope") is False


def test_kill_switch_active_when_file_exists(tmp_path):
    f = tmp_path / "KILL"
    f.touch()
    assert risk.kill_switch_active(f) is True


@pytest.mark.parametrize(
    "today,expected_anchor",
    [
        (date(2026, 4, 27), date(2026, 4, 27)),  # Monday
        (date(2026, 4, 28), date(2026, 4, 27)),  # Tuesday
        (date(2026, 5, 3), date(2026, 4, 27)),   # Sunday
        (date(2026, 5, 4), date(2026, 5, 4)),    # next Monday
    ],
)
def test_week_anchor(today, expected_anchor):
    assert risk.week_anchor(today) == expected_anchor


def test_weekly_loss_below_threshold():
    assert not risk.weekly_loss_exceeded(equity=98.0, week_baseline_equity=100.0, threshold_pct=0.05)


def test_weekly_loss_at_threshold_trips():
    assert risk.weekly_loss_exceeded(equity=95.0, week_baseline_equity=100.0, threshold_pct=0.05)


def test_weekly_loss_above_threshold_trips():
    assert risk.weekly_loss_exceeded(equity=90.0, week_baseline_equity=100.0, threshold_pct=0.05)


def test_weekly_loss_disabled_when_threshold_zero():
    assert not risk.weekly_loss_exceeded(equity=0.0, week_baseline_equity=100.0, threshold_pct=0.0)


def test_weekly_loss_disabled_when_baseline_zero():
    assert not risk.weekly_loss_exceeded(equity=50.0, week_baseline_equity=0.0, threshold_pct=0.05)


def test_slippage_halt_until_adds_cooldown():
    now = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    until = risk.slippage_halt_until(now, 600)
    assert until == now + timedelta(seconds=600)


def test_slippage_halt_until_handles_naive_datetime():
    naive = datetime(2026, 4, 28, 12, 0, 0)
    until = risk.slippage_halt_until(naive, 60)
    assert until.tzinfo is timezone.utc


def test_slippage_halt_until_clamps_negative_cooldown():
    now = datetime(2026, 4, 28, tzinfo=timezone.utc)
    until = risk.slippage_halt_until(now, -100)
    assert until == now


def test_is_slippage_halted_none_returns_false():
    assert risk.is_slippage_halted(None) is False


def test_is_slippage_halted_in_future_returns_true():
    future = datetime.now(timezone.utc) + timedelta(seconds=60)
    assert risk.is_slippage_halted(future) is True


def test_is_slippage_halted_in_past_returns_false():
    past = datetime.now(timezone.utc) - timedelta(seconds=60)
    assert risk.is_slippage_halted(past) is False
