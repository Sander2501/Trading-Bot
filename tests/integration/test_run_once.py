"""
tests.integration.test_run_once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end integration tests that drive the full run_once → BacktestBroker
pipeline.  These tests verify that the live trading loop and the simulated
broker work together correctly end-to-end, not just in isolation.
"""

import os

import pandas as pd
import pytest

# Prevent config.py from requiring real env vars during tests.
os.environ.setdefault("CAPITAL_API_KEY", "test")
os.environ.setdefault("CAPITAL_IDENTIFIER", "test@example.com")
os.environ.setdefault("CAPITAL_PASSWORD", "test")

from brokers.backtest import BacktestBroker
from main import TradingState, run_once


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_broker(tmp_path, closes, symbol="BTC/USD", starting_cash=100_000.0):
    """Build a BacktestBroker from a synthetic price series."""
    highs = [c * 1.002 for c in closes]
    lows  = [c * 0.998 for c in closes]
    df = pd.DataFrame({"c": closes, "h": highs, "l": lows})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)
    return BacktestBroker(csv_path=str(csv), symbol=symbol, starting_cash=starting_cash)


@pytest.fixture
def uptrend_broker(tmp_path):
    """200-bar linear uptrend: guaranteed to produce at least one BUY signal."""
    closes = [100.0 + i * 0.5 for i in range(200)]
    return _make_broker(tmp_path, closes)


@pytest.fixture
def downtrend_broker(tmp_path):
    """200-bar linear downtrend."""
    closes = [200.0 - i * 0.5 for i in range(200)]
    return _make_broker(tmp_path, closes)


@pytest.fixture
def flat_broker(tmp_path):
    """200-bar flat market: ADX stays low, strategy should HOLD throughout."""
    import random
    random.seed(0)
    closes = [100.0 + random.uniform(-0.1, 0.1) for _ in range(200)]
    return _make_broker(tmp_path, closes)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


def test_run_once_returns_without_error(uptrend_broker):
    """run_once must complete a full cycle without raising."""
    state = TradingState()
    run_once(uptrend_broker, state=state, sleep_enabled=False)


def test_equity_curve_populated_after_advances(uptrend_broker):
    """Equity curve must have one entry per advance() call."""
    broker = uptrend_broker
    state = TradingState()
    steps = 0
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()
        steps += 1

    assert len(broker.equity_curve) == steps


def test_final_equity_is_positive(uptrend_broker):
    """Equity must remain positive throughout an uptrend."""
    broker = uptrend_broker
    state = TradingState()
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    equity, _ = broker.get_equity()
    assert equity > 0


# ---------------------------------------------------------------------------
# Trade generation
# ---------------------------------------------------------------------------


def test_uptrend_generates_trades(uptrend_broker):
    """A strong uptrend must produce at least one BUY fill."""
    broker = uptrend_broker
    state = TradingState()
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    buy_fills = [t for t in broker.trades if t["side"] in ("BUY",)]
    assert len(buy_fills) >= 1, "Expected at least one BUY in a 200-bar uptrend"


def test_downtrend_no_long_positions(downtrend_broker):
    """A sustained downtrend must not open any long positions."""
    broker = downtrend_broker
    state = TradingState()
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    buy_fills = [t for t in broker.trades if t["side"] == "BUY"]
    assert len(buy_fills) == 0, "Should not BUY in a persistent downtrend"


def test_flat_market_stays_flat(flat_broker):
    """A flat/choppy market (low ADX) should produce no trades."""
    broker = flat_broker
    state = TradingState()
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    assert len(broker.trades) == 0, "Choppy market should produce no trades"


# ---------------------------------------------------------------------------
# SL/TP simulation
# ---------------------------------------------------------------------------


def test_sl_tp_auto_close_via_broker(tmp_path):
    """
    BacktestBroker should auto-close a long position when the low of a bar
    drops below the submitted SL price, without any Python trailing-stop code
    running.
    """
    # Build a price series: rise for a few bars then crash hard.
    closes = [100.0] * 5 + [105.0] * 5 + [80.0] * 10  # crash below any SL
    broker = _make_broker(tmp_path, closes)

    # Open a long with an SL at 90 (well above the crash price of ~80)
    broker._cursor = 9  # position ourselves at bar 9 (price=105)
    broker.submit_buy("BTC/USD", qty=1.0, sl=90.0, tp=200.0)
    assert broker.get_position_qty("BTC/USD") == 1.0

    # advance() moves to bar 10 (close=80, low≈79.84) — SL must fire
    broker.advance()

    assert broker.get_position_qty("BTC/USD") == 0.0, "SL should have auto-closed the long"
    sl_fills = [t for t in broker.trades if t["side"] == "SL_STOP"]
    assert len(sl_fills) == 1


def test_tp_auto_close_via_broker(tmp_path):
    """BacktestBroker should auto-close a long via TP when bar high exceeds TP."""
    closes = [100.0] * 5 + [110.0] * 10
    broker = _make_broker(tmp_path, closes)

    broker._cursor = 4
    broker.submit_buy("BTC/USD", qty=1.0, sl=90.0, tp=108.0)
    assert broker.get_position_qty("BTC/USD") == 1.0

    broker.advance()  # bar 5: close=110, high≈110.22 → TP at 108 is hit

    assert broker.get_position_qty("BTC/USD") == 0.0, "TP should have auto-closed the long"
    tp_fills = [t for t in broker.trades if t["side"] == "TP_STOP"]
    assert len(tp_fills) == 1


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_trading_state_round_trip(tmp_path):
    """TradingState.save() / load() must round-trip all fields."""
    from datetime import date

    state = TradingState()
    state.position_high = 123.45
    state.position_low  = 98.76
    state.last_snapshot_date = date(2025, 1, 15)

    path = str(tmp_path / "state.json")
    state.save(path)

    restored = TradingState.load(path)
    assert restored.position_high == pytest.approx(123.45)
    assert restored.position_low  == pytest.approx(98.76)
    assert restored.last_snapshot_date == date(2025, 1, 15)


def test_trading_state_load_missing_file(tmp_path):
    """load() on a non-existent file must return a default TradingState."""
    state = TradingState.load(str(tmp_path / "no_such_file.json"))
    assert state.position_high == 0.0
    assert state.position_low == float("inf")
    assert state.last_snapshot_date is None


# ---------------------------------------------------------------------------
# Global drawdown halt
# ---------------------------------------------------------------------------


def test_global_drawdown_halt_triggers(tmp_path, monkeypatch):
    """A 20% equity drop from peak must halt new entries regardless of daily baseline."""
    import main

    broker = _make_broker(tmp_path, [100.0 + i * 0.5 for i in range(200)])
    state = TradingState()

    # Cycle 1 establishes peak_equity = $100k.
    run_once(broker, state=state, sleep_enabled=False)
    broker.advance()
    assert state.peak_equity > 0

    # Simulate a large drawdown: equity reports as 80% of peak.
    monkeypatch.setattr(broker, "get_equity", lambda: (80_000.0, 80_000.0))

    prev_trades = len(broker.trades)
    for _ in range(10):
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    assert len(broker.trades) == prev_trades, (
        "Global drawdown halt must block new fills once equity is 20% below peak."
    )


def test_peak_equity_tracks_highs(tmp_path, monkeypatch):
    """peak_equity must monotonically increase as equity rises."""
    broker = _make_broker(tmp_path, [100.0 + i * 0.5 for i in range(200)])
    state = TradingState()

    for eq in (100_000, 105_000, 103_000, 110_000, 108_000):
        monkeypatch.setattr(broker, "get_equity", lambda eq=eq: (float(eq), float(eq)))
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    assert state.peak_equity == pytest.approx(110_000.0)


# ---------------------------------------------------------------------------
# Loss-streak cooldown
# ---------------------------------------------------------------------------


def test_loss_streak_cooldown_blocks_new_entries(tmp_path, monkeypatch):
    """After LOSS_STREAK_HALT losers, new BUY/SELL signals must be skipped."""
    import main

    broker = _make_broker(tmp_path, [100.0 + i * 0.5 for i in range(200)])
    monkeypatch.setattr(main, "LOSS_STREAK_HALT", 3)
    monkeypatch.setattr(main, "LOSS_COOLDOWN_CYCLES", 5)

    state = TradingState()
    # Force a BUY signal so every cycle tries to open.
    monkeypatch.setattr(main, "moving_average_signal", lambda *a, **k: "BUY")

    # Advance past warm-up, then simulate 3 losses directly via the helper.
    run_once(broker, state=state, sleep_enabled=False)
    broker.advance()
    for _ in range(3):
        state.record_trade_outcome(-100.0)
    # Manually trigger cooldown as if a _record_close_and_maybe_cooldown had fired.
    main._record_close_and_maybe_cooldown(state, -100.0)  # 4th loss — but halt=3 so already tripped once

    # Cooldown is now active.  Run several cycles; no new BUY entries should open.
    # (Broker-side SL/TP fills on the pre-existing position are still OK — those
    #  are forced exits, not new entries governed by the cooldown.)
    prev_trade_count = len(broker.trades)
    for _ in range(3):
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    new_entry_fills = [
        t for t in broker.trades[prev_trade_count:] if t["side"] in ("BUY", "SHORT")
    ]
    assert len(new_entry_fills) == 0, (
        f"Expected 0 new BUY/SHORT fills during cooldown, got {len(new_entry_fills)}."
    )


def test_winning_trade_resets_loss_streak():
    """A winning outcome must reset consecutive_losses to 0."""
    state = TradingState()
    state.record_trade_outcome(-50.0)
    state.record_trade_outcome(-75.0)
    assert state.consecutive_losses == 2
    state.record_trade_outcome(+100.0)
    assert state.consecutive_losses == 0


# ---------------------------------------------------------------------------
# Broker-side close detection (attached SL/TP fills between cycles)
# ---------------------------------------------------------------------------


def test_broker_side_close_recorded_as_loss(tmp_path, monkeypatch):
    """When broker closes between cycles (qty flips to 0), PnL must be recorded."""
    # Downtrend — bars at cursor 60+ are well below 100.
    closes = [200.0 - i * 0.5 for i in range(200)]
    broker = _make_broker(tmp_path, closes)
    state = TradingState()

    # Warm up indicators before triggering the close-detection logic.
    for _ in range(60):
        broker.advance()

    # Pretend the previous cycle ended with a long at $200 (well above current).
    state.prev_qty = 1.0
    state.prev_entry_price = 200.0

    # This cycle the broker reports flat (SL fired between cycles).
    monkeypatch.setattr(broker, "get_position_qty", lambda s: 0.0)
    monkeypatch.setattr(broker, "get_entry_price", lambda s: None)

    run_once(broker, state=state, sleep_enabled=False)

    assert state.consecutive_losses == 1, (
        "Broker-side close at lower price than entry should register as a loss."
    )


# ---------------------------------------------------------------------------
# Dry-run broker
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Daily-loss halt + bar-date rollover
# ---------------------------------------------------------------------------


def test_daily_loss_halt_uses_bar_date(tmp_path, monkeypatch):
    """Baseline must refresh when the bar-date changes, not the wall-clock date."""
    from datetime import datetime, timedelta, timezone

    import main

    # 10 bars on day 1, 10 bars on day 2.  Price is irrelevant to this check.
    base = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [(base + timedelta(minutes=i)).isoformat() for i in range(10)] + [
        (base + timedelta(days=1, minutes=i)).isoformat() for i in range(10)
    ]
    df = pd.DataFrame({"t": timestamps, "c": [100.0] * 20, "h": [100.0] * 20, "l": [100.0] * 20})
    csv = tmp_path / "two_days.csv"
    df.to_csv(csv, index=False)

    broker = BacktestBroker(csv_path=str(csv), symbol="BTC/USD", starting_cash=100_000.0)
    state = TradingState()

    # Cycle 1 of day 1 → baseline snapshot taken for day 1.
    run_once(broker, state=state, sleep_enabled=False)
    day1 = state.last_snapshot_date

    # Advance to the middle of day 2.
    for _ in range(15):
        broker.advance()

    # Patch wall-clock so date.today() would NOT roll over — proves the fix
    # relies on the bar's date, not the wall clock.
    class _FrozenDate:
        @classmethod
        def today(cls):
            return day1

    monkeypatch.setattr(main, "date", _FrozenDate)

    run_once(broker, state=state, sleep_enabled=False)
    assert state.last_snapshot_date != day1, (
        "Baseline should roll over based on bar-date, independent of wall clock."
    )


def test_daily_loss_halt_prevents_new_entries(tmp_path, monkeypatch):
    """When equity drops below the daily-loss threshold, run_once must not open positions."""
    import main

    broker = _make_broker(tmp_path, [100.0 + i * 0.5 for i in range(200)])
    state = TradingState()

    # First cycle: takes the baseline snapshot at $100k equity.
    run_once(broker, state=state, sleep_enabled=False)
    broker.advance()

    # Force broker equity to report a large drawdown (5%) without actually trading.
    monkeypatch.setattr(broker, "get_equity", lambda: (95_000.0, 100_000.0))

    prev_trades = len(broker.trades)
    for _ in range(10):
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    assert len(broker.trades) == prev_trades, (
        "Daily-loss halt must block new fills while equity is below the threshold."
    )


# ---------------------------------------------------------------------------
# ALLOW_SHORTS flag
# ---------------------------------------------------------------------------


def test_shorts_disabled_keeps_bot_flat_on_sell(downtrend_broker, monkeypatch):
    """With ALLOW_SHORTS=False, a SELL signal from flat must NOT open a short."""
    import main

    monkeypatch.setattr(main, "ALLOW_SHORTS", False)

    broker = downtrend_broker
    state = TradingState()
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    assert all(t["side"] != "SHORT" for t in broker.trades), (
        "No SHORT fills should occur when ALLOW_SHORTS is False."
    )


def test_shorts_enabled_opens_short_on_sell_signal(tmp_path, monkeypatch):
    """With ALLOW_SHORTS=True, a SELL signal from flat must open a short."""
    import main

    broker = _make_broker(tmp_path, [100.0 + i * 0.01 for i in range(200)])
    monkeypatch.setattr(main, "ALLOW_SHORTS", True)
    # Force a SELL signal so the test isolates the ALLOW_SHORTS branch.
    monkeypatch.setattr(main, "moving_average_signal", lambda *a, **k: "SELL")

    state = TradingState()
    run_once(broker, state=state, sleep_enabled=False)

    assert len(broker.trades) == 1 and broker.trades[0]["side"] == "SHORT", (
        "A forced SELL signal with ALLOW_SHORTS=True should produce exactly one SHORT fill."
    )


def test_dry_run_broker_no_fills(tmp_path, capsys):
    """DryRunBroker must log order intent but not change position."""
    from main import DryRunBroker

    broker = _make_broker(tmp_path, [100.0 + i * 0.5 for i in range(200)])
    dry = DryRunBroker(broker)

    state = TradingState()
    while not broker.done():
        run_once(dry, state=state, sleep_enabled=False)
        broker.advance()

    # The underlying broker should have zero fills because DryRunBroker
    # never calls the real submit methods.
    assert len(broker.trades) == 0
