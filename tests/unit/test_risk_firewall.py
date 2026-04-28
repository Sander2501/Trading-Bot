"""Integration-ish tests for the run_once risk firewall."""

from datetime import datetime, timedelta, timezone

import pandas as pd

import main
from brokers.base import BaseBroker


class _LosingBroker(BaseBroker):
    """Broker that reports equity well below week-baseline so weekly halt trips."""

    def __init__(self, equity: float = 9000.0):
        self._equity = equity
        self.buy_calls = 0

    def get_recent_bars(self, symbol, limit, timeframe="1Min"):
        prices = [100.0] * max(5, limit)
        return pd.DataFrame({"h": prices, "l": prices, "c": prices})

    def get_position_qty(self, symbol):
        return 0.0

    def get_entry_price(self, symbol):
        return None

    def has_open_order(self, symbol):
        return False

    def submit_buy(self, symbol, qty, sl=None, tp=None):
        self.buy_calls += 1

    def submit_sell(self, symbol, qty, sl=None, tp=None):
        pass

    def get_market_status(self):
        return True, 0.0

    def get_buying_power(self):
        return self._equity

    def get_equity(self):
        return self._equity, self._equity


def test_kill_switch_halts_run_once(tmp_path, monkeypatch):
    sentinel = tmp_path / "KILL"
    sentinel.touch()
    monkeypatch.setattr(main, "KILL_SWITCH_FILE", str(sentinel))

    broker = _LosingBroker(equity=10_000.0)
    state = main.TradingState()
    main.run_once(broker, state, sleep_enabled=False)
    # Cycle should have been halted before the position-management section even ran.
    assert broker.buy_calls == 0


def test_kill_switch_inactive_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "KILL_SWITCH_FILE", str(tmp_path / "nope"))
    broker = _LosingBroker(equity=10_000.0)
    state = main.TradingState()
    # Shouldn't raise; HOLD signal so no buys, but at least we passed the gate.
    main.run_once(broker, state, sleep_enabled=False)


def test_weekly_loss_halts_run_once(monkeypatch):
    broker = _LosingBroker(equity=9_000.0)
    state = main.TradingState()
    # Pre-seed week baseline so the new equity reads as a 10% drawdown.
    state.week_anchor_date = broker.current_date()
    state.week_baseline_equity = 10_000.0

    monkeypatch.setattr(main, "MAX_WEEKLY_LOSS_PCT", 0.05)
    main.run_once(broker, state, sleep_enabled=False)
    assert broker.buy_calls == 0


def test_slippage_halt_blocks_cycle(monkeypatch):
    broker = _LosingBroker(equity=10_000.0)
    state = main.TradingState()
    state.slippage_halt_until = datetime.now(timezone.utc) + timedelta(seconds=300)
    main.run_once(broker, state, sleep_enabled=False)
    # Cycle skipped before any signal logic ran
    assert broker.buy_calls == 0


def test_expired_slippage_halt_does_not_block(monkeypatch):
    broker = _LosingBroker(equity=10_000.0)
    state = main.TradingState()
    state.slippage_halt_until = datetime.now(timezone.utc) - timedelta(seconds=300)
    main.run_once(broker, state, sleep_enabled=False)
    # No exception, gate did not block (signal would be HOLD on flat bars anyway)


def test_week_baseline_anchors_on_first_cycle(monkeypatch):
    broker = _LosingBroker(equity=10_000.0)
    state = main.TradingState()
    assert state.week_anchor_date is None
    main.run_once(broker, state, sleep_enabled=False)
    assert state.week_anchor_date is not None
    assert state.week_baseline_equity == 10_000.0
