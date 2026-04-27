import pandas as pd

import main
from brokers.base import BaseBroker


class DummyBroker(BaseBroker):
    def __init__(self, has_open=False):
        self._has_open = has_open

    def get_recent_bars(self, symbol: str, limit: int, timeframe: str = "1Min") -> pd.DataFrame:
        prices = [100.0] * max(5, limit)
        return pd.DataFrame({"h": prices, "l": prices, "c": prices})

    def get_position_qty(self, symbol: str) -> float:
        return 0.0

    def get_entry_price(self, symbol: str) -> float | None:
        return None

    def has_open_order(self, symbol: str) -> bool:
        return self._has_open

    def submit_buy(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        return None

    def submit_sell(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        return None

    def get_market_status(self) -> tuple[bool, float]:
        return True, 0.0

    def get_buying_power(self) -> float:
        return 10000.0

    def get_equity(self) -> tuple[float, float]:
        return 10000.0, 10000.0


class DummyPositionBroker(BaseBroker):
    def __init__(self):
        self.qty = 1.0
        self.entry = 100.0
        self.sells: list[float] = []

    def get_recent_bars(self, symbol: str, limit: int, timeframe: str = "1Min") -> pd.DataFrame:
        # same bar triggers both potential partial (high) and stop (low)
        return pd.DataFrame({"h": [120.0] * limit, "l": [80.0] * limit, "c": [100.0] * limit})

    def get_position_qty(self, symbol: str) -> float:
        return self.qty

    def get_entry_price(self, symbol: str) -> float | None:
        return self.entry

    def has_open_order(self, symbol: str) -> bool:
        return False

    def submit_buy(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        self.qty += qty

    def submit_sell(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        self.sells.append(qty)
        self.qty -= qty

    def get_market_status(self) -> tuple[bool, float]:
        return True, 0.0

    def get_buying_power(self) -> float:
        return 10000.0

    def get_equity(self) -> tuple[float, float]:
        return 10000.0, 10000.0


def test_update_error_state_trips_circuit_breaker(monkeypatch):
    monkeypatch.setattr(main, "MAX_CONSECUTIVE_ERRORS", 3)
    monkeypatch.setattr(main, "MAX_BACKOFF_SECONDS", 60)

    errors, backoff, tripped = main._update_error_state(2, 30)

    assert errors == 3
    assert backoff == 60
    assert tripped is True


def test_update_error_state_doubles_backoff_without_trip(monkeypatch):
    monkeypatch.setattr(main, "MAX_CONSECUTIVE_ERRORS", 5)
    monkeypatch.setattr(main, "MAX_BACKOFF_SECONDS", 120)

    errors, backoff, tripped = main._update_error_state(1, 10)

    assert errors == 2
    assert backoff == 20
    assert tripped is False


def test_open_order_streak_increments_and_resets(monkeypatch):
    monkeypatch.setattr(main, "OPEN_ORDER_STALE_CYCLES", 2)

    state = main.TradingState()
    broker_with_open = DummyBroker(has_open=True)
    main.run_once(broker_with_open, state, sleep_enabled=False)
    assert state.open_order_streak == 1

    main.run_once(broker_with_open, state, sleep_enabled=False)
    assert state.open_order_streak == 2

    broker_flat = DummyBroker(has_open=False)
    main.run_once(broker_flat, state, sleep_enabled=False)
    assert state.open_order_streak == 0


def test_stop_has_priority_over_partial_tp(monkeypatch):
    broker = DummyPositionBroker()
    state = main.TradingState()
    state.position_high = 100.0
    state.position_low = 100.0
    state.seed_trade_context(100.0, 10.0)

    monkeypatch.setattr(main, "atr_stop_distance", lambda *args, **kwargs: 10.0)
    monkeypatch.setattr(main, "ATR_STOP_MULT", 1.0)
    monkeypatch.setattr(main, "STOP_LOSS_PCT", 0.0)
    monkeypatch.setattr(main, "TAKE_PROFIT_MULT", 100.0)
    monkeypatch.setattr(main, "TAKE_PROFIT_PCT", 100.0)
    monkeypatch.setattr(main, "moving_average_signal", lambda *args, **kwargs: "HOLD")

    main.run_once(broker, state, sleep_enabled=False)

    assert broker.qty == 0.0
    assert broker.sells and broker.sells[0] == 1.0
    assert state.partial_tp1_taken is False
