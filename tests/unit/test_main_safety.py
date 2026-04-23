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
