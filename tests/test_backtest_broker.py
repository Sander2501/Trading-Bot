"""
tests.test_backtest_broker
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for BacktestBroker position tracking and order execution.
"""

import os
import tempfile

import pandas as pd
import pytest

from brokers.backtest import BacktestBroker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _simple_broker(prices: list[float], **kwargs) -> BacktestBroker:
    """Create a BacktestBroker backed by a temp CSV with the given close prices."""
    rows = [{"c": p, "h": p * 1.001, "l": p * 0.999} for p in prices]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name
    _write_csv(rows, tmp_path)
    broker = BacktestBroker(tmp_path, symbol="BTC/USD", **kwargs)
    os.unlink(tmp_path)
    return broker


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestInitValidation:
    def test_invalid_starting_cash_zero(self):
        with pytest.raises(ValueError, match="starting_cash"):
            _simple_broker([100.0], starting_cash=0)

    def test_invalid_starting_cash_negative(self):
        with pytest.raises(ValueError, match="starting_cash"):
            _simple_broker([100.0], starting_cash=-1000)

    def test_invalid_slippage_negative(self):
        with pytest.raises(ValueError, match="slippage_pct"):
            _simple_broker([100.0], slippage_pct=-0.001)

    def test_invalid_slippage_too_high(self):
        with pytest.raises(ValueError, match="slippage_pct"):
            _simple_broker([100.0], slippage_pct=0.2)

    def test_invalid_commission_negative(self):
        with pytest.raises(ValueError, match="commission_per_trade"):
            _simple_broker([100.0], commission_per_trade=-1.0)

    def test_missing_close_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_path = f.name
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(tmp_path, index=False)
        with pytest.raises(ValueError, match="'c' column"):
            BacktestBroker(tmp_path, symbol="BTC/USD")
        os.unlink(tmp_path)

    def test_non_numeric_close_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_path = f.name
        pd.DataFrame({"c": ["a", "b", "c"]}).to_csv(tmp_path, index=False)
        with pytest.raises((ValueError, Exception)):
            BacktestBroker(tmp_path, symbol="BTC/USD")
        os.unlink(tmp_path)

    def test_valid_broker_created(self):
        broker = _simple_broker([100.0, 101.0, 102.0])
        assert broker is not None


# ---------------------------------------------------------------------------
# Navigation: advance / done
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_not_done_at_start(self):
        broker = _simple_broker([100.0, 101.0])
        assert not broker.done()

    def test_done_after_all_bars(self):
        broker = _simple_broker([100.0])
        broker.advance()
        assert broker.done()

    def test_equity_curve_grows_on_advance(self):
        broker = _simple_broker([100.0, 101.0, 102.0])
        broker.advance()
        broker.advance()
        assert len(broker.equity_curve) == 2


# ---------------------------------------------------------------------------
# Position queries
# ---------------------------------------------------------------------------


class TestPositionQueries:
    def test_initial_position_is_zero(self):
        broker = _simple_broker([100.0])
        assert broker.get_position_qty("BTC/USD") == 0.0

    def test_initial_entry_price_is_none(self):
        broker = _simple_broker([100.0])
        assert broker.get_entry_price("BTC/USD") is None

    def test_wrong_symbol_raises(self):
        broker = _simple_broker([100.0])
        with pytest.raises(ValueError, match="ETH/USD"):
            broker.get_position_qty("ETH/USD")


# ---------------------------------------------------------------------------
# Order execution: submit_buy
# ---------------------------------------------------------------------------


class TestSubmitBuy:
    def test_buy_increases_qty(self):
        broker = _simple_broker([100.0] * 5, starting_cash=10_000.0)
        broker.submit_buy("BTC/USD", 1.0)
        assert broker.get_position_qty("BTC/USD") == pytest.approx(1.0)

    def test_buy_reduces_cash(self):
        broker = _simple_broker([100.0] * 5, starting_cash=10_000.0, slippage_pct=0.0)
        broker.submit_buy("BTC/USD", 1.0)
        # cash should have decreased by ~100
        assert broker.get_buying_power() < 10_000.0

    def test_buy_records_trade(self):
        broker = _simple_broker([100.0] * 5, starting_cash=10_000.0)
        broker.submit_buy("BTC/USD", 0.5)
        assert len(broker.trades) == 1
        assert broker.trades[0]["side"] == "BUY"

    def test_buy_zero_qty_raises(self):
        broker = _simple_broker([100.0] * 5)
        with pytest.raises(ValueError, match="quantity"):
            broker.submit_buy("BTC/USD", 0.0)

    def test_buy_negative_qty_raises(self):
        broker = _simple_broker([100.0] * 5)
        with pytest.raises(ValueError, match="quantity"):
            broker.submit_buy("BTC/USD", -1.0)

    def test_buy_insufficient_cash_raises(self):
        broker = _simple_broker([100.0] * 5, starting_cash=50.0)
        with pytest.raises(ValueError, match="[Cc]ash"):
            broker.submit_buy("BTC/USD", 10.0)

    def test_entry_price_set_after_buy(self):
        broker = _simple_broker([100.0] * 5, starting_cash=10_000.0, slippage_pct=0.0)
        broker.submit_buy("BTC/USD", 1.0)
        assert broker.get_entry_price("BTC/USD") == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Order execution: submit_sell
# ---------------------------------------------------------------------------


class TestSubmitSell:
    def _broker_with_long(self, price: float = 100.0) -> BacktestBroker:
        broker = _simple_broker([price] * 10, starting_cash=10_000.0, slippage_pct=0.0)
        broker.submit_buy("BTC/USD", 1.0)
        return broker

    def test_sell_decreases_qty(self):
        broker = self._broker_with_long()
        broker.submit_sell("BTC/USD", 1.0)
        assert broker.get_position_qty("BTC/USD") == pytest.approx(0.0)

    def test_sell_records_trade(self):
        broker = self._broker_with_long()
        broker.submit_sell("BTC/USD", 1.0)
        assert broker.trades[-1]["side"] == "SELL"

    def test_sell_zero_qty_raises(self):
        broker = self._broker_with_long()
        with pytest.raises(ValueError, match="quantity"):
            broker.submit_sell("BTC/USD", 0.0)

    def test_sell_more_than_held_raises(self):
        broker = self._broker_with_long()
        with pytest.raises(ValueError, match="[Cc]annot sell"):
            broker.submit_sell("BTC/USD", 5.0)

    def test_entry_price_cleared_after_full_close(self):
        broker = self._broker_with_long()
        broker.submit_sell("BTC/USD", 1.0)
        assert broker.get_entry_price("BTC/USD") is None

    def test_sell_after_done_with_partial_fill_model_does_not_oob(self):
        """
        Regression: final liquidation in run_backtest can call submit_sell when
        cursor == len(closes). Partial-fill path must clamp index access.
        """
        broker = _simple_broker(
            [100.0, 101.0, 102.0],
            starting_cash=10_000.0,
            slippage_pct=0.0,
            partial_fill_min=0.8,
        )
        broker.submit_buy("BTC/USD", 1.0)
        while not broker.done():
            broker.advance()
        # Should not raise IndexError.
        qty_before = broker.get_position_qty("BTC/USD")
        broker.submit_sell("BTC/USD", broker.get_position_qty("BTC/USD"))
        assert broker.get_position_qty("BTC/USD") < qty_before


# ---------------------------------------------------------------------------
# Equity / buying power
# ---------------------------------------------------------------------------


class TestEquity:
    def test_equity_equals_cash_when_flat(self):
        broker = _simple_broker([100.0] * 5, starting_cash=50_000.0)
        equity, _ = broker.get_equity()
        assert equity == pytest.approx(50_000.0)

    def test_buying_power_decreases_after_buy(self):
        broker = _simple_broker([100.0] * 5, starting_cash=10_000.0, slippage_pct=0.0)
        bp_before = broker.get_buying_power()
        broker.submit_buy("BTC/USD", 1.0)
        assert broker.get_buying_power() < bp_before

    def test_snapshot_day_updates_last_equity(self):
        broker = _simple_broker([100.0] * 5, starting_cash=10_000.0)
        broker.snapshot_day()
        _, last_equity = broker.get_equity()
        assert last_equity == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Market status / misc
# ---------------------------------------------------------------------------


class TestMarketStatus:
    def test_market_always_open(self):
        broker = _simple_broker([100.0])
        is_open, wait = broker.get_market_status()
        assert is_open is True
        assert wait == 0.0

    def test_has_no_open_orders(self):
        broker = _simple_broker([100.0])
        assert broker.has_open_order("BTC/USD") is False

    def test_supports_shorting(self):
        broker = _simple_broker([100.0])
        assert broker.supports_shorting is True

    def test_symbol_sort_by_t_column(self):
        """Rows should be sorted by 't' if the column exists."""
        rows = [
            {"c": 102.0, "h": 103.0, "l": 101.0, "t": "2024-01-03"},
            {"c": 100.0, "h": 101.0, "l": 99.0,  "t": "2024-01-01"},
            {"c": 101.0, "h": 102.0, "l": 100.0, "t": "2024-01-02"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_path = f.name
        _write_csv(rows, tmp_path)
        broker = BacktestBroker(tmp_path, symbol="BTC/USD")
        os.unlink(tmp_path)
        # After sorting, first close should be 100.0
        bars = broker.get_recent_bars("BTC/USD", limit=1)
        assert float(bars["c"].iloc[-1]) == pytest.approx(100.0)

    def test_get_recent_bars_resamples_by_timeframe(self):
        rows = [
            {"t": "2024-01-01 00:00:00", "h": 101.0, "l": 99.0, "c": 100.0},
            {"t": "2024-01-01 00:01:00", "h": 102.0, "l": 100.0, "c": 101.0},
            {"t": "2024-01-01 00:02:00", "h": 103.0, "l": 101.0, "c": 102.0},
            {"t": "2024-01-01 00:03:00", "h": 104.0, "l": 102.0, "c": 103.0},
            {"t": "2024-01-01 00:04:00", "h": 105.0, "l": 103.0, "c": 104.0},
            {"t": "2024-01-01 00:05:00", "h": 106.0, "l": 104.0, "c": 105.0},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_path = f.name
        _write_csv(rows, tmp_path)
        broker = BacktestBroker(tmp_path, symbol="BTC/USD")
        os.unlink(tmp_path)

        # Move cursor to the latest bar.
        while not broker.done():
            broker.advance()

        bars_5m = broker.get_recent_bars("BTC/USD", limit=10, timeframe="5Min")
        # 00:00-00:04 and 00:05 only => at least 2 bars expected.
        assert len(bars_5m) >= 2
        assert float(bars_5m["h"].iloc[0]) == pytest.approx(105.0)
        assert float(bars_5m["l"].iloc[0]) == pytest.approx(99.0)
