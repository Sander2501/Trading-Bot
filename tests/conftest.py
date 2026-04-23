"""
tests.conftest
~~~~~~~~~~~~~~
Shared pytest fixtures for the Trading-Bot test suite.
"""

import os
import tempfile

import pandas as pd
import pytest

from brokers.backtest import BacktestBroker


# ---------------------------------------------------------------------------
# CSV / broker helpers
# ---------------------------------------------------------------------------


def _write_temp_csv(prices: list[float]) -> str:
    """Write close prices to a temp CSV and return the path."""
    rows = [{"c": p, "h": p * 1.001, "l": p * 0.999} for p in prices]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture()
def flat_broker():
    """BacktestBroker with 20 bars at a constant $100 price."""
    path = _write_temp_csv([100.0] * 20)
    broker = BacktestBroker(path, symbol="BTC/USD", starting_cash=10_000.0, slippage_pct=0.0)
    yield broker
    os.unlink(path)


@pytest.fixture()
def rising_broker():
    """BacktestBroker with 100 bars rising from $100 to $199."""
    prices = [100.0 + i for i in range(100)]
    path = _write_temp_csv(prices)
    broker = BacktestBroker(path, symbol="BTC/USD", starting_cash=100_000.0, slippage_pct=0.0)
    yield broker
    os.unlink(path)
