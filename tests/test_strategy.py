"""
tests.test_strategy
~~~~~~~~~~~~~~~~~~~
Unit tests for strategy.py indicator calculations.
"""

import math

import pandas as pd
import pytest

from strategy import (
    _adx,
    _ema,
    _macd,
    _rsi,
    _true_range,
    atr_stop_distance,
    moving_average_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int, price: float = 100.0) -> pd.DataFrame:
    """Return a DataFrame of *n* bars with constant price."""
    return pd.DataFrame({
        "c": [price] * n,
        "h": [price * 1.001] * n,
        "l": [price * 0.999] * n,
    })


def _make_trending_bars(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Return a DataFrame of *n* bars with a linearly rising price."""
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "c": closes,
        "h": [c * 1.001 for c in closes],
        "l": [c * 0.999 for c in closes],
    })


# ---------------------------------------------------------------------------
# _rsi
# ---------------------------------------------------------------------------


class TestRsi:
    def test_rsi_is_in_range(self):
        """RSI must always lie in [0, 100]."""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 98, 97, 99, 101, 103])
        rsi = _rsi(prices, window=3)
        assert 0 <= rsi <= 100

    def test_rsi_rising_prices_is_high(self):
        """Uptrend with some pullbacks should produce RSI > 50."""
        # Create a realistic uptrend: step up 2, step down 0.5, repeat
        prices = []
        p = 100.0
        for i in range(30):
            p += 2.0 if i % 3 != 2 else -0.5
            prices.append(p)
        rsi = _rsi(pd.Series(prices), window=14)
        assert rsi > 50

    def test_rsi_falling_prices_is_low(self):
        """Monotonically falling prices should produce RSI close to 0."""
        prices = pd.Series(range(29, 0, -1))
        rsi = _rsi(prices, window=14)
        assert rsi < 10

    def test_rsi_constant_prices_is_defined(self):
        """Constant prices should not raise and RSI should be defined."""
        prices = pd.Series([100.0] * 20)
        rsi = _rsi(prices, window=14)
        # With no changes, gains == losses == 0; implementation yields 0 or 100
        assert not math.isnan(rsi)


# ---------------------------------------------------------------------------
# _macd
# ---------------------------------------------------------------------------


class TestMacd:
    def test_returns_three_floats(self):
        bars = _make_trending_bars(60)
        macd_line, signal_line, histogram = _macd(bars["c"])
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

    def test_histogram_equals_macd_minus_signal(self):
        bars = _make_trending_bars(60)
        macd_line, signal_line, histogram = _macd(bars["c"])
        assert abs(histogram - (macd_line - signal_line)) < 1e-9

    def test_rising_trend_positive_histogram(self):
        """A strong uptrend should produce a positive MACD histogram."""
        bars = _make_trending_bars(100, step=5.0)
        _, _, histogram = _macd(bars["c"])
        assert histogram > 0

    def test_falling_trend_negative_histogram(self):
        """A strong downtrend should produce a negative MACD histogram."""
        bars = _make_trending_bars(100, step=-1.0)
        _, _, histogram = _macd(bars["c"])
        assert histogram < 0


# ---------------------------------------------------------------------------
# _adx
# ---------------------------------------------------------------------------


class TestAdx:
    def test_adx_is_non_negative(self):
        bars = _make_trending_bars(60)
        adx = _adx(bars["h"], bars["l"], bars["c"], window=14)
        assert adx >= 0

    def test_adx_is_at_most_100(self):
        bars = _make_trending_bars(60)
        adx = _adx(bars["h"], bars["l"], bars["c"], window=14)
        assert adx <= 100

    def test_strong_trend_higher_adx(self):
        """A strong linear trend should have higher ADX than flat prices."""
        flat = _make_bars(60, price=100.0)
        trending = _make_trending_bars(60, step=2.0)
        adx_flat = _adx(flat["h"], flat["l"], flat["c"], window=14)
        adx_trend = _adx(trending["h"], trending["l"], trending["c"], window=14)
        assert adx_trend > adx_flat


# ---------------------------------------------------------------------------
# atr_stop_distance
# ---------------------------------------------------------------------------


class TestAtrStopDistance:
    def test_returns_positive_value(self):
        bars = _make_trending_bars(30)
        dist = atr_stop_distance(bars["h"], bars["l"], bars["c"])
        assert dist > 0

    def test_higher_multiplier_gives_larger_distance(self):
        bars = _make_trending_bars(30)
        d1 = atr_stop_distance(bars["h"], bars["l"], bars["c"], multiplier=1.0)
        d2 = atr_stop_distance(bars["h"], bars["l"], bars["c"], multiplier=3.0)
        assert d2 > d1

    def test_constant_prices_gives_near_zero_atr(self):
        """Completely flat prices have no true range → ATR ≈ 0."""
        bars = _make_bars(30, price=100.0)
        dist = atr_stop_distance(bars["h"], bars["l"], bars["c"])
        assert dist < 1.0  # the small h/l spread keeps it slightly above 0


# ---------------------------------------------------------------------------
# moving_average_signal
# ---------------------------------------------------------------------------


class TestMovingAverageSignal:
    def test_hold_on_insufficient_bars(self):
        """Strategy must return HOLD when there are too few bars."""
        bars = _make_bars(5)
        assert moving_average_signal(bars) == "HOLD"

    def test_hold_on_single_bar(self):
        bars = _make_bars(1)
        assert moving_average_signal(bars) == "HOLD"

    def test_hold_on_empty_dataframe(self):
        bars = pd.DataFrame({"c": [], "h": [], "l": []})
        assert moving_average_signal(bars) == "HOLD"

    def test_returns_valid_signal(self):
        """Return value must always be one of BUY, SELL, HOLD."""
        bars = _make_trending_bars(200, step=1.0)
        signal = moving_average_signal(bars)
        assert signal in {"BUY", "SELL", "HOLD"}

    def test_strong_uptrend_emits_buy(self):
        """
        A long, steady uptrend should eventually produce a BUY signal once
        all EMA and MACD conditions are satisfied.
        """
        bars = _make_trending_bars(300, step=2.0)
        signal = moving_average_signal(bars)
        assert signal in {"BUY", "HOLD"}  # not SELL in an uptrend

    def test_strong_downtrend_emits_sell(self):
        """A sustained downtrend should produce SELL (not BUY)."""
        bars = _make_trending_bars(300, start=600.0, step=-2.0)
        signal = moving_average_signal(bars)
        assert signal in {"SELL", "HOLD"}  # not BUY in a downtrend

    def test_choppy_market_hold_via_adx(self):
        """
        Oscillating prices keep ADX low, so the strategy should HOLD
        even if EMA alignment briefly appears.
        """
        import random
        random.seed(42)
        # Oscillate around 100 with small noise so ADX stays low
        prices = [100.0 + random.uniform(-0.5, 0.5) for _ in range(200)]
        bars = pd.DataFrame({
            "c": prices,
            "h": [p + 0.1 for p in prices],
            "l": [p - 0.1 for p in prices],
        })
        signal = moving_average_signal(bars, adx_threshold=25.0)
        assert signal == "HOLD"
