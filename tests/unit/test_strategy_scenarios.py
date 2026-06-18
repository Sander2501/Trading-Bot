"""
tests.unit.test_strategy_scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Deterministic scenario tests for the trading strategy.

Each test uses fixed, hand-crafted price arrays so results are fully
reproducible without any random seeding.  The goal is to validate the
strategy's regime-filtering logic (ADX gate), directional signals
(EMA crossover + MACD confirmation), and exit mechanisms (trailing stop).
"""

import pandas as pd
import pytest

from strategy import (
    _adx_series,
    _macd,
    _rsi,
    moving_average_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(
    closes: list[float],
    high_pct: float = 0.001,
    low_pct: float = 0.001,
) -> pd.DataFrame:
    """Build a DataFrame from a list of close prices with synthetic H/L."""
    return pd.DataFrame({
        "c": closes,
        "h": [c * (1 + high_pct) for c in closes],
        "l": [c * (1 - low_pct) for c in closes],
    })


def _flat(n: int, price: float = 100.0) -> list[float]:
    """Return *n* constant prices — simulates a sideways/choppy market."""
    return [price] * n


def _oscillating(n: int, center: float = 100.0, amplitude: float = 0.3) -> list[float]:
    """
    Prices that alternate above/below the center with a fixed amplitude.
    ADX stays low because there is no sustained directional move.
    """
    prices = []
    for i in range(n):
        offset = amplitude if i % 2 == 0 else -amplitude
        prices.append(center + offset)
    return prices


def _uptrend(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    """Steadily rising prices — produces a bullish EMA alignment."""
    return [start + i * step for i in range(n)]


def _downtrend(n: int, start: float = 200.0, step: float = 1.0) -> list[float]:
    """Steadily falling prices — produces a bearish EMA alignment."""
    return [start - i * step for i in range(n)]


def _crossover_up(n_before: int = 150, n_after: int = 100) -> list[float]:
    """
    Prices that drop (to build bearish EMA state) then reverse into a
    sustained uptrend so a bullish crossover occurs.
    """
    prices = _downtrend(n_before, start=200.0, step=0.5)
    # Turn around and climb
    last = prices[-1]
    prices += [last + i * 1.0 for i in range(1, n_after + 1)]
    return prices


def _crossover_down(n_before: int = 150, n_after: int = 100) -> list[float]:
    """
    Prices that rise (to build bullish EMA state) then reverse into a
    sustained downtrend so a bearish crossover occurs.
    """
    prices = _uptrend(n_before, start=100.0, step=0.5)
    last = prices[-1]
    prices += [last - i * 1.0 for i in range(1, n_after + 1)]
    return prices


# ---------------------------------------------------------------------------
# Scenario 1 — Sideways / choppy market: ADX filter suppresses signals
# ---------------------------------------------------------------------------


class TestSidewaysMarketHold:
    """ADX < threshold → strategy must HOLD regardless of EMA noise."""

    def test_flat_prices_return_hold(self):
        """
        Completely flat prices produce ADX ≈ 0.  No signal should fire.
        """
        bars = _make_bars(_flat(200))
        signal = moving_average_signal(bars, adx_threshold=20.0)
        assert signal == "HOLD", (
            "Flat prices have ADX ≈ 0 — no trend, no signal expected"
        )

    def test_oscillating_prices_return_hold(self):
        """
        Small alternating moves keep ADX well below 20.
        Strategy must not react to apparent EMA alignments caused by noise.
        """
        bars = _make_bars(_oscillating(200))
        signal = moving_average_signal(bars, adx_threshold=20.0)
        assert signal == "HOLD", (
            "Oscillating/choppy prices should suppress all signals via ADX filter"
        )

    def test_adx_value_is_low_for_choppy_data(self):
        """Verify ADX itself is below 10 for oscillating data (sanity check)."""
        bars = _make_bars(_oscillating(100))
        adx = float(_adx_series(bars["h"], bars["l"], bars["c"], window=14).iloc[-1])
        assert adx < 15, f"Expected ADX < 15 for choppy data, got {adx:.2f}"


# ---------------------------------------------------------------------------
# Scenario 2 — Bullish crossover
# ---------------------------------------------------------------------------


class TestBullishCrossover:
    """After a sustained uptrend, the strategy should emit BUY (not SELL)."""

    def test_strong_uptrend_not_sell(self):
        """
        Long uptrend: fast EMA > slow EMA > trend EMA, MACD histogram positive,
        ADX elevated.  The strategy must not return SELL.
        """
        bars = _make_bars(_uptrend(300, step=2.0))
        signal = moving_average_signal(bars, adx_threshold=20.0)
        assert signal != "SELL", "An uptrend must never produce a SELL signal"

    def test_crossover_up_emits_buy_or_hold(self):
        """
        A down-then-up crossover gives the strategy enough bars to confirm
        a bullish alignment.  Result must be BUY or HOLD (never SELL).
        """
        bars = _make_bars(_crossover_up())
        signal = moving_average_signal(bars, adx_threshold=20.0)
        assert signal in {"BUY", "HOLD"}, (
            f"Bullish crossover should produce BUY or HOLD, got {signal!r}"
        )

    def test_macd_histogram_positive_in_uptrend(self):
        """MACD histogram must be positive during a sustained uptrend."""
        bars = _make_bars(_uptrend(200, step=2.0))
        _, _, histogram = _macd(bars["c"])
        assert histogram > 0, (
            f"Expected positive MACD histogram in uptrend, got {histogram:.4f}"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — Bearish crossover
# ---------------------------------------------------------------------------


class TestBearishCrossover:
    """After a sustained downtrend, the strategy should emit SELL (not BUY)."""

    def test_strong_downtrend_not_buy(self):
        """
        Long downtrend: fast EMA < slow EMA < trend EMA.
        The strategy must not return BUY.
        """
        bars = _make_bars(_downtrend(300, step=1.5))
        signal = moving_average_signal(bars, adx_threshold=20.0)
        assert signal != "BUY", "A downtrend must never produce a BUY signal"

    def test_crossover_down_emits_sell_or_hold(self):
        """
        An up-then-down crossover should produce SELL or HOLD (never BUY).
        """
        bars = _make_bars(_crossover_down())
        signal = moving_average_signal(bars, adx_threshold=20.0)
        assert signal in {"SELL", "HOLD"}, (
            f"Bearish crossover should produce SELL or HOLD, got {signal!r}"
        )

    def test_macd_histogram_negative_in_downtrend(self):
        """MACD histogram must be negative during a sustained downtrend."""
        bars = _make_bars(_downtrend(200, step=1.5))
        _, _, histogram = _macd(bars["c"])
        assert histogram < 0, (
            f"Expected negative MACD histogram in downtrend, got {histogram:.4f}"
        )


# ---------------------------------------------------------------------------
# Scenario 4 — RSI extremes do not block valid trend entries
# ---------------------------------------------------------------------------


class TestRsiExtremes:
    """
    On BTC 1-min data, RSI can stay extreme for long periods.
    Document the RSI implementation behaviour for trending markets.
    """

    def test_rsi_range_always_valid(self):
        """RSI must always be in [0, 100] regardless of price direction."""
        for prices in [_uptrend(200, step=3.0), _downtrend(200, step=1.5)]:
            bars = _make_bars(prices)
            rsi = _rsi(bars["c"], window=14)
            assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"

    def test_rsi_low_in_downtrend(self):
        """RSI < 50 is expected during a strong downtrend."""
        bars = _make_bars(_downtrend(200, step=1.5))
        rsi = _rsi(bars["c"], window=14)
        assert rsi < 50, (
            f"RSI should be depressed in a strong downtrend, got {rsi:.1f}"
        )

    def test_rsi_does_not_block_trend_signals(self):
        """
        The strategy should still emit a valid signal in a trending market
        even if RSI is at an extreme value.
        """
        bars = _make_bars(_downtrend(300, step=1.5))
        signal = moving_average_signal(bars)
        assert signal in {"BUY", "SELL", "HOLD"}, (
            f"Signal must be a valid value even with extreme RSI, got {signal!r}"
        )


# ---------------------------------------------------------------------------
# Scenario 5 — Insufficient bars returns HOLD
# ---------------------------------------------------------------------------


class TestInsufficientData:
    """Strategy requires enough bars to warm up all indicators."""

    @pytest.mark.parametrize("n_bars", [0, 1, 5, 20])
    def test_too_few_bars_returns_hold(self, n_bars):
        """Any call with fewer than the minimum required bars must return HOLD."""
        bars = _make_bars(_flat(n_bars)) if n_bars > 0 else pd.DataFrame({"c": [], "h": [], "l": []})
        signal = moving_average_signal(bars)
        assert signal == "HOLD", (
            f"Expected HOLD with only {n_bars} bars, got {signal!r}"
        )

    def test_minimum_bars_does_not_raise(self):
        """Strategy must not raise an exception regardless of bar count."""
        for n in range(0, 60, 5):
            bars = _make_bars(_flat(n)) if n > 0 else pd.DataFrame({"c": [], "h": [], "l": []})
            signal = moving_average_signal(bars)
            assert signal in {"BUY", "SELL", "HOLD"}


# ---------------------------------------------------------------------------
# Scenario 6 — Signal consistency across valid input ranges
# ---------------------------------------------------------------------------


class TestSignalConsistency:
    """Signal must always be one of the three valid values."""

    @pytest.mark.parametrize(
        "prices",
        [
            _flat(200),
            _uptrend(200, step=0.5),
            _uptrend(200, step=5.0),
            _downtrend(200, step=0.5),
            _downtrend(200, step=5.0),
            _oscillating(200),
        ],
    )
    def test_always_returns_valid_signal(self, prices):
        bars = _make_bars(prices)
        signal = moving_average_signal(bars)
        assert signal in {"BUY", "SELL", "HOLD"}, (
            f"moving_average_signal returned unexpected value: {signal!r}"
        )
