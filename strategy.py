import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (EWM, adjust=False = Wilder-style)."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> float:
    """Wilder RSI of the most recent value."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])


def moving_average_signal(
    closes: pd.Series,
    window: int = 20,           # kept for backward-compat; not used in EMA crossover
    confirm_bars: int = 1,
    fast_window: int = 9,
    slow_window: int = 21,
    rsi_window: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
) -> str:
    """
    EMA crossover signal gated by an RSI filter.

    Returns ``"BUY"``, ``"SELL"``, or ``"HOLD"``.

    A **BUY** fires when the fast EMA crossed *above* the slow EMA within the
    last ``confirm_bars`` bars AND RSI is below ``rsi_overbought`` (avoids
    buying an already-exhausted move).

    A **SELL** fires when the fast EMA crossed *below* the slow EMA within the
    last ``confirm_bars`` bars AND RSI is above ``rsi_oversold`` (avoids
    selling an already-exhausted move).

    ``confirm_bars=1`` (default) means only the most recent bar needs to show
    the crossover; raise it to require the signal to persist for multiple bars.
    """
    min_bars = max(slow_window, rsi_window) + confirm_bars + 1
    if len(closes) < min_bars:
        return "HOLD"

    fast_ema = _ema(closes, fast_window)
    slow_ema = _ema(closes, slow_window)

    # Look back confirm_bars+1 bars to detect whether a crossover occurred
    lookback = confirm_bars + 1
    fast_tail = fast_ema.tail(lookback)
    slow_tail = slow_ema.tail(lookback)

    prev_fast = float(fast_tail.iloc[0])
    prev_slow = float(slow_tail.iloc[0])
    curr_fast = float(fast_tail.iloc[-1])
    curr_slow = float(slow_tail.iloc[-1])

    rsi = _rsi(closes, window=rsi_window)

    logger.debug(
        "fast_ema=%.2f slow_ema=%.2f rsi=%.1f", curr_fast, curr_slow, rsi
    )

    # BUY: fast crossed above slow AND not yet overbought
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        if rsi < rsi_overbought:
            return "BUY"

    # SELL: fast crossed below slow AND not yet oversold
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        if rsi > rsi_oversold:
            return "SELL"

    return "HOLD"
