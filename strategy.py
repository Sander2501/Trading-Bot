import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return float((100 - (100 / (1 + rs))).iloc[-1])


def _macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram) at the final bar."""
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])


def _adx(series: pd.Series, window: int = 14) -> float:
    """
    Closes-only ADX approximation.

    Without high/low data, directional movement is estimated from
    close-to-close changes: positive moves feed DM+, negative DM-.
    The result correlates well with true ADX for trend-strength filtering.
    """
    delta = series.diff()
    dm_plus = delta.clip(lower=0)
    dm_minus = (-delta).clip(lower=0)

    atr = delta.abs().ewm(com=window - 1, adjust=False).mean()
    safe_atr = atr.replace(0, float("inf"))

    di_plus = dm_plus.ewm(com=window - 1, adjust=False).mean() / safe_atr * 100
    di_minus = dm_minus.ewm(com=window - 1, adjust=False).mean() / safe_atr * 100

    di_sum = (di_plus + di_minus).replace(0, float("inf"))
    dx = (di_plus - di_minus).abs() / di_sum * 100
    adx = dx.ewm(com=window - 1, adjust=False).mean()
    return float(adx.iloc[-1])


def atr_stop_distance(series: pd.Series, window: int = 14, multiplier: float = 2.0) -> float:
    """
    Return an ATR-based stop distance in price units.

    Caller adds this to or subtracts it from entry price to get the stop level.
    Uses the same closes-only ATR as the ADX calculation.
    """
    atr = series.diff().abs().ewm(com=window - 1, adjust=False).mean()
    return float(atr.iloc[-1]) * multiplier


def moving_average_signal(
    closes: pd.Series,
    window: int = 20,               # kept for backward-compat; unused
    confirm_bars: int = 2,
    fast_window: int = 9,
    slow_window: int = 21,
    trend_window: int = 50,
    rsi_window: int = 14,
    rsi_overbought: float = 75.0,
    rsi_oversold: float = 25.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
) -> str:
    """
    Multi-indicator signal tuned for BTC/USD.

    BUY requires ALL of:
      - Fast EMA crossed above slow EMA within the last ``confirm_bars``
      - Triple EMA structure: fast > slow > trend  (bull regime)
      - MACD histogram positive  (momentum confirming upside)
      - RSI below ``rsi_overbought``  (move not exhausted)
      - ADX above ``adx_threshold``  (trending, not choppy)

    SELL requires ALL of:
      - Fast EMA crossed below slow EMA within ``confirm_bars``
      - Triple EMA structure: fast < slow < trend  (bear regime)
      - MACD histogram negative  (momentum confirming downside)
      - RSI above ``rsi_oversold``  (not already washed out)
      - ADX above ``adx_threshold``  (confirmed trend direction)

    Requiring all conditions simultaneously produces fewer but higher-conviction
    trades, which matters most on noisy 1-minute BTC data.
    """
    min_bars = max(trend_window, macd_slow, rsi_window) + confirm_bars + 5
    if len(closes) < min_bars:
        return "HOLD"

    fast_ema = _ema(closes, fast_window)
    slow_ema = _ema(closes, slow_window)
    trend_ema = _ema(closes, trend_window)

    curr_fast = float(fast_ema.iloc[-1])
    curr_slow = float(slow_ema.iloc[-1])
    curr_trend = float(trend_ema.iloc[-1])

    # Crossover detection over the confirm_bars window
    prev_fast = float(fast_ema.iloc[-(confirm_bars + 1)])
    prev_slow = float(slow_ema.iloc[-(confirm_bars + 1)])

    rsi = _rsi(closes, window=rsi_window)
    _, _, macd_hist = _macd(closes, macd_fast, macd_slow, macd_signal)
    adx = _adx(closes, adx_window)

    logger.info(
        "indicators | fast=%.2f slow=%.2f trend=%.2f | rsi=%.1f macd_hist=%.2f adx=%.1f",
        curr_fast, curr_slow, curr_trend, rsi, macd_hist, adx,
    )

    # Regime gate: flat/choppy markets produce too many false crossovers
    if adx < adx_threshold:
        logger.info("HOLD reason: ADX %.1f < threshold %.1f (choppy market)", adx, adx_threshold)
        return "HOLD"

    bull_aligned = curr_fast > curr_slow > curr_trend
    crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
    if crossed_up and bull_aligned and macd_hist > 0 and rsi < rsi_overbought:
        return "BUY"

    bear_aligned = curr_fast < curr_slow < curr_trend
    crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow
    if crossed_down and bear_aligned and macd_hist < 0 and rsi > rsi_oversold:
        return "SELL"

    # Log which specific conditions are blocking a signal
    crossed_up_any = prev_fast <= prev_slow and curr_fast > curr_slow
    crossed_dn_any = prev_fast >= prev_slow and curr_fast < curr_slow
    if crossed_up_any or crossed_dn_any:
        direction = "UP" if crossed_up_any else "DN"
        logger.info(
            "HOLD reason: crossover %s detected but conditions not met — "
            "bull_aligned=%s bear_aligned=%s macd_hist=%.2f rsi=%.1f",
            direction, bull_aligned,
            curr_fast < curr_slow < curr_trend,
            macd_hist, rsi,
        )
    else:
        ema_state = (
            "fast>slow>trend (bull)" if bull_aligned
            else "fast<slow<trend (bear)" if curr_fast < curr_slow < curr_trend
            else f"mixed (fast={'>' if curr_fast > curr_slow else '<'}slow, slow={'>' if curr_slow > curr_trend else '<'}trend)"
        )
        logger.info("HOLD reason: no crossover — EMA structure: %s", ema_state)

    return "HOLD"
