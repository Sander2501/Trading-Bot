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


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    """True ATR-based ADX using high, low, close bars."""
    tr = _true_range(high, low, close)
    atr = tr.ewm(com=window - 1, adjust=False).mean()

    up_move   = high.diff()
    down_move = -(low.diff())
    dm_plus  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    safe_atr = atr.replace(0, float("inf"))
    di_plus  = dm_plus.ewm(com=window - 1, adjust=False).mean()  / safe_atr * 100
    di_minus = dm_minus.ewm(com=window - 1, adjust=False).mean() / safe_atr * 100
    di_sum   = (di_plus + di_minus).replace(0, float("inf"))
    dx       = (di_plus - di_minus).abs() / di_sum * 100
    adx      = dx.ewm(com=window - 1, adjust=False).mean()
    return float(adx.iloc[-1])


def atr_stop_distance(
    high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 14, multiplier: float = 2.0
) -> float:
    """Return an ATR-based stop distance in price units using true range."""
    atr = _true_range(high, low, close).ewm(com=window - 1, adjust=False).mean()
    return float(atr.iloc[-1]) * multiplier


def moving_average_signal(
    bars: pd.DataFrame,
    fast_window: int = 9,
    slow_window: int = 21,
    trend_window: int = 50,
    confirm_bars: int = 2,
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
    State-based trend-following signal tuned for BTC/USD.

    BUY when ALL of:
      - Triple EMA aligned bull (fast > slow > trend) for the last ``confirm_bars``
        consecutive bars — single-bar alignments are usually fakeouts on 5-min data
      - MACD histogram positive  (momentum confirms upside)
      - ADX > adx_threshold      (genuine trend, not sideways chop)

    SELL when ALL of:
      - Triple EMA aligned bear (fast < slow < trend) for ``confirm_bars`` bars
      - MACD histogram negative
      - ADX > adx_threshold

    RSI is logged for context but is NOT a hard entry gate.  During strong BTC
    trends RSI stays above 80 (bull) or below 20 (bear) for hours; using it as
    a gate caused the strategy to miss every major move.  ADX already filters
    low-momentum, choppy environments.
    """
    closes = bars["c"]
    highs  = bars["h"]
    lows   = bars["l"]

    min_bars = max(trend_window, macd_slow, rsi_window) + confirm_bars + 5
    if len(closes) < min_bars:
        return "HOLD"

    fast_ema  = _ema(closes, fast_window)
    slow_ema  = _ema(closes, slow_window)
    trend_ema = _ema(closes, trend_window)

    curr_fast  = float(fast_ema.iloc[-1])
    curr_slow  = float(slow_ema.iloc[-1])
    curr_trend = float(trend_ema.iloc[-1])

    rsi = _rsi(closes, window=rsi_window)
    _, _, macd_hist = _macd(closes, macd_fast, macd_slow, macd_signal)
    adx = _adx(highs, lows, closes, adx_window)

    rsi_tag = " [OVERBOUGHT]" if rsi >= rsi_overbought else " [OVERSOLD]" if rsi <= rsi_oversold else ""
    logger.info(
        "indicators | fast=%.2f slow=%.2f trend=%.2f | rsi=%.1f%s macd_hist=%.2f adx=%.1f",
        curr_fast, curr_slow, curr_trend, rsi, rsi_tag, macd_hist, adx,
    )

    if adx < adx_threshold:
        logger.info("HOLD reason: ADX %.1f < threshold %.1f (choppy market)", adx, adx_threshold)
        return "HOLD"

    bull_now  = curr_fast > curr_slow > curr_trend
    bear_now  = curr_fast < curr_slow < curr_trend

    # Require alignment to have held for `confirm_bars` consecutive prior bars
    # so the strategy only enters after a trend is confirmed, not on the first bar.
    if confirm_bars > 1:
        lookback_f = fast_ema.iloc[-(confirm_bars + 1):-1]
        lookback_s = slow_ema.iloc[-(confirm_bars + 1):-1]
        lookback_t = trend_ema.iloc[-(confirm_bars + 1):-1]
        bull_persistent = bull_now and all(
            f > s > t for f, s, t in zip(lookback_f, lookback_s, lookback_t)
        )
        bear_persistent = bear_now and all(
            f < s < t for f, s, t in zip(lookback_f, lookback_s, lookback_t)
        )
    else:
        bull_persistent = bull_now
        bear_persistent = bear_now

    if bull_persistent and macd_hist > 0:
        return "BUY"

    if bear_persistent and macd_hist < 0:
        return "SELL"

    # Log the specific blocker
    if bull_now and not bull_persistent:
        logger.info("HOLD reason: bull alignment is NEW (needs %d consecutive bars)", confirm_bars)
    elif bull_persistent:
        logger.info("HOLD reason: bull EMA aligned but MACD hist=%.2f (needs >0)", macd_hist)
    elif bear_now and not bear_persistent:
        logger.info("HOLD reason: bear alignment is NEW (needs %d consecutive bars)", confirm_bars)
    elif bear_persistent:
        logger.info("HOLD reason: bear EMA aligned but MACD hist=%.2f (needs <0)", macd_hist)
    else:
        ema_state = (
            f"mixed (fast={'>' if curr_fast > curr_slow else '<'}slow, "
            f"slow={'>' if curr_slow > curr_trend else '<'}trend)"
        )
        logger.info("HOLD reason: EMA not aligned — %s", ema_state)

    return "HOLD"
