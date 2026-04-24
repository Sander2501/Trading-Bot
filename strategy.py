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
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    min_atr_pct: float = 0.001,
) -> str:
    """
    Strategy with Market Regime Filter, RSI Entry Timing, and confirmed crossovers.

    Regimes:
    1. TRENDING (ADX > 25): Follow EMA crossovers with MACD + RSI confirmation.
    2. RANGING (ADX 20-25): Use RSI mean-reversion within the trend.
    3. SIDEWAYS (ADX < threshold): Stay flat.

    ``confirm_bars`` controls how many consecutive bars the fast/slow EMA
    crossover must have been in place before a signal fires.  A value of 2
    means the crossover must have held for at least 2 bars (current bar + one
    preceding bar) while the bar before that was on the opposite side.  This
    eliminates single-bar false crossovers without adding meaningful lag.
    """
    closes = bars["c"]
    highs  = bars["h"]
    lows   = bars["l"]

    # Need confirm_bars + 1 extra to check the bar before the crossover
    min_bars = max(trend_window, macd_slow, rsi_window) + confirm_bars + 5
    if len(closes) < min_bars:
        return "HOLD"

    latest_close = float(closes.iloc[-1])
    fast_ema  = _ema(closes, fast_window)
    slow_ema  = _ema(closes, slow_window)
    trend_ema = _ema(closes, trend_window)

    curr_fast  = float(fast_ema.iloc[-1])
    curr_slow  = float(slow_ema.iloc[-1])
    curr_trend = float(trend_ema.iloc[-1])

    rsi = _rsi(closes, window=rsi_window)
    _, _, macd_hist = _macd(closes, macd_fast, macd_slow, macd_signal)
    adx = _adx(highs, lows, closes, adx_window)
    atr = _true_range(highs, lows, closes).ewm(com=adx_window - 1, adjust=False).mean()
    atr_pct = float(atr.iloc[-1]) / latest_close if latest_close > 0 else 0.0

    if atr_pct < min_atr_pct:
        logger.info("REGIME: LOW_VOL (ATR%% %.4f < %.4f) | Skipping.", atr_pct, min_atr_pct)
        return "HOLD"

    if adx < adx_threshold:
        logger.info("REGIME: SIDEWAYS (ADX %.1f < %.1f) | Skipping.", adx, adx_threshold)
        return "HOLD"

    trend  = "BULL" if latest_close > curr_trend else "BEAR"
    regime = "TRENDING" if adx > 25 else "RANGING"

    logger.info(
        "REGIME: %s (%s) | fast=%.2f slow=%.2f trend=%.2f | rsi=%.1f macd_hist=%.2f adx=%.1f atr%%=%.3f",
        regime, trend, curr_fast, curr_slow, curr_trend, rsi, macd_hist, adx, atr_pct * 100,
    )

    # Helpers: check that the crossover has been held for confirm_bars bars and
    # that the bar before that was on the opposite side (fresh crossover).
    def _bull_crossover_confirmed() -> bool:
        held = all(
            float(fast_ema.iloc[-(i + 1)]) > float(slow_ema.iloc[-(i + 1)])
            for i in range(confirm_bars)
        )
        was_below = float(fast_ema.iloc[-(confirm_bars + 1)]) <= float(slow_ema.iloc[-(confirm_bars + 1)])
        return held and was_below

    def _bear_crossover_confirmed() -> bool:
        held = all(
            float(fast_ema.iloc[-(i + 1)]) < float(slow_ema.iloc[-(i + 1)])
            for i in range(confirm_bars)
        )
        was_above = float(fast_ema.iloc[-(confirm_bars + 1)]) >= float(slow_ema.iloc[-(confirm_bars + 1)])
        return held and was_above

    if trend == "BULL":
        is_ema_bull  = curr_fast > curr_slow
        is_macd_bull = macd_hist > 0
        is_rsi_dip   = rsi < 45

        if regime == "TRENDING":
            if is_ema_bull and is_macd_bull and is_rsi_dip:
                logger.info("SIGNAL: TREND BUY (EMA Bull + MACD Bull + RSI Dip)")
                return "BUY"
            if _bull_crossover_confirmed():
                logger.info("SIGNAL: CONFIRMED CROSSOVER BUY (%d bars)", confirm_bars)
                return "BUY"
        else:
            if rsi <= rsi_oversold:
                logger.info("SIGNAL: RSI OVERSOLD BUY (Ranging)")
                return "BUY"

    elif trend == "BEAR":
        is_ema_bear  = curr_fast < curr_slow
        is_macd_bear = macd_hist < 0
        is_rsi_spike = rsi > 55

        if regime == "TRENDING":
            if is_ema_bear and is_macd_bear and is_rsi_spike:
                logger.info("SIGNAL: TREND SELL (EMA Bear + MACD Bear + RSI Spike)")
                return "SELL"
            if _bear_crossover_confirmed():
                logger.info("SIGNAL: CONFIRMED CROSSOVER SELL (%d bars)", confirm_bars)
                return "SELL"
        else:
            if rsi >= rsi_overbought:
                logger.info("SIGNAL: RSI OVERBOUGHT SELL (Ranging)")
                return "SELL"

    return "HOLD"
