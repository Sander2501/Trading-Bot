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
    # NaN-then-fillna is intentional: replacing avg_loss=0 directly with inf
    # would make the division yield 0 (gain/inf=0 in pandas), producing RSI=0
    # for all-gain series.  Instead, replace 0 with NaN so the division
    # propagates NaN, then fill those NaN entries with inf so that the
    # RSI formula yields 100 (= 100 - 100/(1+inf)) as intended.
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return float((100 - (100 / (1 + rs.fillna(float("inf"))))).iloc[-1])


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


def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Return full ADX series so adaptive thresholds can be percentile-based."""
    tr = _true_range(high, low, close)
    atr = tr.ewm(com=window - 1, adjust=False).mean()

    up_move = high.diff()
    down_move = -(low.diff())
    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    safe_atr = atr.replace(0, float("inf"))
    di_plus = dm_plus.ewm(com=window - 1, adjust=False).mean() / safe_atr * 100
    di_minus = dm_minus.ewm(com=window - 1, adjust=False).mean() / safe_atr * 100
    di_sum = (di_plus + di_minus).replace(0, float("inf"))
    dx = (di_plus - di_minus).abs() / di_sum * 100
    return dx.ewm(com=window - 1, adjust=False).mean()


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
    adx_trending_threshold: float = 25.0,
    min_atr_pct: float = 0.001,
    adaptive_lookback_bars: int = 0,
    adx_threshold_percentile: float = 60.0,
    min_atr_pct_percentile: float = 35.0,
    session_filter_enabled: bool = False,
    session_start_hour_utc: int = 0,
    session_end_hour_utc: int = 24,
    atr_accel_window: int = 20,
    min_atr_accel: float = 0.0,
    structure_filter_enabled: bool = False,
    structure_lookback: int = 5,
    volume_filter_enabled: bool = False,
    min_volume: float = 0.0,
) -> str:
    """
    Strategy with Market Regime Filter, RSI Entry Timing, and confirmed crossovers.

    Regimes:
    1. TRENDING (ADX >= adx_trending_threshold): Follow EMA crossovers with MACD + RSI
       confirmation.  RSI must be above 50 (bull) or below 50 (bear) to confirm that
       momentum direction agrees with the EMA/MACD alignment.
    2. RANGING (adx_threshold <= ADX < adx_trending_threshold): Use RSI mean-reversion
       against the trend (oversold for longs, overbought for shorts).
    3. SIDEWAYS (ADX < adx_threshold): Stay flat — no signal.

    ``confirm_bars`` controls how many consecutive bars the fast/slow EMA
    crossover must have been in place before a signal fires.  A value of 2
    means the crossover must have held for at least 2 bars (current bar + one
    preceding bar) while the bar before that was on the opposite side.  This
    eliminates single-bar false crossovers without adding meaningful lag.
    """
    closes = bars["c"]
    highs  = bars["h"]
    lows   = bars["l"]
    ts = bars.get("t")
    vols = bars.get("v")

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
    adx_s = _adx_series(highs, lows, closes, adx_window)
    adx = float(adx_s.iloc[-1])
    atr = _true_range(highs, lows, closes).ewm(com=adx_window - 1, adjust=False).mean()
    atr_pct_series = (atr / closes.replace(0, float("nan"))).fillna(0.0)
    atr_pct = float(atr_pct_series.iloc[-1]) if latest_close > 0 else 0.0

    adx_threshold_eff = adx_threshold
    min_atr_pct_eff = min_atr_pct
    if adaptive_lookback_bars and adaptive_lookback_bars > 1:
        lookback = min(int(adaptive_lookback_bars), len(closes))
        adx_threshold_eff = float(adx_s.tail(lookback).quantile(adx_threshold_percentile / 100.0))
        min_atr_pct_eff = float(atr_pct_series.tail(lookback).quantile(min_atr_pct_percentile / 100.0))

    if atr_pct < min_atr_pct_eff:
        logger.info("REGIME: LOW_VOL (ATR%% %.4f < %.4f) | Skipping.", atr_pct, min_atr_pct_eff)
        return "HOLD"

    # Determine regime before the early-return guard so the RANGING band is
    # reachable.  Ensure the trending boundary is never lower than the ranging
    # boundary (important when adaptive mode raises adx_threshold_eff).
    adx_trending_eff = max(adx_trending_threshold, adx_threshold_eff)
    trend  = "BULL" if latest_close > curr_trend else "BEAR"
    if adx >= adx_trending_eff:
        regime = "TRENDING"
    elif adx >= adx_threshold_eff:
        regime = "RANGING"
    else:
        regime = "SIDEWAYS"

    if regime == "SIDEWAYS":
        logger.info("REGIME: SIDEWAYS (ADX %.1f < %.1f) | Skipping.", adx, adx_threshold_eff)
        return "HOLD"

    if session_filter_enabled and ts is not None:
        try:
            hour = pd.to_datetime(ts.iloc[-1], utc=True).hour
            if not (session_start_hour_utc <= hour < session_end_hour_utc):
                logger.info(
                    "REGIME: OUT_OF_SESSION (hour=%d, allowed=[%d,%d)) | Skipping.",
                    hour,
                    session_start_hour_utc,
                    session_end_hour_utc,
                )
                return "HOLD"
        except Exception:
            pass

    if atr_accel_window > 1 and min_atr_accel > 0:
        atr_baseline = float(atr_pct_series.tail(atr_accel_window).mean())
        atr_accel = (atr_pct / atr_baseline) if atr_baseline > 0 else 0.0
        if atr_accel < min_atr_accel:
            logger.info(
                "REGIME: LOW_ATR_ACCEL (%.3f < %.3f) | Skipping.",
                atr_accel,
                min_atr_accel,
            )
            return "HOLD"

    if volume_filter_enabled and vols is not None:
        try:
            latest_vol = float(vols.iloc[-1])
            if latest_vol < min_volume:
                logger.info("REGIME: LOW_VOLUME (%.2f < %.2f) | Skipping.", latest_vol, min_volume)
                return "HOLD"
        except Exception:
            pass

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

    if structure_filter_enabled and len(highs) >= structure_lookback + 1:
        recent_highs = highs.tail(structure_lookback + 1)
        recent_lows = lows.tail(structure_lookback + 1)
        bull_structure = bool(recent_highs.iloc[-1] > recent_highs.iloc[0] and recent_lows.iloc[-1] > recent_lows.iloc[0])
        bear_structure = bool(recent_highs.iloc[-1] < recent_highs.iloc[0] and recent_lows.iloc[-1] < recent_lows.iloc[0])
    else:
        bull_structure = True
        bear_structure = True

    if trend == "BULL":
        is_ema_bull  = curr_fast > curr_slow
        is_macd_bull = macd_hist > 0
        is_rsi_bull  = rsi > 50  # momentum confirms direction (was: rsi < 45 — contradicted MACD bull)

        if regime == "TRENDING":
            if is_ema_bull and is_macd_bull and is_rsi_bull and bull_structure:
                logger.info("SIGNAL: TREND BUY (EMA Bull + MACD Bull + RSI Bull)")
                return "BUY"
            if _bull_crossover_confirmed() and bull_structure:
                logger.info("SIGNAL: CONFIRMED CROSSOVER BUY (%d bars)", confirm_bars)
                return "BUY"
        else:
            if rsi <= rsi_oversold:
                logger.info("SIGNAL: RSI OVERSOLD BUY (Ranging)")
                return "BUY"

    elif trend == "BEAR":
        is_ema_bear  = curr_fast < curr_slow
        is_macd_bear = macd_hist < 0
        is_rsi_bear  = rsi < 50  # momentum confirms direction (was: rsi > 55 — contradicted MACD bear)

        if regime == "TRENDING":
            if is_ema_bear and is_macd_bear and is_rsi_bear and bear_structure:
                logger.info("SIGNAL: TREND SELL (EMA Bear + MACD Bear + RSI Bear)")
                return "SELL"
            if _bear_crossover_confirmed() and bear_structure:
                logger.info("SIGNAL: CONFIRMED CROSSOVER SELL (%d bars)", confirm_bars)
                return "SELL"
        else:
            if rsi >= rsi_overbought:
                logger.info("SIGNAL: RSI OVERBOUGHT SELL (Ranging)")
                return "SELL"

    return "HOLD"
