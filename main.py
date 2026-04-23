"""
main
~~~~
Live trading entry-point.

Run with::

    python main.py

The bot connects to the Capital.com API, fetches recent price bars,
generates an EMA-crossover + RSI signal, and submits market orders according
to the configured risk parameters.  It runs in an infinite loop, sleeping
``CHECK_INTERVAL_SECONDS`` between cycles, with exponential back-off on errors.
"""

import logging
import time
from datetime import date

from brokers import CapitalBroker, BaseBroker
from config import (
    ADX_THRESHOLD,
    ADX_WINDOW,
    ATR_STOP_MULT,
    ATR_STOP_WINDOW,
    CHECK_INTERVAL_SECONDS,
    CONFIRM_BARS,
    FAST_WINDOW,
    MACD_FAST,
    MACD_SIGNAL_WINDOW,
    MACD_SLOW,
    MAX_BACKOFF_SECONDS,
    MAX_DAILY_LOSS_PCT,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_WINDOW,
    RISK_PER_TRADE,
    SLOW_WINDOW,
    STOP_LOSS_PCT,
    SYMBOL,
    TIMEFRAME,
    TREND_WINDOW,
)
from strategy import atr_stop_distance, moving_average_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# State container
# ------------------------------------------------------------------


class TradingState:
    """
    Encapsulates mutable bot state that must persist across ``run_once`` cycles.

    Attributes
    ----------
    position_high:
        Highest price seen since the current long position was opened.
        Used to trail the ATR stop upward as the trade moves in our favour.
    position_low:
        Lowest price seen since the current short position was opened.
        Used to trail the ATR stop downward for short trades.
    last_snapshot_date:
        Calendar date on which ``snapshot_day()`` was last called so the
        daily-loss baseline is refreshed exactly once per day.
    """

    def __init__(self) -> None:
        self.position_high: float = 0.0
        self.position_low: float = float("inf")
        self.last_snapshot_date: date | None = None

    def reset_watermarks(self, price: float) -> None:
        """Seed both watermarks to *price* when a new position is opened."""
        self.position_high = price
        self.position_low = price


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def position_size(equity: float, price: float) -> float:
    """
    Calculate order quantity as ``RISK_PER_TRADE`` fraction of current equity.

    Using equity (rather than raw buying power) keeps position sizes stable
    as the account grows or shrinks over time.
    """
    if price <= 0:
        return 0.0
    qty = (equity * RISK_PER_TRADE) / price
    return round(max(0.0, qty), 6)  # 6 decimal places — safe for crypto


def daily_loss_exceeded(broker: BaseBroker) -> bool:
    """Return True when today's drawdown has reached ``MAX_DAILY_LOSS_PCT``."""
    equity, last_equity = broker.get_equity()
    if last_equity <= 0:
        return False
    return (last_equity - equity) / last_equity >= MAX_DAILY_LOSS_PCT


def _maybe_snapshot_day(broker: BaseBroker, state: TradingState) -> None:
    """Refresh the daily-loss baseline once per calendar day."""
    today = date.today()
    if state.last_snapshot_date != today:
        broker.snapshot_day()
        state.last_snapshot_date = today
        logger.info("Daily equity snapshot taken.")


# ------------------------------------------------------------------
# Core trading cycle
# ------------------------------------------------------------------


def run_once(
    broker: BaseBroker,
    state: TradingState | None = None,
    sleep_enabled: bool = True,
) -> None:
    """
    Execute a single trading cycle:

    1. Refresh the daily-loss baseline if the calendar day has rolled over.
    2. Flush any stale per-cycle position cache.
    3. Skip if market is closed.
    4. Halt for the day if the daily-loss limit has been reached.
    5. Generate an EMA-crossover + RSI signal.
    6. Apply stop-loss, then act on BUY / SELL signals.

    Parameters
    ----------
    broker:
        Broker instance (live or backtest).
    state:
        Mutable trading state for watermarks and daily snapshot tracking.
        A fresh ``TradingState`` is created automatically if *None* is passed,
        which preserves backward compatibility with callers that omit the argument.
    sleep_enabled:
        When ``False`` the function returns immediately instead of sleeping
        (useful for backtests and unit tests).
    """
    if state is None:
        state = TradingState()

    _maybe_snapshot_day(broker, state)
    broker.flush_position_cache()

    is_open, wait = broker.get_market_status()
    if not is_open:
        logger.info(f"Market is closed. Sleeping {wait:.0f}s until open.")
        if sleep_enabled:
            time.sleep(wait if wait > 0 else CHECK_INTERVAL_SECONDS)
        return

    if daily_loss_exceeded(broker):
        logger.warning(
            f"Daily loss limit ({MAX_DAILY_LOSS_PCT:.1%}) reached. "
            "Halting for the day."
        )
        if sleep_enabled:
            time.sleep(3600)
        return

    # Fetch enough bars for all indicators to warm up
    bars_needed = max(TREND_WINDOW, MACD_SLOW, RSI_WINDOW) * 2 + CONFIRM_BARS + 5
    bars = broker.get_recent_bars(SYMBOL, limit=bars_needed, timeframe=TIMEFRAME)

    latest_price = float(bars["c"].iloc[-1])
    signal = moving_average_signal(
        bars,
        fast_window=FAST_WINDOW,
        slow_window=SLOW_WINDOW,
        trend_window=TREND_WINDOW,
        confirm_bars=CONFIRM_BARS,
        rsi_window=RSI_WINDOW,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
        macd_fast=MACD_FAST,
        macd_slow=MACD_SLOW,
        macd_signal=MACD_SIGNAL_WINDOW,
        adx_window=ADX_WINDOW,
        adx_threshold=ADX_THRESHOLD,
    )
    current_qty = broker.get_position_qty(SYMBOL)
    open_order_exists = broker.has_open_order(SYMBOL)

    logger.info(
        f"{SYMBOL} | price={latest_price:.2f} | signal={signal} "
        f"| held={current_qty:.6f} | open_order={open_order_exists}"
    )

    if open_order_exists:
        logger.info("Open order already exists. Skipping cycle.")
        if sleep_enabled:
            time.sleep(CHECK_INTERVAL_SECONDS)
        return

    # --- Trailing ATR stop-loss ---
    # Watermarks follow price in the profitable direction so the stop moves
    # with the trade instead of staying pinned to the entry price.
    if current_qty > 0:
        state.position_high = max(state.position_high, latest_price)
    elif current_qty < 0:
        state.position_low = min(state.position_low, latest_price)
    else:
        state.reset_watermarks(latest_price)  # pre-seed for the position about to open

    if current_qty != 0:
        entry_price = broker.get_entry_price(SYMBOL)
        if entry_price:
            atr_stop = atr_stop_distance(bars["h"], bars["l"], bars["c"], window=ATR_STOP_WINDOW, multiplier=ATR_STOP_MULT)
            stop_floor = entry_price * STOP_LOSS_PCT
            stop_distance = max(atr_stop, stop_floor)

            # Trail from the watermark, not from entry — locks in profits on big moves
            long_stopped  = current_qty > 0 and latest_price <= state.position_high - stop_distance
            short_stopped = current_qty < 0 and latest_price >= state.position_low  + stop_distance

            if long_stopped or short_stopped:
                if current_qty > 0:
                    broker.submit_sell(SYMBOL, current_qty)
                    direction, ref = "long", state.position_high
                else:
                    broker.submit_buy(SYMBOL, abs(current_qty))
                    direction, ref = "short", state.position_low
                state.reset_watermarks(latest_price)
                logger.warning(
                    f"TRAILING STOP ({direction}): closed {abs(current_qty):.6f} {SYMBOL} "
                    f"at {latest_price:.2f} (peak/trough {ref:.2f}, "
                    f"stop_dist={stop_distance:.2f})"
                )
                if sleep_enabled:
                    time.sleep(CHECK_INTERVAL_SECONDS)
                return

    # --- Signal execution ---
    equity, _ = broker.get_equity()
    qty = position_size(equity, latest_price)

    if signal == "BUY":
        if current_qty > 0:
            logger.info("Already long — holding.")
        elif current_qty < 0:
            # Cover short first; next cycle opens the long if signal persists
            broker.submit_buy(SYMBOL, abs(current_qty))
            state.reset_watermarks(latest_price)
            logger.info(f"COVER {abs(current_qty):.6f} {SYMBOL} @ ~{latest_price:.2f}")
        else:
            if qty <= 0:
                logger.info("Insufficient buying power to open long.")
            else:
                broker.submit_buy(SYMBOL, qty)
                state.position_high = latest_price
                logger.info(f"BUY   {qty:.6f} {SYMBOL} @ ~{latest_price:.2f}")

    elif signal == "SELL":
        if current_qty < 0:
            logger.info("Already short — holding.")
        elif current_qty > 0:
            # Close long first; next cycle opens the short if signal persists
            broker.submit_sell(SYMBOL, current_qty)
            state.reset_watermarks(latest_price)
            logger.info(f"SELL  {current_qty:.6f} {SYMBOL} @ ~{latest_price:.2f}")
        else:
            if not broker.supports_shorting:
                logger.info("SELL signal — broker does not support shorting, staying flat.")
            elif qty <= 0:
                logger.info("Insufficient equity to open short.")
            else:
                broker.submit_sell(SYMBOL, qty)
                state.position_low = latest_price
                logger.info(f"SHORT {qty:.6f} {SYMBOL} @ ~{latest_price:.2f}")

    else:
        logger.info("No action.")

    if sleep_enabled:
        time.sleep(CHECK_INTERVAL_SECONDS)


# ------------------------------------------------------------------
# Bot runner
# ------------------------------------------------------------------


def run_bot(broker: BaseBroker | None = None) -> None:
    """
    Start the live trading loop.

    Accepts an optional ``broker`` argument to allow dependency injection
    (useful for testing with a BacktestBroker).
    """
    if broker is None:
        from brokers import CapitalBroker
        broker = CapitalBroker()

    logger.info(f"Crypto bot started using {broker.__class__.__name__}.")

    state = TradingState()
    error_backoff = CHECK_INTERVAL_SECONDS

    while True:
        try:
            run_once(broker, state, sleep_enabled=True)
        except Exception as exc:
            logger.exception(f"Bot error: {exc}")
            time.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, MAX_BACKOFF_SECONDS)
            continue

        error_backoff = CHECK_INTERVAL_SECONDS  # reset on successful cycle


if __name__ == "__main__":
    run_bot()
