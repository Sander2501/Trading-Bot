"""
main
~~~~
Live trading entry-point.

Run with::

    python main.py

The bot connects to the Alpaca paper-trading API, fetches recent price bars,
generates an EMA-crossover + RSI signal, and submits market orders according
to the configured risk parameters.  It runs in an infinite loop, sleeping
``CHECK_INTERVAL_SECONDS`` between cycles, with exponential back-off on errors.
"""

import logging
import time
from datetime import date

from brokers import AlpacaBroker, BaseBroker
from config import (
    CHECK_INTERVAL_SECONDS,
    CONFIRM_BARS,
    FAST_WINDOW,
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
    WINDOW,
)
from strategy import moving_average_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Track the last calendar day on which snapshot_day() was called so the
# daily-loss baseline is refreshed exactly once per day.
_last_snapshot_date: date | None = None


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


def _maybe_snapshot_day(broker: BaseBroker) -> None:
    """Refresh the daily-loss baseline once per calendar day."""
    global _last_snapshot_date
    today = date.today()
    if _last_snapshot_date != today:
        broker.snapshot_day()
        _last_snapshot_date = today
        logger.info("Daily equity snapshot taken.")


# ------------------------------------------------------------------
# Core trading cycle
# ------------------------------------------------------------------


def run_once(broker: BaseBroker, sleep_enabled: bool = True) -> None:
    """
    Execute a single trading cycle:

    1. Refresh the daily-loss baseline if the calendar day has rolled over.
    2. Flush any stale per-cycle position cache.
    3. Skip if market is closed.
    4. Halt for the day if the daily-loss limit has been reached.
    5. Generate an EMA-crossover + RSI signal.
    6. Apply stop-loss, then act on BUY / SELL signals.
    """
    _maybe_snapshot_day(broker)
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

    # Fetch enough bars for EMA + RSI warm-up plus a small safety buffer
    bars_needed = max(SLOW_WINDOW, RSI_WINDOW) * 2 + CONFIRM_BARS + 5
    closes = broker.get_recent_closes(SYMBOL, limit=bars_needed, timeframe=TIMEFRAME)

    latest_price = float(closes.iloc[-1])
    signal = moving_average_signal(
        closes,
        window=WINDOW,
        confirm_bars=CONFIRM_BARS,
        fast_window=FAST_WINDOW,
        slow_window=SLOW_WINDOW,
        rsi_window=RSI_WINDOW,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
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

    # --- Stop-loss check ---
    if current_qty > 0:
        entry_price = broker.get_entry_price(SYMBOL)
        if entry_price and latest_price <= entry_price * (1 - STOP_LOSS_PCT):
            broker.submit_sell(SYMBOL, current_qty)
            logger.warning(
                f"STOP LOSS triggered: sold {current_qty:.6f} {SYMBOL} "
                f"at {latest_price:.2f} (entry {entry_price:.2f})"
            )
            if sleep_enabled:
                time.sleep(CHECK_INTERVAL_SECONDS)
            return

    # --- Signal execution ---
    equity, _ = broker.get_equity()

    if signal == "BUY" and current_qty == 0:
        qty = position_size(equity, latest_price)
        if qty <= 0:
            logger.info("Insufficient buying power to open a position.")
        else:
            broker.submit_buy(SYMBOL, qty)
            logger.info(f"BUY  {qty:.6f} {SYMBOL} @ ~{latest_price:.2f}")

    elif signal == "SELL" and current_qty > 0:
        broker.submit_sell(SYMBOL, current_qty)
        logger.info(f"SELL {current_qty:.6f} {SYMBOL} @ ~{latest_price:.2f}")

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
    broker = broker or AlpacaBroker()
    logger.info("Crypto bot started.")

    error_backoff = CHECK_INTERVAL_SECONDS

    while True:
        try:
            run_once(broker, sleep_enabled=True)
        except Exception as exc:
            logger.exception(f"Bot error: {exc}")
            time.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, MAX_BACKOFF_SECONDS)
            continue

        error_backoff = CHECK_INTERVAL_SECONDS  # reset on successful cycle


if __name__ == "__main__":
    run_bot()
