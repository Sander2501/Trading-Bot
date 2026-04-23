import logging
import os
import time

from dotenv import load_dotenv

from base_broker import BaseBroker
from broker import AlpacaBroker
from strategy import moving_average_signal

load_dotenv()

SYMBOL = os.getenv("SYMBOL", "AAPL")
TIMEFRAME = os.getenv("TIMEFRAME", "5Min")
WINDOW = int(os.getenv("WINDOW", "20"))
CONFIRM_BARS = int(os.getenv("CONFIRM_BARS", "3"))
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
MAX_BACKOFF_SECONDS = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def position_size(buying_power: float, price: float) -> int:
    """
    Simple sizing rule:
    allocate RISK_PER_TRADE fraction of buying power to a new position.
    """
    if price <= 0:
        return 0

    qty = int((buying_power * RISK_PER_TRADE) / price)

    if buying_power < price:
        return 0

    return max(1, qty)


def daily_loss_exceeded(broker: BaseBroker) -> bool:
    equity, last_equity = broker.get_equity()

    if last_equity <= 0:
        return False

    drawdown = (last_equity - equity) / last_equity
    return drawdown >= MAX_DAILY_LOSS_PCT


def run_once(broker: BaseBroker, sleep_enabled: bool = True) -> None:
    is_open, wait = broker.get_market_status()

    if not is_open:
        logger.info(f"Market is closed. Sleeping {wait:.0f}s until open.")
        if sleep_enabled:
            time.sleep(wait if wait > 0 else CHECK_INTERVAL_SECONDS)
        return

    if daily_loss_exceeded(broker):
        logger.warning(
            f"Daily loss limit ({MAX_DAILY_LOSS_PCT:.1%}) reached. Halting for the day."
        )
        if sleep_enabled:
            time.sleep(3600)
        return

    closes = broker.get_recent_closes(
        SYMBOL,
        limit=WINDOW + CONFIRM_BARS,
        timeframe=TIMEFRAME,
    )

    latest_price = float(closes.iloc[-1])
    signal = moving_average_signal(
        closes,
        window=WINDOW,
        confirm_bars=CONFIRM_BARS,
    )
    current_qty = broker.get_position_qty(SYMBOL)
    open_order_exists = broker.has_open_order(SYMBOL)

    logger.info(
        f"{SYMBOL} | price={latest_price:.2f} | signal={signal} "
        f"| held={current_qty} | open_order={open_order_exists}"
    )

    if open_order_exists:
        logger.info("Open order already exists. No new action.")
        if sleep_enabled:
            time.sleep(CHECK_INTERVAL_SECONDS)
        return

    # Stop loss
    if current_qty > 0:
        entry_price = broker.get_entry_price(SYMBOL)
        if entry_price is not None and latest_price <= entry_price * (1 - STOP_LOSS_PCT):
            broker.submit_sell(SYMBOL, current_qty)
            logger.warning(
                f"STOP LOSS: sold {current_qty} {SYMBOL} at {latest_price:.2f} "
                f"(entry {entry_price:.2f})"
            )
            if sleep_enabled:
                time.sleep(CHECK_INTERVAL_SECONDS)
            return

    if signal == "BUY" and current_qty == 0:
        qty = position_size(broker.get_buying_power(), latest_price)

        if qty <= 0:
            logger.info("Insufficient buying power to open a position.")
        else:
            broker.submit_buy(SYMBOL, qty)
            logger.info(f"BUY sent for {qty} share(s) of {SYMBOL}")

    elif signal == "SELL" and current_qty > 0:
        broker.submit_sell(SYMBOL, current_qty)
        logger.info(f"SELL sent for {current_qty} share(s) of {SYMBOL}")

    else:
        logger.info("No action")

    if sleep_enabled:
        time.sleep(CHECK_INTERVAL_SECONDS)


def run_bot(broker: BaseBroker | None = None) -> None:
    broker = broker or AlpacaBroker()
    logger.info("Bot started")

    error_backoff = CHECK_INTERVAL_SECONDS

    while True:
        try:
            run_once(broker, sleep_enabled=True)
        except Exception as e:
            logger.exception(f"Bot error: {e}")
            time.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, MAX_BACKOFF_SECONDS)
            continue

        error_backoff = CHECK_INTERVAL_SECONDS


if __name__ == "__main__":
    run_bot()