import time
import logging

from broker import AlpacaBroker
from strategy import moving_average_signal

SYMBOL = "AAPL"
WINDOW = 20
QTY = 1
CHECK_INTERVAL_SECONDS = 60

# -------------------
# LOGGING
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_bot():
    broker = AlpacaBroker()
    logger.info("Bot started")

    while True:
        try:
            # 1. Check if market is open
            if not broker.is_market_open():
                logger.info("Market is closed. Waiting...")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 2. Get recent prices
            closes = broker.get_recent_closes(SYMBOL, limit=WINDOW + 5)

            latest_price = float(closes.iloc[-1])
            signal = moving_average_signal(closes, window=WINDOW)
            current_qty = broker.get_position_qty(SYMBOL)
            open_order_exists = broker.has_open_order(SYMBOL)

            logger.info(
                f"{SYMBOL} | price={latest_price:.2f} | signal={signal} | held={current_qty} | open_order={open_order_exists}"
            )

            # 3. Prevent duplicate orders
            if open_order_exists:
                logger.info("Open order already exists. No new action.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 4. Execute logic
            if signal == "BUY" and current_qty == 0:
                broker.submit_buy(SYMBOL, QTY)
                logger.info(f"BUY sent for {QTY} share(s) of {SYMBOL}")

            elif signal == "SELL" and current_qty > 0:
                broker.submit_sell(SYMBOL, current_qty)
                logger.info(f"SELL sent for {current_qty} share(s) of {SYMBOL}")

            else:
                logger.info("No action")

        except Exception as e:
            logger.exception(f"Bot error: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_bot()