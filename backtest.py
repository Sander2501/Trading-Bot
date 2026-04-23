import os

from backtest_broker import BacktestBroker
from main import run_once

CSV_PATH = os.getenv("BACKTEST_CSV", "historical_data.csv")
SYMBOL = os.getenv("SYMBOL", "AAPL")
STARTING_CASH = float(os.getenv("BACKTEST_STARTING_CASH", "100000"))


def main() -> None:
    broker = BacktestBroker(
        csv_path=CSV_PATH,
        symbol=SYMBOL,
        starting_cash=STARTING_CASH,
    )

    while True:
        run_once(broker, sleep_enabled=False)

        if broker._cursor >= len(broker._closes) - 1:
            break

        broker.advance()

    # Force-close any open position at the end
    final_qty = broker.get_position_qty(SYMBOL)
    if final_qty > 0:
        broker.submit_sell(SYMBOL, final_qty)

    equity, _ = broker.get_equity()

    print("\n=== BACKTEST COMPLETE ===")
    print(f"Final equity: {equity:.2f}")
    print(f"Total trades: {len(broker.trades)}")

    if broker.trades:
        print("\nTrades:")
        for trade in broker.trades:
            print(trade)


if __name__ == "__main__":
    main()