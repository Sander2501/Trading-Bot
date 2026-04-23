import os

from brokers import BacktestBroker
from config import CSV_PATH, STARTING_CASH, SYMBOL
from main import run_once


def main() -> None:
    # Initialize the simulated broker using environment variables from config
    broker = BacktestBroker(
        csv_path=CSV_PATH,
        symbol=SYMBOL,
        starting_cash=STARTING_CASH,
    )

    # Replay every bar through the same run_once logic used in live trading.
    # broker.done() becomes True once the cursor has passed the last bar.
    while not broker.done():
        run_once(broker, sleep_enabled=False)
        broker.advance()

    # Force-close any open position at the end of the simulation
    final_qty = broker.get_position_qty(SYMBOL)
    if final_qty > 0:
        broker.submit_sell(SYMBOL, final_qty)

    equity, _ = broker.get_equity()

    print("\n=== BACKTEST COMPLETE ===")
    print(f"Final equity: {equity:,.2f}")
    print(f"Total trades: {len(broker.trades)}")

    if broker.trades:
        print("\nTrades:")
        for trade in broker.trades:
            # Format output clearly rather than dumping the raw dictionary
            print(
                f"  [{trade['t']:>4}] {trade['side']:<4} "
                f"{trade['qty']:.6f} @ {trade['price']:>10,.2f}  "
                f"equity={trade['equity']:>12,.2f}"
            )


if __name__ == "__main__":
    main()