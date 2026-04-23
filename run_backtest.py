"""
run_backtest
~~~~~~~~~~~~
Offline backtest entry-point.

Run with::

    python run_backtest.py

Replays ``historical_data.csv`` (or the file set by ``BACKTEST_CSV``) through
the same ``run_once`` loop used in live trading, then prints a full
performance report.
"""

import statistics

from brokers import BacktestBroker
from config import CSV_PATH, STARTING_CASH, SYMBOL
from main import run_once


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def _compute_metrics(broker: BacktestBroker) -> dict:
    """
    Compute standard post-run performance metrics.

    Returns
    -------
    dict with keys:
        roi_pct          – Return on investment vs. starting cash (%).
        max_drawdown_pct – Largest peak-to-trough decline in equity (%).
        win_rate_pct     – % of round-trip trades that were profitable.
        sharpe_ratio     – Annualised Sharpe ratio (rf = 0, 1-min bars).
        total_trades     – Total number of fills (BUY + SELL combined).
        final_equity     – Closing account value in USD.
    """
    starting = broker._starting_equity
    equity_curve = broker.equity_curve or [starting]
    trades = broker.trades

    # ROI
    final_equity, _ = broker.get_equity()
    roi = (final_equity - starting) / starting * 100

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

    # Win rate — count profitable round-trips (handles long and short legs)
    long_entry: float | None = None
    short_entry: float | None = None
    wins = losses = 0
    for t in trades:
        if t["side"] == "BUY":
            long_entry = t["price"]
        elif t["side"] == "SELL" and long_entry is not None:
            if t["price"] > long_entry:
                wins += 1
            else:
                losses += 1
            long_entry = None
        elif t["side"] == "SHORT":
            short_entry = t["price"]
        elif t["side"] == "COVER" and short_entry is not None:
            if t["price"] < short_entry:   # profit on short = cover below entry
                wins += 1
            else:
                losses += 1
            short_entry = None

    total_trips = wins + losses
    win_rate = wins / total_trips * 100 if total_trips > 0 else 0.0

    # Annualised Sharpe (rf = 0, assumes 1-min bars, 252 trading days)
    sharpe = 0.0
    if len(equity_curve) > 1:
        bar_returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1] > 0
        ]
        if len(bar_returns) > 1:
            std_r = statistics.stdev(bar_returns)
            if std_r > 0:
                bars_per_year = 252 * 390  # trading days × 1-min bars/day
                sharpe = statistics.mean(bar_returns) / std_r * (bars_per_year**0.5)

    return {
        "roi_pct": roi,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate,
        "sharpe_ratio": sharpe,
        "total_trades": len(trades),
        "final_equity": final_equity,
    }


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


def main() -> None:
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

    # Close any remaining open position at the last available price
    final_qty = broker.get_position_qty(SYMBOL)
    if final_qty > 0:
        broker.submit_sell(SYMBOL, final_qty)
    elif final_qty < 0:
        broker.submit_buy(SYMBOL, abs(final_qty))

    m = _compute_metrics(broker)

    print("\n=== BACKTEST COMPLETE ===")
    print(f"Starting cash    : ${STARTING_CASH:>12,.2f}")
    print(f"Final equity     : ${m['final_equity']:>12,.2f}")
    print(f"ROI              : {m['roi_pct']:>+.2f}%")
    print(f"Max drawdown     : {m['max_drawdown_pct']:.2f}%")
    print(f"Win rate         : {m['win_rate_pct']:.1f}%  ({m['total_trades']} total fills)")
    print(f"Sharpe ratio     : {m['sharpe_ratio']:.3f}")

    if broker.trades:
        print("\nTrades:")
        for t in broker.trades:
            print(
                f"  [{t['t']:>4}] {t['side']:<5}  {t['qty']:.6f}"
                f" @ {t['price']:>10,.2f}  equity={t['equity']:>12,.2f}"
            )


if __name__ == "__main__":
    main()
