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
from config import CSV_PATH, STARTING_CASH, SYMBOL, SLIPPAGE_PCT, COMMISSION_PER_TRADE
from main import run_once


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def _compute_metrics(broker: BacktestBroker) -> dict:
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

    # Win rate & Avg Win/Loss
    long_entry: float | None = None
    short_entry: float | None = None
    wins: list[float] = []
    losses: list[float] = []
    
    for t in trades:
        if t["side"] == "BUY":
            long_entry = t["price"]
        elif t["side"] == "SELL" and long_entry is not None:
            pnl = t["price"] - long_entry
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)
            long_entry = None
        elif t["side"] == "SHORT":
            short_entry = t["price"]
        elif t["side"] == "COVER" and short_entry is not None:
            pnl = short_entry - t["price"]  # profit on short = entry > cover
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)
            short_entry = None

    total_trips = len(wins) + len(losses)
    win_rate = len(wins) / total_trips * 100 if total_trips > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

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
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
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
        slippage_pct=SLIPPAGE_PCT,
        commission_per_trade=COMMISSION_PER_TRADE,
    )

    # Replay every bar through the same run_once logic used in live trading.
    while not broker.done():
        run_once(broker, sleep_enabled=False)
        broker.advance()

    final_qty = broker.get_position_qty(SYMBOL)
    if final_qty > 0:
        broker.submit_sell(SYMBOL, final_qty)
    elif final_qty < 0:
        broker.submit_buy(SYMBOL, abs(final_qty))

    m = _compute_metrics(broker)

    print("\n" + "="*40)
    print("       BACKTEST PERFORMANCE       ")
    print("="*40)
    print(f"Starting Cash    : ${STARTING_CASH:>12,.2f}")
    print(f"Final Equity     : ${m['final_equity']:>12,.2f}")
    print(f"Return %         : {m['roi_pct']:>11.2f}%")
    print(f"Win Rate         : {m['win_rate_pct']:>11.1f}%")
    print(f"Profit Factor    : {m['profit_factor']:>12.2f}")
    print(f"Max Drawdown     : {m['max_drawdown_pct']:>11.2f}%")
    print(f"Sharpe Ratio     : {m['sharpe_ratio']:>12.3f}")
    print("-"*40)
    print(f"Avg Win          : ${m['avg_win']:>12.2f}")
    print(f"Avg Loss         : ${m['avg_loss']:>12.2f}")
    print(f"Total Fills      : {m['total_trades']:>12}")
    print("="*40)

    if broker.trades:
        print("\nTRADE LOG:")
        for t in broker.trades:
            print(
                f"  [{t['t']:>4}] {t['side']:<5}  {t['qty']:.6f}"
                f" @ {t['price']:>10,.2f}  equity={t['equity']:>12,.2f}"
            )


if __name__ == "__main__":
    main()
