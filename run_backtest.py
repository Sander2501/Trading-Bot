"""
run_backtest
~~~~~~~~~~~~
Offline backtest entry-point.

Run with::

    python run_backtest.py

Replays ``historical_data.csv`` (or the file set by ``BACKTEST_CSV``) through
the same ``run_once`` loop used in live trading, then prints a full
performance report and saves ``backtest_report.json``.
"""

import json
import statistics
from collections import deque
from datetime import datetime, timezone

from brokers import BacktestBroker
from config import (
    BACKTEST_DYNAMIC_SLIPPAGE_K,
    BACKTEST_LATENCY_BARS,
    BACKTEST_PARTIAL_FILL_MIN,
    COMMISSION_PER_TRADE,
    CSV_PATH,
    SLIPPAGE_PCT,
    STARTING_CASH,
    SYMBOL,
    TIMEFRAME,
)
from main import TradingState, run_once


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

    # Win rate & Avg Win/Loss using FIFO lot matching.
    # This handles scale-ins/scale-outs better than single-entry/single-exit pairing.
    long_lots: deque[tuple[float, float]] = deque()   # (qty, entry_price)
    short_lots: deque[tuple[float, float]] = deque()  # (qty, entry_price)
    wins: list[float] = []
    losses: list[float] = []

    for t in trades:
        qty = float(t["qty"])
        price = float(t["price"])
        side = t["side"]

        # BUY/COVER: close existing short lots first, then open long lots.
        if side in {"BUY", "COVER", "SL_STOP", "TP_STOP"} and short_lots:
            qty_left = qty
            while qty_left > 0 and short_lots:
                lot_qty, lot_price = short_lots[0]
                matched = min(qty_left, lot_qty)
                pnl = (lot_price - price) * matched
                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(pnl)
                qty_left -= matched
                lot_qty -= matched
                if lot_qty == 0:
                    short_lots.popleft()
                else:
                    short_lots[0] = (lot_qty, lot_price)
            if qty_left > 0:
                long_lots.append((qty_left, price))
        elif side in {"BUY", "COVER"}:
            long_lots.append((qty, price))

        # SELL/SHORT: close existing long lots first, then open short lots.
        elif side in {"SELL", "SHORT", "SL_STOP", "TP_STOP"} and long_lots:
            qty_left = qty
            while qty_left > 0 and long_lots:
                lot_qty, lot_price = long_lots[0]
                matched = min(qty_left, lot_qty)
                pnl = (price - lot_price) * matched
                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(pnl)
                qty_left -= matched
                lot_qty -= matched
                if lot_qty == 0:
                    long_lots.popleft()
                else:
                    long_lots[0] = (lot_qty, lot_price)
            if qty_left > 0:
                short_lots.append((qty_left, price))
        elif side in {"SELL", "SHORT"}:
            short_lots.append((qty, price))

    total_trips = len(wins) + len(losses)
    win_rate = len(wins) / total_trips * 100 if total_trips > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

    # Annualised Sharpe (rf = 0). Bars/year are derived from configured timeframe.
    def _bars_per_year(timeframe: str) -> int:
        tf = timeframe.strip()
        mapping = {
            "1Min": 365 * 24 * 60,
            "5Min": 365 * 24 * 12,
            "15Min": 365 * 24 * 4,
            "30Min": 365 * 24 * 2,
            "1H": 365 * 24,
            "4H": 365 * 6,
            "1D": 365,
            "1W": 52,
        }
        return mapping.get(tf, 365 * 24 * 4)  # default to 15Min

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
                bars_per_year = _bars_per_year(TIMEFRAME)
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


def run_backtest_for_csv(
    csv_path: str,
    starting_cash: float = STARTING_CASH,
) -> tuple[BacktestBroker, dict]:
    broker = BacktestBroker(
        csv_path=csv_path,
        symbol=SYMBOL,
        starting_cash=starting_cash,
        slippage_pct=SLIPPAGE_PCT,
        commission_per_trade=COMMISSION_PER_TRADE,
        dynamic_slippage_k=BACKTEST_DYNAMIC_SLIPPAGE_K,
        latency_bars=BACKTEST_LATENCY_BARS,
        partial_fill_min=BACKTEST_PARTIAL_FILL_MIN,
    )

    # A single persistent state is essential — previously a fresh TradingState
    # was created on every cycle, which reset the daily-loss baseline every
    # cycle (halt never tripped) and wiped trailing-stop watermarks.
    state = TradingState()

    # Replay every bar through the same run_once logic used in live trading.
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    final_qty = broker.get_position_qty(SYMBOL)
    if final_qty > 0:
        broker.submit_sell(SYMBOL, final_qty)
    elif final_qty < 0:
        broker.submit_buy(SYMBOL, abs(final_qty))

    m = _compute_metrics(broker)
    return broker, m


def main() -> None:
    broker, m = run_backtest_for_csv(CSV_PATH, starting_cash=STARTING_CASH)

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

    # ------------------------------------------------------------------
    # Save machine-readable report
    # ------------------------------------------------------------------
    # Replace non-finite floats (inf, nan) so the JSON is always valid.
    def _safe(v):
        if isinstance(v, float) and not (v == v) or (isinstance(v, float) and abs(v) == float("inf")):
            return None
        return v

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "starting_cash": STARTING_CASH,
        **{k: _safe(v) for k, v in m.items()},
    }
    with open("backtest_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nReport saved to backtest_report.json")


if __name__ == "__main__":
    main()
