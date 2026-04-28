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
from collections import deque
from random import Random

import telemetry
from brokers import BacktestBroker
from config import (
    BACKTEST_DYNAMIC_SLIPPAGE_K,
    BACKTEST_LATENCY_BARS,
    BACKTEST_PARTIAL_FILL_MIN,
    COMMISSION_PER_TRADE,
    CSV_PATH,
    MIN_EVAL_TRADES,
    SLIPPAGE_PCT,
    STARTING_CASH,
    SYMBOL,
    TELEMETRY_ENABLED,
    TELEMETRY_FILE,
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
    trade_pnls = wins + losses
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
    bar_returns: list[float] = []
    bars_per_year = _bars_per_year(TIMEFRAME)
    annualization = bars_per_year**0.5
    if len(equity_curve) > 1:
        bar_returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1] > 0
        ]
        if len(bar_returns) > 1:
            std_r = statistics.stdev(bar_returns)
            if std_r > 0:
                sharpe = statistics.mean(bar_returns) / std_r * annualization

    def _bootstrap_ci(values: list[float], stat_fn, n_boot: int = 500, alpha: float = 0.05) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        rng = Random(42)
        boot = []
        n = len(values)
        for _ in range(n_boot):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            s = stat_fn(sample)
            if s is not None:
                boot.append(float(s))
        if not boot:
            return None, None
        boot.sort()
        lo_i = int((alpha / 2) * (len(boot) - 1))
        hi_i = int((1 - alpha / 2) * (len(boot) - 1))
        return boot[lo_i], boot[hi_i]

    def _pf(samples: list[float]) -> float | None:
        pos = [x for x in samples if x > 0]
        neg = [x for x in samples if x < 0]
        if not neg:
            return None
        return sum(pos) / abs(sum(neg))

    def _sh(samples: list[float]) -> float | None:
        # Annualized to match the headline Sharpe; bootstrap CI must be in the
        # same units as the point estimate or comparison is meaningless.
        if len(samples) < 2:
            return None
        std = statistics.stdev(samples)
        if std <= 0:
            return None
        return statistics.mean(samples) / std * annualization

    pf_ci_low, pf_ci_high = _bootstrap_ci(trade_pnls, _pf)
    sh_ci_low, sh_ci_high = _bootstrap_ci(bar_returns, _sh)

    monthly_returns: dict[str, float] = {}
    regime_trade_counts = {"TRENDING": 0, "RANGING": 0}
    ts = getattr(broker, "_timestamps", None)
    if ts is not None and len(ts) > 0:
        for i in range(1, len(equity_curve)):
            if i >= len(ts):
                break
            month = str(ts.iloc[i])[:7]
            prev = equity_curve[i - 1]
            if prev > 0:
                monthly_returns.setdefault(month, 1.0)
                monthly_returns[month] *= 1.0 + (equity_curve[i] - prev) / prev
        monthly_returns = {m: (v - 1.0) * 100.0 for m, v in monthly_returns.items()}

    for i in range(1, len(equity_curve)):
        if i >= len(equity_curve):
            break
        if i <= 0:
            continue
        lookback = max(0, i - 20)
        recent = equity_curve[lookback:i + 1]
        if len(recent) < 2:
            continue
        changes = [abs(recent[j] - recent[j - 1]) for j in range(1, len(recent))]
        if sum(changes) == 0:
            regime_trade_counts["RANGING"] += 1
        else:
            regime_trade_counts["TRENDING"] += 1

    quality_gate_pass = len(trades) >= MIN_EVAL_TRADES

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
        "metrics_quality_gate_pass": quality_gate_pass,
        "min_eval_trades_required": MIN_EVAL_TRADES,
        "pf_ci_95": [pf_ci_low, pf_ci_high],
        "sharpe_ci_95": [sh_ci_low, sh_ci_high],
        "monthly_returns_pct": monthly_returns,
        "regime_bar_counts": regime_trade_counts,
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
    telemetry.configure(TELEMETRY_FILE if TELEMETRY_ENABLED else None, enabled=TELEMETRY_ENABLED)
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
    print(f"Quality Gate     : {'PASS' if m['metrics_quality_gate_pass'] else 'FAIL'} (min trades={m['min_eval_trades_required']})")
    print(f"PF 95% CI        : {m['pf_ci_95']}")
    print(f"Sharpe 95% CI    : {m['sharpe_ci_95']}")
    print("="*40)

if __name__ == "__main__":
    main()
