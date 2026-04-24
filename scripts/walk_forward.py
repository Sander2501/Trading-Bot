"""
scripts/walk_forward.py
~~~~~~~~~~~~~~~~~~~~~~~
Walk-forward validation for the current strategy configuration.

Splits a historical CSV into sliding (in-sample, out-of-sample) windows and
runs a full BacktestBroker backtest on each window.  The report shows whether
in-sample performance transfers out-of-sample — if IS is green and OOS is red
across windows, the apparent edge is curve-fitting noise.

This script does NOT tune parameters across windows (doing so sensibly would
require a grid search per window, which is out of scope here).  It instead
validates *stability*: does the strategy's behaviour on bars it has "seen"
(IS) match its behaviour on bars it hasn't (OOS)?

Usage
-----
    # 90-day data, 60-day IS / 30-day OOS, slide 30 days → 2 windows
    python scripts/walk_forward.py --csv historical_data.csv \
                                   --train-days 60 --test-days 30 --step-days 30

    # With 15-minute bars you need many more rows:
    python scripts/fetch_history.py --interval 15m --days 180 --out data/btc_15m_180d.csv
    python scripts/walk_forward.py --csv data/btc_15m_180d.csv \
                                   --train-days 60 --test-days 30 --step-days 30 \
                                   --bars-per-day 96
"""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

# Make the project root importable so this script can be invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brokers import BacktestBroker  # noqa: E402
from config import (  # noqa: E402
    COMMISSION_PER_TRADE,
    SLIPPAGE_PCT,
    STARTING_CASH,
    SYMBOL,
)
from main import TradingState, run_once  # noqa: E402
from run_backtest import _compute_metrics  # noqa: E402


def _run_segment(csv_path: str) -> dict:
    """Run a full backtest on one CSV segment and return its metrics dict."""
    broker = BacktestBroker(
        csv_path=csv_path,
        symbol=SYMBOL,
        starting_cash=STARTING_CASH,
        slippage_pct=SLIPPAGE_PCT,
        commission_per_trade=COMMISSION_PER_TRADE,
    )
    state = TradingState()
    while not broker.done():
        run_once(broker, state=state, sleep_enabled=False)
        broker.advance()

    # Flatten any open position so final equity reflects it.
    qty = broker.get_position_qty(SYMBOL)
    if qty > 0:
        broker.submit_sell(SYMBOL, qty)
    elif qty < 0:
        broker.submit_buy(SYMBOL, abs(qty))

    return _compute_metrics(broker)


def _window_slices(total_bars: int, train_bars: int, test_bars: int, step_bars: int):
    """Yield (is_start, is_end, oos_start, oos_end) index tuples."""
    start = 0
    while start + train_bars + test_bars <= total_bars:
        yield (start, start + train_bars, start + train_bars, start + train_bars + test_bars)
        start += step_bars


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for the trading bot strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Historical CSV to walk-forward over")
    parser.add_argument("--train-days", type=int, default=60, help="In-sample window length in days")
    parser.add_argument("--test-days",  type=int, default=30, help="Out-of-sample window length in days")
    parser.add_argument("--step-days",  type=int, default=30, help="How many days to slide forward each step")
    parser.add_argument(
        "--bars-per-day", type=int, default=1440,
        help="Bars in one day (1440 for 1m, 96 for 15m, 24 for 1h, etc.)",
    )
    parser.add_argument(
        "--out", default="walk_forward_report.csv",
        help="Where to save the per-window metrics table",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    total = len(df)
    train_bars = args.train_days * args.bars_per_day
    test_bars  = args.test_days  * args.bars_per_day
    step_bars  = args.step_days  * args.bars_per_day

    if total < train_bars + test_bars:
        raise SystemExit(
            f"Not enough data: need {train_bars + test_bars:,} bars for one window, have {total:,}. "
            f"Fetch more with scripts/fetch_history.py."
        )

    print(
        f"Walk-forward over {total:,} bars "
        f"(~{total / args.bars_per_day:.1f} days at {args.bars_per_day} bars/day)"
    )
    print(f"Train window: {train_bars:,} bars | Test window: {test_bars:,} bars | Step: {step_bars:,} bars")

    rows = []
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, (is_start, is_end, oos_start, oos_end) in enumerate(
            _window_slices(total, train_bars, test_bars, step_bars), start=1
        ):
            is_csv  = tmp_dir / f"is_{i}.csv"
            oos_csv = tmp_dir / f"oos_{i}.csv"
            df.iloc[is_start:is_end].to_csv(is_csv, index=False)
            df.iloc[oos_start:oos_end].to_csv(oos_csv, index=False)

            print(f"\n── Window {i}: IS [{is_start}:{is_end}]  OOS [{oos_start}:{oos_end}]")
            is_m  = _run_segment(str(is_csv))
            oos_m = _run_segment(str(oos_csv))

            rows.append({
                "window": i,
                "is_start": is_start, "is_end": is_end,
                "oos_start": oos_start, "oos_end": oos_end,
                "is_roi_pct": is_m["roi_pct"],
                "oos_roi_pct": oos_m["roi_pct"],
                "is_pf": is_m["profit_factor"],
                "oos_pf": oos_m["profit_factor"],
                "is_max_dd_pct": is_m["max_drawdown_pct"],
                "oos_max_dd_pct": oos_m["max_drawdown_pct"],
                "is_trades": is_m["total_trades"],
                "oos_trades": oos_m["total_trades"],
            })

    out_df = pd.DataFrame(rows)

    print("\n" + "=" * 96)
    print("WALK-FORWARD RESULTS")
    print("=" * 96)
    print(f"{'Window':<7}{'IS ROI':>10}{'OOS ROI':>10}{'IS PF':>8}{'OOS PF':>8}"
          f"{'IS DD':>8}{'OOS DD':>8}{'IS N':>6}{'OOS N':>6}")
    print("-" * 96)
    for r in rows:
        print(
            f"{r['window']:<7}"
            f"{r['is_roi_pct']:>9.2f}%{r['oos_roi_pct']:>9.2f}%"
            f"{r['is_pf']:>8.2f}{r['oos_pf']:>8.2f}"
            f"{r['is_max_dd_pct']:>7.2f}%{r['oos_max_dd_pct']:>7.2f}%"
            f"{r['is_trades']:>6}{r['oos_trades']:>6}"
        )
    print("-" * 96)

    # Aggregate: does OOS track IS?
    is_mean  = out_df["is_roi_pct"].mean()
    oos_mean = out_df["oos_roi_pct"].mean()
    oos_positive = (out_df["oos_roi_pct"] > 0).sum()
    total_wins_oos = f"{oos_positive}/{len(out_df)}"
    print(f"Mean IS ROI:  {is_mean:+.2f}%")
    print(f"Mean OOS ROI: {oos_mean:+.2f}%   (positive windows: {total_wins_oos})")
    if abs(is_mean) > 0.1:
        degradation = (oos_mean - is_mean) / abs(is_mean) * 100
        print(f"OOS degradation vs IS: {degradation:+.1f}%   "
              f"(close to 0 = stable; strongly negative = curve-fit)")

    out_df.to_csv(args.out, index=False)
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
