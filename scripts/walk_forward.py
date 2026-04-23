"""
Walk-forward validator for Trading-Bot backtests.

Usage:
    python scripts/walk_forward.py --csv historical_data.csv --train-bars 8000 --test-bars 4000
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from run_backtest import run_backtest_for_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation runner.")
    p.add_argument("--csv", default="historical_data.csv")
    p.add_argument("--train-bars", type=int, default=8000)
    p.add_argument("--test-bars", type=int, default=4000)
    p.add_argument("--step-bars", type=int, default=4000)
    p.add_argument("--max-folds", type=int, default=10)
    p.add_argument("--out", default="walk_forward_report.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.csv)
    if len(df) < args.train-bars + args.test-bars:
        raise ValueError("Not enough bars for one walk-forward fold.")

    folds: list[dict] = []
    start = 0
    fold_id = 0
    while fold_id < args.max_folds and start + args.train-bars + args.test-bars <= len(df):
        test_start = start + args.train-bars
        test_end = test_start + args.test-bars
        test_df = df.iloc[test_start:test_end].copy()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tmp_path = Path(tf.name)
        test_df.to_csv(tmp_path, index=False)
        _, m = run_backtest_for_csv(str(tmp_path))
        folds.append({
            "fold": fold_id,
            "train_start": start,
            "train_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
            **m,
        })
        tmp_path.unlink(missing_ok=True)
        fold_id += 1
        start += args.step-bars

    if not folds:
        raise ValueError("No folds produced. Check window sizes.")

    def _avg(key: str) -> float:
        return sum(float(f[key]) for f in folds) / len(folds)

    summary = {
        "folds": len(folds),
        "avg_roi_pct": _avg("roi_pct"),
        "avg_sharpe_ratio": _avg("sharpe_ratio"),
        "avg_profit_factor": _avg("profit_factor"),
        "avg_max_drawdown_pct": _avg("max_drawdown_pct"),
        "avg_win_rate_pct": _avg("win_rate_pct"),
    }

    out = {"summary": summary, "fold_results": folds}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Walk-forward report saved to {args.out}")


if __name__ == "__main__":
    main()
