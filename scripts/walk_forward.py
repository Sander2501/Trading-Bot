"""
Walk-forward validator for Trading-Bot backtests.

Usage:
    python scripts/walk_forward.py --csv historical_data.csv --train-bars 8000 --test-bars 4000
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

import config
import main as bot_main
from run_backtest import run_backtest_for_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation runner.")
    p.add_argument("--csv", default="historical_data.csv")
    p.add_argument("--train-bars", type=int, default=8000)
    p.add_argument("--test-bars", type=int, default=4000)
    p.add_argument("--step-bars", type=int, default=4000)
    p.add_argument("--max-folds", type=int, default=10)
    p.add_argument("--nested-opt", action="store_true", help="Optimize params on train fold before testing.")
    p.add_argument("--out", default="walk_forward_report.json")
    return p.parse_args()


GRID = {
    "ADX_THRESHOLD": [22, 25, 28],
    "MIN_ATR_PCT": [0.0008, 0.0010, 0.0012],
    "ATR_STOP_MULT": [1.5, 2.0, 2.5],
    "TAKE_PROFIT_MULT": [3.0, 4.0, 5.0],
    "CONFIRM_BARS": [1, 2, 3],
}


def _score(m: dict) -> float:
    return (
        float(m["roi_pct"]) * 0.25
        + float(m["sharpe_ratio"]) * 30.0
        + float(m["profit_factor"]) * 12.0
        - float(m["max_drawdown_pct"]) * 0.9
    )


def _run_with_overrides(csv_path: str, overrides: dict[str, str] | None = None) -> dict:
    overrides = overrides or {}
    old_env = {k: os.environ.get(k) for k in overrides}
    old_cfg = {k: getattr(config, k) for k in overrides}
    old_main = {k: getattr(bot_main, k) for k in overrides if hasattr(bot_main, k)}
    try:
        for k, v in overrides.items():
            os.environ[k] = str(v)
            parsed = float(v) if "." in str(v) else int(v)
            setattr(config, k, parsed)
            if hasattr(bot_main, k):
                setattr(bot_main, k, parsed)
        _, m = run_backtest_for_csv(csv_path)
        return m
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in old_cfg.items():
            setattr(config, k, v)
        for k, v in old_main.items():
            setattr(bot_main, k, v)


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.csv)
    if len(df) < args.train-bars + args.test-bars:
        raise ValueError("Not enough bars for one walk-forward fold.")

    folds: list[dict] = []
    start = 0
    fold_id = 0
    while fold_id < args.max_folds and start + args.train-bars + args.embargo_bars + args.test-bars <= len(df):
        train_end = start + args.train-bars
        test_start = train_end + args.embargo_bars
        test_end = test_start + args.test-bars
        train_df = df.iloc[start:test_start].copy()
        test_df = df.iloc[test_start:test_end].copy()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            train_path = Path(tf.name)
        train_df.to_csv(train_path, index=False)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tmp_path = Path(tf.name)
        test_df.to_csv(tmp_path, index=False)
        best_params: dict[str, str] = {}
        train_metrics: dict = {}
        if args.nested_opt:
            keys = list(GRID.keys())
            best_score = float("-inf")
            for combo in itertools.product(*(GRID[k] for k in keys)):
                params = dict(zip(keys, map(str, combo)))
                m_train = _run_with_overrides(str(train_path), params)
                score = _score(m_train)
                if score > best_score:
                    best_score = score
                    best_params = params
                    train_metrics = m_train
        m = _run_with_overrides(str(tmp_path), best_params if args.nested_opt else None)
        folds.append({
            "fold": fold_id,
            "train_start": start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "nested_opt": bool(args.nested_opt),
            "selected_params": best_params,
            "train_metrics": train_metrics,
            **m,
        })
        train_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)
        fold_id += 1
        start += args.step-bars

    if not folds:
        raise ValueError("No folds produced. Check window sizes.")

    def _avg(key: str) -> float:
        return sum(float(f[key]) for f in folds) / len(folds)

    def _median(key: str) -> float:
        vals = sorted(float(f[key]) for f in folds)
        n = len(vals)
        mid = n // 2
        if n % 2 == 1:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    profitable_folds_pct = 100.0 * sum(1 for f in folds if float(f["roi_pct"]) > 0.0) / len(folds)

    summary = {
        "folds": len(folds),
        "avg_roi_pct": _avg("roi_pct"),
        "avg_sharpe_ratio": _avg("sharpe_ratio"),
        "avg_profit_factor": _avg("profit_factor"),
        "avg_max_drawdown_pct": _avg("max_drawdown_pct"),
        "avg_win_rate_pct": _avg("win_rate_pct"),
        "median_profit_factor": _median("profit_factor"),
        "median_sharpe_ratio": _median("sharpe_ratio"),
        "median_max_drawdown_pct": _median("max_drawdown_pct"),
        "profitable_folds_pct": profitable_folds_pct,
    }

    out = {"summary": summary, "fold_results": folds}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Walk-forward report saved to {args.out}")


if __name__ == "__main__":
    main()
