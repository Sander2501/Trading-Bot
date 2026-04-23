"""
Parameter robustness sweeper.

Runs a simple grid search over key strategy/risk knobs and writes a ranked report.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import config
import main as bot_main
from run_backtest import run_backtest_for_csv


CSV_PATH = os.getenv("BACKTEST_CSV", "historical_data.csv")


GRID = {
    "ADX_THRESHOLD": [22, 25, 28],
    "MIN_ATR_PCT": [0.0008, 0.0010, 0.0012],
    "ATR_STOP_MULT": [1.5, 2.0, 2.5],
    "TAKE_PROFIT_MULT": [3.0, 4.0, 5.0],
    "CONFIRM_BARS": [1, 2, 3],
}


def _score(m: dict) -> float:
    # Reward edge + risk-adjusted return, penalize drawdown.
    return (
        float(m["roi_pct"]) * 0.35
        + float(m["sharpe_ratio"]) * 25.0
        + float(m["profit_factor"]) * 12.0
        - float(m["max_drawdown_pct"]) * 0.8
    )


def main() -> None:
    keys = list(GRID)
    rows = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        env_updates = dict(zip(keys, map(str, values)))
        old = {k: os.environ.get(k) for k in env_updates}
        old_cfg = {k: getattr(config, k) for k in env_updates}
        old_main = {k: getattr(bot_main, k) for k in env_updates if hasattr(bot_main, k)}
        try:
            os.environ.update(env_updates)
            # Apply overrides to already-imported modules used by run_once.
            for k, v in env_updates.items():
                parsed = float(v) if "." in v else int(v)
                setattr(config, k, parsed)
                if hasattr(bot_main, k):
                    setattr(bot_main, k, parsed)
            _, m = run_backtest_for_csv(CSV_PATH)
            row = {"params": dict(zip(keys, values)), "metrics": m, "score": _score(m)}
            rows.append(row)
            print(f"{row['score']:.2f} | {row['params']} | roi={m['roi_pct']:.2f}% pf={m['profit_factor']:.2f}")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            for k, v in old_cfg.items():
                setattr(config, k, v)
            for k, v in old_main.items():
                setattr(bot_main, k, v)

    rows.sort(key=lambda r: r["score"], reverse=True)
    report = {"top_10": rows[:10], "all_results_count": len(rows)}
    Path("sweep_report.json").write_text(json.dumps(report, indent=2))
    print("Saved sweep_report.json")


if __name__ == "__main__":
    main()
