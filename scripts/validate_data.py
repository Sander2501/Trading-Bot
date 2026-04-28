"""
scripts/validate_data
~~~~~~~~~~~~~~~~~~~~~
Offline integrity check for a historical-bar CSV. Run before backtesting
to catch missing bars, duplicate timestamps, broken OHLC invariants, and
NaN/inf values before they corrupt your performance numbers.

Usage::

    python scripts/validate_data.py [path/to/bars.csv] [--interval 900]

If no path is given, falls back to ``BACKTEST_CSV`` from config. The
interval (in seconds) enables gap detection; if omitted it is inferred
from ``TIMEFRAME``. Exits non-zero on fatal issues so CI can gate on it.
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/validate_data.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CSV_PATH, TIMEFRAME  # noqa: E402
from data_integrity import check_csv, interval_seconds_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("path", nargs="?", default=CSV_PATH, help="CSV path (default: BACKTEST_CSV)")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Expected bar interval in seconds (default: derived from TIMEFRAME)",
    )
    args = parser.parse_args()

    interval = args.interval if args.interval is not None else interval_seconds_for(TIMEFRAME)
    result = check_csv(args.path, expected_interval_seconds=interval)

    print(f"Validating {args.path} (expected interval: {interval}s)")
    if not result.issues:
        print("OK — no issues detected.")
        return 0

    for msg in result.issues:
        prefix = "FATAL" if msg in result.fatal else "WARN "
        print(f"  [{prefix}] {msg}")

    print()
    print(f"{len(result.issues)} issue(s), {len(result.fatal)} fatal.")
    return 1 if result.fatal else 0


if __name__ == "__main__":
    sys.exit(main())
