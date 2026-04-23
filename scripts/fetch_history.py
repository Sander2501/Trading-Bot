"""
scripts/fetch_history.py
~~~~~~~~~~~~~~~~~~~~~~~~
Fetch OHLCV history from the Binance public REST API (no API key needed)
and write a CSV file compatible with BacktestBroker.

Output columns: t, h, l, c
  t – ISO-8601 UTC timestamp (bar open time)
  h – bar high
  l – bar low
  c – bar close

Usage examples
--------------
  # Last 7 days of BTC/USDT 1-minute bars → historical_data.csv
  python scripts/fetch_history.py

  # Last 30 days, custom output path
  python scripts/fetch_history.py --days 30 --out data/btc_30d.csv

  # Different pair / interval
  python scripts/fetch_history.py --symbol ETHUSDT --interval 5m --days 14

  # US users: swap to the Binance.US endpoint
  python scripts/fetch_history.py --api-url https://api.binance.us

Notes
-----
- Binance.com is unavailable in some regions (e.g. the US).
  Use --api-url https://api.binance.us in that case.
- 1-minute data at 30 days = ~43,200 bars (~44 API pages). Expect ~30 s.
- Data is sorted ascending by bar open time.
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

BINANCE_KLINES_PATH = "/api/v3/klines"
MAX_LIMIT = 1000  # Binance hard cap per request


def _interval_ms(interval: str) -> int:
    """Convert a Binance interval string (e.g. '1m', '5m', '1h') to milliseconds."""
    unit = interval[-1]
    value = int(interval[:-1])
    ms = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    if unit not in ms:
        raise ValueError(f"Unknown interval unit '{unit}' — expected one of: m h d w")
    return value * ms[unit]


def fetch_klines(
    base_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Fetch all klines in [start_ms, end_ms), handling Binance pagination."""
    url = base_url.rstrip("/") + BINANCE_KLINES_PATH
    bar_ms = _interval_ms(interval)
    rows: list[list] = []
    cursor = start_ms

    with requests.Session() as session:
        session.headers["User-Agent"] = "trading-bot-backtest-fetcher/1.0"

        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": MAX_LIMIT,
            }

            for attempt in range(5):
                try:
                    resp = session.get(url, params=params, timeout=30)
                except requests.RequestException as exc:
                    if attempt == 4:
                        raise SystemExit(f"Network error after 5 attempts: {exc}") from exc
                    wait = 2 ** attempt
                    print(f"  Network error ({exc}), retrying in {wait}s …", flush=True)
                    time.sleep(wait)
                    continue

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "60"))
                    print(f"  Rate limited — waiting {retry_after}s …", flush=True)
                    time.sleep(retry_after)
                    continue

                if resp.status_code == 451:
                    raise SystemExit(
                        "HTTP 451 (geo-restricted). "
                        "Try --api-url https://api.binance.us if you are in the US."
                    )

                if not resp.ok:
                    raise SystemExit(
                        f"API error {resp.status_code}: {resp.text[:200]}\n"
                        "Check --symbol spelling (e.g. BTCUSDT, not BTC/USDT)."
                    )

                break

            batch: list[list] = resp.json()
            if not batch:
                break

            rows.extend(batch)
            last_open_ms: int = batch[-1][0]
            cursor = last_open_ms + bar_ms

            ts_str = datetime.fromtimestamp(last_open_ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
            print(f"  {len(rows):>7,} bars fetched  (last bar: {ts_str} UTC)", flush=True)

            if len(batch) < MAX_LIMIT:
                break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV history from Binance into a BacktestBroker-compatible CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Binance trading pair symbol",
    )
    parser.add_argument(
        "--interval", default="1m",
        help="Bar interval: 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d",
    )
    parser.add_argument(
        "--days", type=float, default=7.0,
        help="Number of days of history to fetch",
    )
    parser.add_argument(
        "--out", default="historical_data.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--api-url", default="https://api.binance.com",
        dest="api_url",
        help="Binance base URL (use https://api.binance.us for US users)",
    )
    args = parser.parse_args()

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_dt = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
    start_ms = int(start_dt.timestamp() * 1000)

    bar_ms = _interval_ms(args.interval)
    expected = int((now_ms - start_ms) / bar_ms)
    print(
        f"Fetching ~{expected:,} {args.interval} bars for {args.symbol} "
        f"over {args.days:.1f} days  →  {args.out}",
        flush=True,
    )

    rows = fetch_klines(args.api_url, args.symbol, args.interval, start_ms, now_ms)

    if not rows:
        print("No data returned — check --symbol and --interval.", file=sys.stderr)
        sys.exit(1)

    # Binance kline format:
    # [0] open time ms, [1] open, [2] high, [3] low, [4] close, [5] volume, …
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "h", "l", "c"])
        for row in rows:
            open_time_ms: int = row[0]
            ts = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            writer.writerow([ts, row[2], row[3], row[4]])  # h, l, c

    print(f"\nDone — wrote {len(rows):,} bars to {args.out}", flush=True)
    print(
        f"Date range: {rows[0][0]} ms → {rows[-1][0]} ms  "
        f"({datetime.fromtimestamp(rows[0][0]/1000, tz=timezone.utc).date()} "
        f"to {datetime.fromtimestamp(rows[-1][0]/1000, tz=timezone.utc).date()})",
        flush=True,
    )


if __name__ == "__main__":
    main()
