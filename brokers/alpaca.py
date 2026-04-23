"""
brokers.alpaca
~~~~~~~~~~~~~~
Live and paper-trading broker implementation backed by the Alpaca API.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from brokers.base import BaseBroker

load_dotenv()

logger = logging.getLogger(__name__)

# Alpaca crypto data endpoints
_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
_LATEST_QUOTES_URL = "https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes"
_MAX_HTTP_ATTEMPTS = 3

# HTTP status codes that must NOT be retried (permanent client errors)
_NON_RETRYABLE = frozenset({400, 401, 403, 404, 422})

# Warn if the newest bar is older than this many minutes (catches feed outages)
_DATA_STALENESS_WARN_MINUTES = 10

# Map Alpaca timeframe strings to their duration in minutes.
# Used to calculate a time-anchored lookback window for bar queries.
_TIMEFRAME_MINUTES: dict[str, int] = {
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
    "30Min": 30,
    "1H": 60,
    "2H": 120,
    "4H": 240,
    "1D": 1440,
}


class AlpacaBroker(BaseBroker):
    """
    Broker implementation that connects to the Alpaca paper/live trading API.

    Credentials are read from environment variables ``ALPACA_API_KEY`` and
    ``ALPACA_SECRET_KEY`` (typically supplied via a ``.env`` file).

    Position data is cached for the duration of a single ``run_once`` cycle
    to avoid making separate ``get_open_position`` API calls for
    ``get_position_qty`` and ``get_entry_price``.  Call
    ``flush_position_cache()`` at the start of each cycle to invalidate it.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not api_secret:
            raise ValueError(
                "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. "
                "Add them to your .env file."
            )

        self._data_headers: dict[str, str] = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        self.client = TradingClient(api_key, api_secret, paper=True)

        # Per-cycle cache: dict[symbol, position_object | None] or None (unset)
        self._position_cache: dict | None = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def flush_position_cache(self) -> None:
        """Invalidate the position cache at the start of each trading cycle."""
        self._position_cache = None

    def _get_position(self, symbol: str) -> object | None:
        """
        Return the open position object for ``symbol``, reusing a cached
        result if available within the current cycle.
        """
        if self._position_cache is not None:
            return self._position_cache.get(symbol)

        try:
            pos = self.client.get_open_position(symbol)
            self._position_cache = {symbol: pos}
            return pos
        except Exception:
            self._position_cache = {symbol: None}
            return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_recent_closes(
        self, symbol: str, limit: int = 30, timeframe: str = "1Min"
    ) -> pd.Series:
        """
        Return the most recent ``limit`` closing prices as a float Series.

        Bars are fetched using a time-anchored ``start`` window rather than
        a bare ``limit`` query.  This prevents Alpaca from serving stale
        archived bars when the data feed has not been updated recently —
        a common issue with the crypto paper-trading endpoint.
        """
        now = datetime.now(timezone.utc)

        # Calculate a lookback window large enough to contain `limit` bars.
        # We use a 2× multiplier as a buffer for weekends / gaps in crypto data.
        tf_minutes = _TIMEFRAME_MINUTES.get(timeframe, 1)
        lookback = timedelta(minutes=tf_minutes * limit * 2)
        start = (now - lookback).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start,
            "limit": limit,
            "sort": "asc",
        }

        payload = self._get_with_retry(_BARS_URL, params).json()
        bars = payload.get("bars", {}).get(symbol, [])

        if not bars:
            raise ValueError(
                f"No bars returned for {symbol!r} in the last "
                f"{lookback}. Check Alpaca status or your data subscription."
            )

        bars = sorted(bars, key=lambda b: b["t"])

        latest_bar = bars[-1]
        latest_ts = datetime.fromisoformat(latest_bar["t"].replace("Z", "+00:00"))
        age_minutes = (now - latest_ts).total_seconds() / 60

        closes_list = [bar["c"] for bar in bars]

        if age_minutes > _DATA_STALENESS_WARN_MINUTES:
            fresh_price = self._get_current_price(symbol)
            if fresh_price is not None:
                closes_list.append(fresh_price)
                logger.debug(
                    "Bar data is %.0f min old; appended real-time quote %.2f.",
                    age_minutes,
                    fresh_price,
                )
            else:
                logger.warning(
                    "Bar data is %.0f min old (latest bar: %s). "
                    "Alpaca may have a data feed delay or outage.",
                    age_minutes,
                    latest_bar["t"],
                )
        else:
            logger.debug(
                "Latest bar  ts=%s  close=%.2f  age=%.1fmin  (%d bars)",
                latest_bar["t"],
                latest_bar["c"],
                age_minutes,
                len(bars),
            )

        return pd.Series(closes_list, dtype=float)

    def _get_current_price(self, symbol: str) -> float | None:
        """Return a real-time mid-price from the latest quotes endpoint."""
        try:
            payload = self._get_with_retry(
                _LATEST_QUOTES_URL, {"symbols": symbol}
            ).json()
            quote = payload.get("quotes", {}).get(symbol)
            if quote:
                ask = float(quote.get("ap", 0))
                bid = float(quote.get("bp", 0))
                if ask > 0 and bid > 0:
                    return (ask + bid) / 2.0
                return ask if ask > 0 else (bid if bid > 0 else None)
        except Exception:
            logger.debug("Latest quote unavailable; using bar close as current price.")
        return None

    def _get_with_retry(self, url: str, params: dict) -> requests.Response:
        """
        HTTP GET with exponential-backoff retry.

        Permanent client errors (any 4xx in ``_NON_RETRYABLE``) are raised
        immediately without retrying.
        """
        last_exc: Exception | None = None

        for attempt in range(_MAX_HTTP_ATTEMPTS):
            try:
                response = requests.get(
                    url,
                    headers=self._data_headers,
                    params=params,
                    timeout=20,
                )
                response.raise_for_status()
                return response

            except requests.HTTPError as exc:
                if (
                    exc.response is not None
                    and exc.response.status_code in _NON_RETRYABLE
                ):
                    raise  # permanent error — do not retry
                last_exc = exc

            except requests.RequestException as exc:
                last_exc = exc

            if attempt < _MAX_HTTP_ATTEMPTS - 1:
                time.sleep(2**attempt)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_position_qty(self, symbol: str) -> float:
        pos = self._get_position(symbol)
        return float(pos.qty) if pos is not None else 0.0

    def get_entry_price(self, symbol: str) -> float | None:
        pos = self._get_position(symbol)
        return float(pos.avg_entry_price) if pos is not None else None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def has_open_order(self, symbol: str) -> bool:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        return len(self.client.get_orders(filter=request)) > 0

    def submit_buy(self, symbol: str, qty: float) -> None:
        self._position_cache = None  # stale after order submission
        self.client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
        )

    def submit_sell(self, symbol: str, qty: float) -> None:
        self._position_cache = None  # stale after order submission
        self.client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
        )

    # ------------------------------------------------------------------
    # Account / market status
    # ------------------------------------------------------------------

    def get_market_status(self) -> tuple[bool, float]:
        # Crypto trades 24/7 — always open
        return True, 0.0

    def get_buying_power(self) -> float:
        return float(self.client.get_account().buying_power)

    def get_equity(self) -> tuple[float, float]:
        account = self.client.get_account()
        return float(account.equity), float(account.last_equity)
