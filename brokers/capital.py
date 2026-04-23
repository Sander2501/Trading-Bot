"""
brokers.capital
~~~~~~~~~~~~~~~
Live and demo-trading broker implementation backed by the Capital.com REST API.
"""

import logging
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from brokers.base import BaseBroker

load_dotenv()

logger = logging.getLogger(__name__)

_DEMO_BASE_URL = "https://demo-api-capital.backend-capital.com"
_LIVE_BASE_URL = "https://api-capital.backend-capital.com"

# Map bot symbol names to Capital.com epics
_EPIC_MAP: dict[str, str] = {
    "BTC/USD": "BTCUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD",
    "LTC/USD": "LTCUSD",
}

# Map bot timeframe strings to Capital.com resolution names
_TF_MAP: dict[str, str] = {
    "1Min":  "MINUTE",
    "5Min":  "MINUTE_5",
    "15Min": "MINUTE_15",
    "30Min": "MINUTE_30",
    "1H":    "HOUR",
    "4H":    "HOUR_4",
    "1D":    "DAY",
    "1W":    "WEEK",
}

_MAX_HTTP_ATTEMPTS = 3
_NON_RETRYABLE = frozenset({400, 403, 404, 422})


def _mid(field: dict | None) -> float:
    """Return (bid + ask) / 2 from a Capital.com price field."""
    if not field:
        return 0.0
    bid = float(field.get("bid") or 0.0)
    ask = float(field.get("ask") or bid)
    return (bid + ask) / 2.0


class CapitalBroker(BaseBroker):
    """
    Broker implementation for Capital.com (CFD trading).

    Credentials are read from environment variables:
        CAPITAL_API_KEY     – API key generated in the Capital.com dashboard
        CAPITAL_IDENTIFIER  – Account login (email address)
        CAPITAL_PASSWORD    – Account password
        CAPITAL_USE_DEMO    – "true" uses the demo environment (default); "false" goes live

    CFDs support both long and short positions, so ``supports_shorting`` is True.

    Session tokens (CST + X-SECURITY-TOKEN) are obtained on construction and
    transparently refreshed on 401 responses.
    """

    def __init__(self) -> None:
        api_key    = os.getenv("CAPITAL_API_KEY")
        identifier = os.getenv("CAPITAL_IDENTIFIER")
        password   = os.getenv("CAPITAL_PASSWORD")
        use_demo   = os.getenv("CAPITAL_USE_DEMO", "true").lower() != "false"

        if not api_key or not identifier or not password:
            raise ValueError(
                "Missing CAPITAL_API_KEY, CAPITAL_IDENTIFIER, or CAPITAL_PASSWORD. "
                "Add them to your .env file."
            )

        self._api_key    = api_key
        self._identifier = identifier
        self._password   = password
        self._base_url   = _DEMO_BASE_URL if use_demo else _LIVE_BASE_URL

        self._cst:            str = ""
        self._security_token: str = ""

        # Equity snapshot for daily-loss baseline (Capital.com has no native last_equity)
        self._last_equity: float | None = None

        # Per-cycle position cache: dict[symbol, position_dict | None], or None (unset)
        self._position_cache: dict | None = None

        self._create_session()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _create_session(self) -> None:
        """Authenticate and store CST + X-SECURITY-TOKEN for subsequent requests."""
        response = requests.post(
            f"{self._base_url}/api/v1/session",
            headers={
                "X-CAP-API-KEY": self._api_key,
                "Content-Type": "application/json",
            },
            json={
                "identifier": self._identifier,
                "password": self._password,
                "encryptedPassword": False,
            },
            timeout=20,
        )
        response.raise_for_status()
        self._cst            = response.headers["CST"]
        self._security_token = response.headers["X-SECURITY-TOKEN"]
        use_demo_env = _DEMO_BASE_URL in self._base_url
        env_label = "demo" if use_demo_env else "live"
        logger.info("Capital.com session established (%s).", "demo" if _DEMO_BASE_URL in self._base_url else "live")

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "X-CAP-API-KEY":    self._api_key,
            "CST":              self._cst,
            "X-SECURITY-TOKEN": self._security_token,
            "Content-Type":     "application/json",
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """HTTP request with automatic session-refresh on 401 and exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_HTTP_ATTEMPTS):
            try:
                response = requests.request(
                    method,
                    f"{self._base_url}{path}",
                    headers=self._auth_headers,
                    timeout=20,
                    **kwargs,
                )
                if response.status_code == 401:
                    self._create_session()
                    continue
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in _NON_RETRYABLE:
                    raise
                last_exc = exc
            except requests.RequestException as exc:
                last_exc = exc
            if attempt < _MAX_HTTP_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]

    def _get(self, path: str, params: dict | None = None) -> requests.Response:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict) -> requests.Response:
        return self._request("POST", path, json=body)

    def _delete(self, path: str) -> requests.Response:
        return self._request("DELETE", path)

    # ------------------------------------------------------------------
    # Position cache
    # ------------------------------------------------------------------

    def flush_position_cache(self) -> None:
        """Invalidate the position cache at the start of each trading cycle."""
        self._position_cache = None

    def _get_open_position(self, symbol: str) -> dict | None:
        """Return the open position dict for ``symbol``, or None if flat."""
        if self._position_cache is not None:
            return self._position_cache.get(symbol)

        epic      = _EPIC_MAP.get(symbol, symbol)
        positions = self._get("/api/v1/positions").json().get("positions", [])
        pos = next(
            (p["position"] for p in positions if p.get("market", {}).get("epic") == epic),
            None,
        )
        self._position_cache = {symbol: pos}
        return pos

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_recent_bars(
        self, symbol: str, limit: int = 30, timeframe: str = "5Min"
    ) -> pd.DataFrame:
        """Return the most recent ``limit`` bars as a DataFrame with columns h, l, c."""
        epic       = _EPIC_MAP.get(symbol, symbol)
        resolution = _TF_MAP.get(timeframe, "MINUTE_5")

        data   = self._get(f"/api/v1/prices/{epic}", params={"resolution": resolution, "max": limit}).json()
        prices = data.get("prices", [])

        if not prices:
            raise ValueError(
                f"No price data returned for {symbol!r} (epic={epic!r}). "
                "Check your Capital.com subscription or the epic name in _EPIC_MAP."
            )

        rows = [
            {
                "h": _mid(p.get("highPrice")),
                "l": _mid(p.get("lowPrice")),
                "c": _mid(p.get("closePrice")),
            }
            for p in prices
        ]
        return pd.DataFrame(rows, columns=["h", "l", "c"])

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_position_qty(self, symbol: str) -> float:
        """Positive = long, negative = short, 0 = flat."""
        pos = self._get_open_position(symbol)
        if pos is None:
            return 0.0
        size = float(pos.get("size", 0))
        return size if pos.get("direction") == "BUY" else -size

    def get_entry_price(self, symbol: str) -> float | None:
        pos = self._get_open_position(symbol)
        return float(pos["level"]) if pos else None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def has_open_order(self, symbol: str) -> bool:
        epic   = _EPIC_MAP.get(symbol, symbol)
        orders = self._get("/api/v1/workingorders").json().get("workingOrders", [])
        return any(o.get("workingOrderData", {}).get("epic") == epic for o in orders)

    def submit_buy(self, symbol: str, qty: float) -> None:
        """
        Open a long, or close an existing short.
        Uses the cached position to avoid an extra API round-trip.
        """
        pos = self._get_open_position(symbol)
        if pos and pos.get("direction") == "SELL":
            self._delete(f"/api/v1/positions/{pos['dealId']}")
        else:
            self._post("/api/v1/positions", {
                "epic":          _EPIC_MAP.get(symbol, symbol),
                "direction":     "BUY",
                "size":          qty,
                "guaranteedStop": False,
            })
        self._position_cache = None

    def submit_sell(self, symbol: str, qty: float) -> None:
        """
        Close an existing long, or open a short.
        Uses the cached position to avoid an extra API round-trip.
        """
        pos = self._get_open_position(symbol)
        if pos and pos.get("direction") == "BUY":
            self._delete(f"/api/v1/positions/{pos['dealId']}")
        else:
            self._post("/api/v1/positions", {
                "epic":          _EPIC_MAP.get(symbol, symbol),
                "direction":     "SELL",
                "size":          qty,
                "guaranteedStop": False,
            })
        self._position_cache = None

    # ------------------------------------------------------------------
    # Account / market status
    # ------------------------------------------------------------------

    @property
    def supports_shorting(self) -> bool:
        return True  # CFDs allow both long and short positions

    def get_market_status(self) -> tuple[bool, float]:
        return True, 0.0  # crypto CFDs trade 24/7

    def get_buying_power(self) -> float:
        accounts = self._get("/api/v1/accounts").json().get("accounts", [])
        if not accounts:
            return 0.0
        return float(accounts[0]["balance"].get("available", 0))

    def get_equity(self) -> tuple[float, float]:
        accounts = self._get("/api/v1/accounts").json().get("accounts", [])
        if not accounts:
            return 0.0, 0.0
        bal    = accounts[0]["balance"]
        equity = float(bal.get("balance", 0)) + float(bal.get("profitLoss", 0))
        last   = self._last_equity if self._last_equity is not None else equity
        return equity, last

    def snapshot_day(self) -> None:
        """Record current equity as today's daily-loss baseline."""
        equity, _ = self.get_equity()
        self._last_equity = equity
