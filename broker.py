import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from base_broker import BaseBroker

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")

DATA_HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
MAX_HTTP_ATTEMPTS = 3


class AlpacaBroker(BaseBroker):
    def __init__(self):
        self.client = TradingClient(API_KEY, API_SECRET, paper=True)

    def get_recent_closes(
        self, symbol: str, limit: int = 30, timeframe: str = "1Min"
    ) -> pd.Series:
        params = {"symbols": symbol, "timeframe": timeframe, "limit": limit}

        payload = self._get_with_retry(BARS_URL, params).json()
        bars = payload.get("bars", {}).get(symbol, [])
        if not bars:
            raise ValueError(f"No bars returned for {symbol}")

        # Don't trust the API to return bars in chronological order.
        bars = sorted(bars, key=lambda b: b["t"])
        closes = [bar["c"] for bar in bars]
        return pd.Series(closes, dtype=float)

    @staticmethod
    def _get_with_retry(url: str, params: dict) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(MAX_HTTP_ATTEMPTS):
            try:
                response = requests.get(
                    url, headers=DATA_HEADERS, params=params, timeout=20
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == MAX_HTTP_ATTEMPTS - 1:
                    break
                time.sleep(2**attempt)
        raise last_exc  # type: ignore[misc]

    def get_position_qty(self, symbol: str) -> int:
        try:
            position = self.client.get_open_position(symbol)
            return int(float(position.qty))
        except Exception:
            return 0

    def get_entry_price(self, symbol: str) -> float | None:
        try:
            position = self.client.get_open_position(symbol)
            return float(position.avg_entry_price)
        except Exception:
            return None

    def has_open_order(self, symbol: str) -> bool:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = self.client.get_orders(filter=request)
        return len(orders) > 0

    def submit_buy(self, symbol: str, qty: int) -> None:
        self.client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )

    def submit_sell(self, symbol: str, qty: int) -> None:
        self.client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )

    def get_market_status(self) -> tuple[bool, float]:
        """Returns (is_open, seconds_until_open). Single API call."""
        clock = self.client.get_clock()
        if clock.is_open:
            return True, 0.0
        delta = clock.next_open - clock.timestamp
        return False, max(0.0, delta.total_seconds())

    def get_buying_power(self) -> float:
        return float(self.client.get_account().buying_power)

    def get_equity(self) -> tuple[float, float]:
        account = self.client.get_account()
        return float(account.equity), float(account.last_equity)
