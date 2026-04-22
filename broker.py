import os
import requests
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")

DATA_HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}


class AlpacaBroker:
    def __init__(self):
        self.client = TradingClient(API_KEY, API_SECRET, paper=True)

    def get_recent_closes(self, symbol: str, limit: int = 30) -> pd.Series:
        url = "https://data.alpaca.markets/v2/stocks/bars"
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "limit": limit,
        }

        response = requests.get(url, headers=DATA_HEADERS, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        bars = payload["bars"].get(symbol, [])
        if not bars:
            raise ValueError(f"No bars returned for {symbol}")

        closes = [bar["c"] for bar in bars]
        return pd.Series(closes, dtype=float)

    def get_position_qty(self, symbol: str) -> int:
        positions = self.client.get_all_positions()

        for position in positions:
            if position.symbol == symbol:
                return int(float(position.qty))

        return 0

    def has_open_order(self, symbol: str) -> bool:
        orders = self.client.get_orders(filter=QueryOrderStatus.OPEN)

        for order in orders:
            if order.symbol == symbol:
                return True

        return False

    def submit_buy(self, symbol: str, qty: int) -> None:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        self.client.submit_order(order_data=order)

    def submit_sell(self, symbol: str, qty: int) -> None:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        self.client.submit_order(order_data=order)

    def is_market_open(self) -> bool:
        clock = self.client.get_clock()
        return bool(clock.is_open)