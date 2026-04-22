import pandas as pd

from base_broker import BaseBroker


class BacktestBroker(BaseBroker):
    """
    Replays a CSV of historical bars so the bot's signal logic can be tested
    offline. Expected CSV columns: `t` (timestamp) and `c` (close price).

    `advance()` moves the simulated clock forward one bar. All read methods
    return state as of the current cursor.
    """

    def __init__(
        self,
        csv_path: str,
        symbol: str,
        starting_cash: float = 100_000.0,
    ):
        df = pd.read_csv(csv_path)
        if "t" in df.columns:
            df = df.sort_values("t").reset_index(drop=True)
        self._closes: pd.Series = df["c"].astype(float)
        self._symbol = symbol
        self._cursor = 0
        self._qty = 0
        self._entry_price: float | None = None
        self._cash = starting_cash
        self._starting_equity = starting_cash
        self._last_equity = starting_cash
        self.trades: list[dict] = []

    # -------- simulation controls --------

    def advance(self) -> bool:
        self._cursor += 1
        return self._cursor < len(self._closes)

    def done(self) -> bool:
        return self._cursor >= len(self._closes)

    def snapshot_day(self) -> None:
        """Call at session boundaries to freeze last_equity."""
        self._last_equity = self._equity()

    # -------- BaseBroker implementation --------

    def get_recent_closes(
        self, symbol: str, limit: int = 30, timeframe: str = "1Min"
    ) -> pd.Series:
        self._check_symbol(symbol)
        end = self._cursor + 1
        start = max(0, end - limit)
        return self._closes.iloc[start:end].reset_index(drop=True)

    def get_position_qty(self, symbol: str) -> int:
        self._check_symbol(symbol)
        return self._qty

    def get_entry_price(self, symbol: str) -> float | None:
        self._check_symbol(symbol)
        return self._entry_price

    def has_open_order(self, symbol: str) -> bool:
        return False

    def submit_buy(self, symbol: str, qty: int) -> None:
        self._check_symbol(symbol)
        price = self._current_price()
        self._cash -= price * qty
        if self._qty == 0:
            self._entry_price = price
        else:
            total_cost = (self._entry_price or 0) * self._qty + price * qty
            self._entry_price = total_cost / (self._qty + qty)
        self._qty += qty
        self.trades.append({"side": "BUY", "qty": qty, "price": price, "t": self._cursor})

    def submit_sell(self, symbol: str, qty: int) -> None:
        self._check_symbol(symbol)
        price = self._current_price()
        self._cash += price * qty
        self._qty -= qty
        if self._qty == 0:
            self._entry_price = None
        self.trades.append({"side": "SELL", "qty": qty, "price": price, "t": self._cursor})

    def get_market_status(self) -> tuple[bool, float]:
        return True, 0.0

    def get_buying_power(self) -> float:
        return max(0.0, self._cash)

    def get_equity(self) -> tuple[float, float]:
        return self._equity(), self._last_equity

    # -------- helpers --------

    def _current_price(self) -> float:
        return float(self._closes.iloc[self._cursor])

    def _equity(self) -> float:
        return self._cash + self._qty * self._current_price()

    def _check_symbol(self, symbol: str) -> None:
        if symbol != self._symbol:
            raise ValueError(
                f"BacktestBroker configured for {self._symbol!r}, got {symbol!r}"
            )
