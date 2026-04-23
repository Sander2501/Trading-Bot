"""
brokers.backtest
~~~~~~~~~~~~~~~~
Simulated broker that replays a CSV of historical price bars.

Designed to be a drop-in replacement for AlpacaBroker so that the exact same
``run_once`` loop from ``main.py`` can be used for both live trading and
offline backtesting without any code changes.
"""

import pandas as pd

from brokers.base import BaseBroker


class BacktestBroker(BaseBroker):
    """
    Replays a CSV of historical bars so the bot's signal logic can be tested
    offline.

    Expected CSV columns
    --------------------
    t : timestamp (optional, used for sorting)
    c : close price (required)

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    symbol:
        The symbol string the broker is configured for.  Any method called
        with a different symbol raises ``ValueError``.
    starting_cash:
        Starting account balance in USD.
    slippage_pct:
        Fraction of the close price added (buys) or subtracted (sells) to
        simulate bid/ask spread.  Defaults to 0.05 %.
    commission_per_trade:
        Flat fee in USD deducted from each fill.  Defaults to $0.
    """

    def __init__(
        self,
        csv_path: str,
        symbol: str,
        starting_cash: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_per_trade: float = 0.0,
    ) -> None:
        df = pd.read_csv(csv_path)

        if "c" not in df.columns:
            raise ValueError("CSV must contain a 'c' column for close prices.")

        if "t" in df.columns:
            df = df.sort_values("t").reset_index(drop=True)

        self._closes: pd.Series = df["c"].astype(float)
        self._symbol = symbol
        self._cursor = 0

        self._qty: float = 0.0
        self._entry_price: float | None = None
        self._cash: float = starting_cash
        self._starting_equity: float = starting_cash
        self._last_equity: float = starting_cash

        self._slippage_pct = slippage_pct
        self._commission = commission_per_trade

        #: All fills recorded during the run.
        self.trades: list[dict] = []

        #: Equity snapshot after each bar, populated by :meth:`advance`.
        self.equity_curve: list[float] = []

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def advance(self) -> bool:
        """
        Snapshot equity for the current bar then move the cursor forward.
        Returns ``True`` while more bars remain.
        """
        self.equity_curve.append(self._equity())
        self._cursor += 1
        return self._cursor < len(self._closes)

    def done(self) -> bool:
        """Return ``True`` when all bars have been processed."""
        return self._cursor >= len(self._closes)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_recent_closes(
        self, symbol: str, limit: int = 30, timeframe: str = "1Min"
    ) -> pd.Series:
        self._check_symbol(symbol)
        end = self._cursor + 1
        start = max(0, end - limit)
        return self._closes.iloc[start:end].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_position_qty(self, symbol: str) -> float:
        self._check_symbol(symbol)
        return self._qty

    def get_entry_price(self, symbol: str) -> float | None:
        self._check_symbol(symbol)
        return self._entry_price

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def has_open_order(self, symbol: str) -> bool:
        self._check_symbol(symbol)
        return False  # simulated orders fill instantly

    def submit_buy(self, symbol: str, qty: float) -> None:
        self._check_symbol(symbol)

        if qty <= 0:
            raise ValueError("Buy quantity must be > 0")

        price = self._current_price() * (1 + self._slippage_pct)
        cost = price * qty + self._commission

        if cost > self._cash:
            raise ValueError(
                f"Not enough cash to buy {qty} units at {price:.2f}. "
                f"Cash available: {self._cash:.2f}"
            )

        self._cash -= cost

        # Weighted-average entry price for pyramid positions
        if self._qty == 0:
            self._entry_price = price
        else:
            total_cost = (self._entry_price or 0.0) * self._qty + cost
            self._entry_price = total_cost / (self._qty + qty)

        self._qty += qty
        self.trades.append(self._fill("BUY", qty, price))

    def submit_sell(self, symbol: str, qty: float) -> None:
        self._check_symbol(symbol)

        if qty <= 0:
            raise ValueError("Sell quantity must be > 0")
        if qty > self._qty:
            raise ValueError(
                f"Cannot sell {qty} units; only {self._qty} currently held."
            )

        price = self._current_price() * (1 - self._slippage_pct)
        proceeds = price * qty - self._commission

        self._cash += proceeds
        self._qty -= qty

        if self._qty == 0:
            self._entry_price = None

        self.trades.append(self._fill("SELL", qty, price))

    # ------------------------------------------------------------------
    # Account / market status
    # ------------------------------------------------------------------

    def get_market_status(self) -> tuple[bool, float]:
        return True, 0.0

    def get_buying_power(self) -> float:
        return max(0.0, self._cash)

    def get_equity(self) -> tuple[float, float]:
        return self._equity(), self._last_equity

    def snapshot_day(self) -> None:
        """Record current equity as the daily-loss baseline."""
        self._last_equity = self._equity()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_price(self) -> float:
        idx = min(self._cursor, len(self._closes) - 1)
        return float(self._closes.iloc[idx])

    def _equity(self) -> float:
        return self._cash + self._qty * self._current_price()

    def _fill(self, side: str, qty: float, price: float) -> dict:
        return {
            "side": side,
            "qty": qty,
            "price": price,
            "t": self._cursor,
            "equity": self._equity(),
        }

    def _check_symbol(self, symbol: str) -> None:
        if symbol != self._symbol:
            raise ValueError(
                f"BacktestBroker is configured for {self._symbol!r}, "
                f"got {symbol!r}"
            )
