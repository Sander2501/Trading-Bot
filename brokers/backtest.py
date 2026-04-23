"""
brokers.backtest
~~~~~~~~~~~~~~~~
Simulated broker that replays a CSV of historical price bars.

Designed to be a drop-in replacement for AlpacaBroker so that the exact same
``run_once`` loop from ``main.py`` can be used for both live trading and
offline backtesting without any code changes.
"""

import logging
from datetime import date

import pandas as pd

from brokers.base import BaseBroker

logger = logging.getLogger(__name__)


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
        if starting_cash <= 0:
            raise ValueError(
                f"starting_cash must be > 0, got {starting_cash}"
            )
        if slippage_pct < 0 or slippage_pct > 0.1:
            raise ValueError(
                f"slippage_pct must be in [0, 0.1], got {slippage_pct}"
            )
        if commission_per_trade < 0:
            raise ValueError(
                f"commission_per_trade must be >= 0, got {commission_per_trade}"
            )

        df = pd.read_csv(csv_path)

        if "c" not in df.columns:
            raise ValueError("CSV must contain a 'c' column for close prices.")

        if "t" in df.columns:
            df = df.sort_values("t").reset_index(drop=True)
            # Parse timestamps once so current_date() is cheap per cycle.
            self._timestamps: pd.Series | None = pd.to_datetime(df["t"], utc=True, errors="coerce")
        else:
            self._timestamps = None

        self._closes: pd.Series = df["c"].astype(float)
        self._highs: pd.Series = df["h"].astype(float) if "h" in df.columns else self._closes.copy()
        self._lows:  pd.Series = df["l"].astype(float) if "l" in df.columns else self._closes.copy()
        self._symbol = symbol
        self._cursor = 0

        self._qty: float = 0.0
        self._entry_price: float | None = None
        self._cash: float = starting_cash
        self._starting_equity: float = starting_cash
        self._last_equity: float = starting_cash

        self._slippage_pct = slippage_pct
        self._commission = commission_per_trade

        # Pending standing orders (set when a position is opened with SL/TP).
        self._sl_price: float | None = None
        self._tp_price: float | None = None

        #: All fills recorded during the run.
        self.trades: list[dict] = []

        #: Equity snapshot after each bar, populated by :meth:`advance`.
        self.equity_curve: list[float] = []

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def advance(self) -> bool:
        """
        Snapshot equity for the current bar, move the cursor forward, then
        simulate any standing SL/TP orders against the new bar's high/low.
        Returns ``True`` while more bars remain.
        """
        self.equity_curve.append(self._equity())
        self._cursor += 1
        if self._qty != 0 and self._cursor < len(self._closes):
            self._check_sl_tp()
        return self._cursor < len(self._closes)

    def done(self) -> bool:
        """Return ``True`` when all bars have been processed."""
        return self._cursor >= len(self._closes)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_recent_bars(
        self, symbol: str, limit: int = 30, timeframe: str = "1Min"  # noqa: ARG002
    ) -> pd.DataFrame:
        self._check_symbol(symbol)
        end = self._cursor + 1
        start = max(0, end - limit)
        return pd.DataFrame({
            "h": self._highs.iloc[start:end].values,
            "l": self._lows.iloc[start:end].values,
            "c": self._closes.iloc[start:end].values,
        })

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

    def submit_buy(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        self._check_symbol(symbol)
        if qty <= 0:
            raise ValueError("Buy quantity must be > 0")

        price = self._current_price() * (1 + self._slippage_pct)
        cost = price * qty + self._commission

        # Reject orders where the total cost exceeds available cash and there
        # is no existing short position to close.  (Covering a short is always
        # permitted because it reduces exposure rather than increasing it.)
        if self._qty >= 0 and cost > self._cash:
            raise ValueError(
                f"Insufficient cash: cost ({cost:.2f}) exceeds cash ({self._cash:.2f})"
            )

        prev_qty = self._qty
        self._cash -= cost
        self._qty += qty

        if prev_qty > 0:
            # Adding to an existing long — weighted average entry
            total_cost = (self._entry_price or 0.0) * prev_qty + cost
            self._entry_price = total_cost / self._qty
        elif prev_qty <= 0 and self._qty > 0:
            # Flat→long, or short cover that flipped to long
            self._entry_price = price
        elif self._qty == 0:
            # Covered short entirely — now flat
            self._entry_price = None
        # partial short cover (prev_qty < 0, self._qty still < 0): entry unchanged

        # Store pending standing orders for the new long, clear on cover/close.
        if self._qty > 0:
            self._sl_price = sl
            self._tp_price = tp
        else:
            self._sl_price = None
            self._tp_price = None

        side = "COVER" if prev_qty < 0 else "BUY"
        self.trades.append(self._fill(side, qty, price))

    def submit_sell(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        self._check_symbol(symbol)
        if qty <= 0:
            raise ValueError("Sell quantity must be > 0")

        # Cannot sell more than held when already long
        if self._qty > 0 and qty > self._qty:
            raise ValueError(
                f"Cannot sell {qty} units; only {self._qty} currently held."
            )

        price = self._current_price() * (1 - self._slippage_pct)
        proceeds = price * qty - self._commission

        prev_qty = self._qty
        self._cash += proceeds
        self._qty -= qty

        if prev_qty == 0:
            # Flat→short: record the short entry price
            self._entry_price = price
        elif self._qty == 0:
            # Closed entire long position
            self._entry_price = None
        # Partial long close: entry_price unchanged for the remaining position

        # Store pending standing orders for the new short, clear on close.
        if self._qty < 0:
            self._sl_price = sl
            self._tp_price = tp
        else:
            self._sl_price = None
            self._tp_price = None

        side = "SHORT" if prev_qty == 0 else "SELL"
        self.trades.append(self._fill(side, qty, price))

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

    def current_date(self) -> date:
        """Return the date of the current bar so daily-loss rolls over per simulated day."""
        if self._timestamps is None:
            return date.today()
        idx = min(self._cursor, len(self._timestamps) - 1)
        ts = self._timestamps.iloc[idx]
        if pd.isna(ts):
            return date.today()
        return ts.date()

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

    def _check_sl_tp(self) -> None:
        """Simulate standing SL/TP orders against the current bar's high/low.

        Called from :meth:`advance` after the cursor moves to a new bar.
        SL takes priority over TP when both are hit on the same bar (worst-case
        assumption is more conservative for backtesting purposes).
        """
        if self._sl_price is None and self._tp_price is None:
            return

        high = float(self._highs.iloc[self._cursor])
        low  = float(self._lows.iloc[self._cursor])
        qty  = abs(self._qty)

        if self._qty > 0:  # Long position
            sl_hit = self._sl_price is not None and low  <= self._sl_price
            tp_hit = self._tp_price is not None and high >= self._tp_price
            if sl_hit:
                fill = self._sl_price
                self._close_position_at(fill, qty, "SL_STOP")
                logger.info("SL triggered on long @ %.4f", fill)
            elif tp_hit:
                fill = self._tp_price
                self._close_position_at(fill, qty, "TP_STOP")
                logger.info("TP triggered on long @ %.4f", fill)

        elif self._qty < 0:  # Short position
            sl_hit = self._sl_price is not None and high >= self._sl_price
            tp_hit = self._tp_price is not None and low  <= self._tp_price
            if sl_hit:
                fill = self._sl_price
                self._cover_position_at(fill, qty, "SL_STOP")
                logger.info("SL triggered on short @ %.4f", fill)
            elif tp_hit:
                fill = self._tp_price
                self._cover_position_at(fill, qty, "TP_STOP")
                logger.info("TP triggered on short @ %.4f", fill)

    def _close_position_at(self, price: float, qty: float, side: str) -> None:
        """Close a long position at an exact price (SL/TP fill, no slippage)."""
        self._cash += price * qty - self._commission
        self._qty = 0.0
        self._entry_price = None
        self._sl_price = None
        self._tp_price = None
        self.trades.append(self._fill(side, qty, price))

    def _cover_position_at(self, price: float, qty: float, side: str) -> None:
        """Cover a short position at an exact price (SL/TP fill, no slippage)."""
        self._cash -= price * qty + self._commission
        self._qty = 0.0
        self._entry_price = None
        self._sl_price = None
        self._tp_price = None
        self.trades.append(self._fill(side, qty, price))

    def _check_symbol(self, symbol: str) -> None:
        if symbol != self._symbol:
            raise ValueError(
                f"BacktestBroker is configured for {self._symbol!r}, "
                f"got {symbol!r}"
            )
