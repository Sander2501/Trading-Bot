"""
brokers.base
~~~~~~~~~~~~
Abstract base class that defines the broker interface.

All concrete broker implementations (live or simulated) must subclass
``BaseBroker`` and implement every abstract method.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseBroker(ABC):
    """
    Minimal broker interface the bot depends on.
    Implementations may be live (CapitalBroker) or simulated (BacktestBroker).
    """

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    def get_recent_bars(
        self, symbol: str, limit: int, timeframe: str = "1Min"
    ) -> pd.DataFrame:
        """Return the most recent ``limit`` bars as a DataFrame with columns h, l, c."""
        pass

    def get_recent_closes(
        self, symbol: str, limit: int, timeframe: str = "1Min"
    ) -> pd.Series:
        """Return the most recent ``limit`` closing prices as a float Series."""
        return self.get_recent_bars(symbol, limit, timeframe)["c"]

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    @abstractmethod
    def get_position_qty(self, symbol: str) -> float:
        """Return the current held quantity (0.0 if flat)."""
        pass

    @abstractmethod
    def get_entry_price(self, symbol: str) -> float | None:
        """Return the weighted-average entry price, or None if flat."""
        pass

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    def has_open_order(self, symbol: str) -> bool:
        """Return True if there is at least one open order for ``symbol``."""
        pass

    @abstractmethod
    def submit_buy(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        """Submit a market buy order for ``qty`` units of ``symbol`` with optional SL/TP."""
        pass

    @abstractmethod
    def submit_sell(self, symbol: str, qty: float, sl: float | None = None, tp: float | None = None) -> None:
        """Submit a market sell order for ``qty`` units of ``symbol`` with optional SL/TP."""
        pass

    # ------------------------------------------------------------------
    # Account / market status
    # ------------------------------------------------------------------

    @abstractmethod
    def get_market_status(self) -> tuple[bool, float]:
        """
        Return ``(is_open, seconds_until_open)``.
        Crypto brokers should always return ``(True, 0.0)``.
        """
        pass

    @abstractmethod
    def get_buying_power(self) -> float:
        """Return available cash / buying power."""
        pass

    @abstractmethod
    def get_equity(self) -> tuple[float, float]:
        """
        Return ``(current_equity, last_equity)``.
        ``last_equity`` is used as the daily-loss baseline.
        """
        pass

    # ------------------------------------------------------------------
    # Optional lifecycle hooks (no-op defaults)
    # ------------------------------------------------------------------

    def snapshot_day(self) -> None:
        """
        Record today's equity as the daily-loss baseline.

        Override in simulated brokers that maintain their own equity state
        (e.g. BacktestBroker).  Live brokers such as CapitalBroker can leave
        this as a no-op because the broker API already provides a native
        ``last_equity`` value.
        """
        pass

    def flush_position_cache(self) -> None:
        """
        Invalidate any per-cycle position cache.

        Called at the start of each ``run_once`` cycle.  Override in
        implementations that cache ``get_open_position`` results to avoid
        redundant API round-trips within a single cycle (e.g. CapitalBroker).
        The default is a no-op; simulated brokers read internal state directly.
        """
        pass

    @property
    def supports_shorting(self) -> bool:
        """
        Whether this broker can open short (sell) positions from flat.

        Defaults to True for simulated brokers.
        """
        return True
