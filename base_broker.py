from abc import ABC, abstractmethod

import pandas as pd


class BaseBroker(ABC):
    """
    Minimal broker interface the bot depends on. Implementations may be live
    (AlpacaBroker) or simulated (BacktestBroker).
    """

    @abstractmethod
    def get_recent_closes(
        self, symbol: str, limit: int, timeframe: str = "1Min"
    ) -> pd.Series: ...

    @abstractmethod
    def get_position_qty(self, symbol: str) -> int: ...

    @abstractmethod
    def get_entry_price(self, symbol: str) -> float | None: ...

    @abstractmethod
    def has_open_order(self, symbol: str) -> bool: ...

    @abstractmethod
    def submit_buy(self, symbol: str, qty: int) -> None: ...

    @abstractmethod
    def submit_sell(self, symbol: str, qty: int) -> None: ...

    @abstractmethod
    def get_market_status(self) -> tuple[bool, float]: ...

    @abstractmethod
    def get_buying_power(self) -> float: ...

    @abstractmethod
    def get_equity(self) -> tuple[float, float]:
        """Returns (current_equity, last_equity). Used for daily-loss checks."""
        ...
