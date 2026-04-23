"""
metrics
~~~~~~~
Trade performance tracking for the Trading Bot.

``TradeMetrics`` is a lightweight dataclass that accumulates statistics about
filled trades.  It can be used standalone or integrated with
``BacktestBroker`` by passing it to the helper methods below.

Example usage with BacktestBroker::

    from metrics import TradeMetrics
    from brokers import BacktestBroker

    broker = BacktestBroker(csv_path="data.csv", symbol="BTC/USD")
    # ... run backtest ...
    metrics = TradeMetrics.from_trades(broker.trades, broker.equity_curve)
    print(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeMetrics:
    """
    Aggregated performance statistics for a completed (or in-progress) run.

    Attributes
    ----------
    total_trades:
        Total number of fills recorded (each BUY/SELL/SHORT/COVER counts as one).
    winning_trades:
        Number of fills that resulted in a positive P&L relative to the
        preceding fill's equity.
    losing_trades:
        Number of fills that resulted in a negative P&L.
    total_pnl:
        Sum of all per-trade P&L values in account currency.
    max_drawdown:
        Largest peak-to-trough decline in equity observed across the run,
        expressed as a positive fraction (e.g. 0.05 means 5 % drawdown).
    win_rate:
        Fraction of winning trades: ``winning_trades / total_trades``.
        ``0.0`` when ``total_trades == 0``.
    avg_win:
        Average P&L of winning trades.  ``0.0`` when there are none.
    avg_loss:
        Average P&L of losing trades (expressed as a negative number).
        ``0.0`` when there are none.
    last_updated:
        Timestamp of the most recent call to :meth:`update` or
        :meth:`from_trades`.
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def update(self, trades: list[dict], equity_curve: list[float]) -> None:
        """
        Recompute all metrics from *trades* and *equity_curve* in-place.

        Parameters
        ----------
        trades:
            List of fill dictionaries as produced by
            ``BacktestBroker.trades``.  Each dict must have an
            ``"equity"`` key with the account equity immediately after
            the fill.
        equity_curve:
            List of equity values sampled once per bar (as produced by
            ``BacktestBroker.equity_curve``).  Used to compute
            ``max_drawdown``.
        """
        if not trades:
            self.last_updated = datetime.utcnow()
            return

        self.total_trades = len(trades)

        # Per-trade P&L: difference in equity between consecutive fills
        wins: list[float] = []
        losses: list[float] = []
        prev_equity = trades[0]["equity"]
        for trade in trades[1:]:
            pnl = trade["equity"] - prev_equity
            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(pnl)
            prev_equity = trade["equity"]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.total_pnl = sum(wins) + sum(losses)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades else 0.0
        self.avg_win = sum(wins) / len(wins) if wins else 0.0
        self.avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.max_drawdown = _max_drawdown(equity_curve)
        self.last_updated = datetime.utcnow()

    @classmethod
    def from_trades(
        cls, trades: list[dict], equity_curve: list[float]
    ) -> "TradeMetrics":
        """
        Create a fully-populated ``TradeMetrics`` from *trades* and
        *equity_curve*.

        Convenience constructor that calls :meth:`update` immediately.
        """
        m = cls()
        m.update(trades, equity_curve)
        return m

    def __str__(self) -> str:
        return (
            f"TradeMetrics("
            f"total={self.total_trades}, "
            f"wins={self.winning_trades}, "
            f"losses={self.losing_trades}, "
            f"win_rate={self.win_rate:.1%}, "
            f"total_pnl={self.total_pnl:+,.2f}, "
            f"avg_win={self.avg_win:+,.2f}, "
            f"avg_loss={self.avg_loss:+,.2f}, "
            f"max_drawdown={self.max_drawdown:.2%}"
            f")"
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _max_drawdown(equity_curve: list[float]) -> float:
    """
    Compute the maximum peak-to-trough drawdown from an equity series.

    Returns the drawdown as a positive fraction in [0, 1].
    Returns ``0.0`` for an empty or single-element curve.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd
