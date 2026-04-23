"""
brokers
~~~~~~~
Broker implementations for the trading bot.

Available classes
-----------------
BaseBroker      – Abstract interface that all brokers must implement.
AlpacaBroker    – Live / paper trading via the Alpaca API.
BacktestBroker  – Offline replay of a CSV file of historical price bars.

Usage
-----
    from brokers import BaseBroker, AlpacaBroker, BacktestBroker
"""

from brokers.base import BaseBroker
from brokers.alpaca import AlpacaBroker
from brokers.backtest import BacktestBroker

__all__ = ["BaseBroker", "AlpacaBroker", "BacktestBroker"]
