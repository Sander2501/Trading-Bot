"""
brokers
~~~~~~~
Broker implementations for the trading bot.

Available classes
-----------------
BaseBroker      – Abstract interface that all brokers must implement.

CapitalBroker   – Live / demo trading via the Capital.com REST API (CFDs).
BacktestBroker  – Offline replay of a CSV file of historical price bars.

Usage
-----
    from brokers import BaseBroker, CapitalBroker, BacktestBroker
"""

from brokers.base import BaseBroker

from brokers.capital import CapitalBroker
from brokers.backtest import BacktestBroker

__all__ = ["BaseBroker", "CapitalBroker", "BacktestBroker"]
