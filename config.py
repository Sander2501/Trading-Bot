"""
config
~~~~~~
Central configuration module.

All environment variables are loaded and validated here.  Every other module
imports constants from this file instead of calling ``os.getenv`` directly,
which keeps configuration in one place and avoids scattered ``load_dotenv``
calls throughout the codebase.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Asset & timeframe ─────────────────────────────────────────────────────────
SYMBOL: str = os.getenv("SYMBOL", "BTC/USD")
TIMEFRAME: str = os.getenv("TIMEFRAME", "1Min")

# ── Strategy: EMA crossover + RSI filter ─────────────────────────────────────
#: Legacy single-MA window forwarded to strategy for backward compatibility.
WINDOW: int = int(os.getenv("WINDOW", "20"))

#: Number of bars the EMA crossover must persist before a signal fires.
CONFIRM_BARS: int = int(os.getenv("CONFIRM_BARS", "1"))

#: Fast and slow EMA periods for the crossover signal.
FAST_WINDOW: int = int(os.getenv("FAST_WINDOW", "9"))
SLOW_WINDOW: int = int(os.getenv("SLOW_WINDOW", "21"))

#: RSI parameters for the overbought/oversold gate.
RSI_WINDOW: int = int(os.getenv("RSI_WINDOW", "14"))
RSI_OVERBOUGHT: float = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD: float = float(os.getenv("RSI_OVERSOLD", "30"))

# ── Risk management ───────────────────────────────────────────────────────────
#: Fraction of current equity to allocate per trade.
RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.02"))

#: Exit a long position when price drops this far below entry.
STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.02"))

#: Halt trading for the day once intraday drawdown exceeds this threshold.
MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))

# ── Timing ────────────────────────────────────────────────────────────────────
#: Seconds to sleep between each ``run_once`` cycle.
CHECK_INTERVAL_SECONDS: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))

#: Hard cap on exponential-backoff sleep after repeated errors.
MAX_BACKOFF_SECONDS: int = 600

# ── Backtest ──────────────────────────────────────────────────────────────────
#: Path to the CSV file used by BacktestBroker.
CSV_PATH: str = os.getenv("BACKTEST_CSV", "historical_data.csv")

#: Starting cash for the simulated backtest account.
STARTING_CASH: float = float(os.getenv("BACKTEST_STARTING_CASH", "100000"))
