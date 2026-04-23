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

# ── Strategy: triple EMA + MACD + ADX + RSI ──────────────────────────────────
#: Legacy single-MA window (unused by current strategy; kept for compat).
WINDOW: int = int(os.getenv("WINDOW", "20"))

#: Bars the EMA crossover must persist before a signal fires.
#: 2 reduces whipsaws on noisy 1-minute BTC data vs the old default of 1.
CONFIRM_BARS: int = int(os.getenv("CONFIRM_BARS", "2"))

#: Fast / slow EMA for the crossover signal.
FAST_WINDOW: int = int(os.getenv("FAST_WINDOW", "9"))
SLOW_WINDOW: int = int(os.getenv("SLOW_WINDOW", "21"))

#: Third (trend) EMA — all three must align before a signal fires.
#: Prevents counter-trend entries when price is on the wrong side of the 50-bar mean.
TREND_WINDOW: int = int(os.getenv("TREND_WINDOW", "50"))

#: RSI parameters.  BTC regularly runs to RSI 80+ in bull legs without reversing,
#: so 75/25 avoids blocking valid entries that 70/30 would filter out.
RSI_WINDOW: int = int(os.getenv("RSI_WINDOW", "14"))
RSI_OVERBOUGHT: float = float(os.getenv("RSI_OVERBOUGHT", "75"))
RSI_OVERSOLD: float = float(os.getenv("RSI_OVERSOLD", "25"))

#: MACD parameters.  Histogram direction must confirm the EMA crossover direction.
MACD_FAST: int = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW: int = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL_WINDOW: int = int(os.getenv("MACD_SIGNAL_WINDOW", "9"))

#: ADX trend-strength filter.  Signals are suppressed when ADX < threshold,
#: which eliminates most false crossovers in sideways/choppy markets.
ADX_WINDOW: int = int(os.getenv("ADX_WINDOW", "14"))
ADX_THRESHOLD: float = float(os.getenv("ADX_THRESHOLD", "20.0"))

# ── Risk management ───────────────────────────────────────────────────────────
#: Fraction of current equity to allocate per trade.
RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.02"))

#: ATR-based stop: exit when price falls more than ATR_STOP_MULT × ATR below entry.
#: A multiplier of 2.0 gives the trade enough room to breathe on BTC 1-min noise
#: without letting a real move run too far against us.
ATR_STOP_MULT: float = float(os.getenv("ATR_STOP_MULT", "2.0"))
ATR_STOP_WINDOW: int = int(os.getenv("ATR_STOP_WINDOW", "14"))

#: Minimum stop distance as a fraction of entry price (floor for low-volatility periods).
STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.005"))

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
