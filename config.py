"""
config
~~~~~~
Central configuration module.

All environment variables are loaded and validated here.  Every other module
imports constants from this file instead of calling ``os.getenv`` directly,
which keeps configuration in one place and avoids scattered ``load_dotenv``
calls throughout the codebase.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# ── Asset & timeframe ─────────────────────────────────────────────────────────
SYMBOL: str = os.getenv("SYMBOL", "BTC/USD")
#: Bar interval. 15Min is the default — 1m is too noisy for this strategy on BTC
#: (38% win rate, sub-1 profit factor in testing). Coarser bars reduce friction
#: (fewer fills × fixed commission + slippage) and filter whipsaws.
TIMEFRAME: str = os.getenv("TIMEFRAME", "15Min")

# ── Strategy: triple EMA + MACD + ADX + RSI ──────────────────────────────────
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
#: Raised to 25 to suppress crossover signals during sideways/chop regimes.
#: 20 let too many low-trend-strength setups through in 30-day backtests.
ADX_THRESHOLD: float = float(os.getenv("ADX_THRESHOLD", "25.0"))

#: Volatility floor (ATR / price). Suppress new signals when volatility is too low
#: to cover spread/fees; helps reduce overtrading in micro-chop regimes.
MIN_ATR_PCT: float = float(os.getenv("MIN_ATR_PCT", "0.001"))

#: Enable adaptive percentile-based thresholds for ADX and ATR%.
ADAPTIVE_THRESHOLDS_ENABLED: bool = os.getenv("ADAPTIVE_THRESHOLDS_ENABLED", "false").lower() in {"1", "true", "yes"}
#: Lookback horizon (in days) used to compute adaptive ADX/ATR% percentiles.
ADAPTIVE_LOOKBACK_DAYS: int = int(os.getenv("ADAPTIVE_LOOKBACK_DAYS", "60"))
#: Percentile of recent ADX values used as dynamic trend-strength threshold.
ADAPTIVE_ADX_PERCENTILE: float = float(os.getenv("ADAPTIVE_ADX_PERCENTILE", "60.0"))
#: Percentile of recent ATR% values used as dynamic low-volatility floor.
ADAPTIVE_ATR_PERCENTILE: float = float(os.getenv("ADAPTIVE_ATR_PERCENTILE", "35.0"))
SESSION_FILTER_ENABLED: bool = os.getenv("SESSION_FILTER_ENABLED", "false").lower() in {"1", "true", "yes"}
SESSION_START_HOUR_UTC: int = int(os.getenv("SESSION_START_HOUR_UTC", "7"))
SESSION_END_HOUR_UTC: int = int(os.getenv("SESSION_END_HOUR_UTC", "22"))
ATR_ACCEL_WINDOW: int = int(os.getenv("ATR_ACCEL_WINDOW", "20"))
MIN_ATR_ACCEL: float = float(os.getenv("MIN_ATR_ACCEL", "0.0"))
STRUCTURE_FILTER_ENABLED: bool = os.getenv("STRUCTURE_FILTER_ENABLED", "false").lower() in {"1", "true", "yes"}
STRUCTURE_LOOKBACK: int = int(os.getenv("STRUCTURE_LOOKBACK", "5"))
VOLUME_FILTER_ENABLED: bool = os.getenv("VOLUME_FILTER_ENABLED", "false").lower() in {"1", "true", "yes"}
MIN_VOLUME: float = float(os.getenv("MIN_VOLUME", "0.0"))

#: Allow opening short positions from flat on SELL signals.  Default is False
#: because the strategy's short leg has historically been a net loser on BTC.
#: Long-exit (closing an existing long on a SELL signal) is unaffected by this.
ALLOW_SHORTS: bool = os.getenv("ALLOW_SHORTS", "false").lower() in {"1", "true", "yes"}

# ── Risk management ───────────────────────────────────────────────────────────
# Risk profiles (set RISK_PER_TRADE env var to select):
#   Conservative : 0.01  (1%  per trade, suitable for beginners)
#   Moderate     : 0.02  (2%  per trade, reasonable for experienced traders) ← DEFAULT
#   Aggressive   : 0.05  (5%  per trade, high volatility tolerance required)
#
# Position size formula: qty = (equity * RISK_PER_TRADE) / price
# Example at $100,000 equity, price=$50,000:
#   Moderate:    qty = (100,000 * 0.02) / 50,000 = 0.04 BTC (~$2,000 exposure)
#   Aggressive:  qty = (100,000 * 0.05) / 50,000 = 0.10 BTC (~$5,000 exposure)
#
# Expected max drawdown ranges (approximate, based on historical volatility):
#   Conservative  (1%): ~3-5%  max drawdown under normal conditions
#   Moderate      (2%): ~6-10% max drawdown under normal conditions
#   Aggressive    (5%): ~15-25% max drawdown — only for experienced traders
#
# Hard cap: RISK_PER_TRADE must not exceed 0.10 (10%).  Values above this
# threshold are rejected at startup to prevent accidental over-leveraging.
#: Fraction of current equity to allocate per trade.
RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.02"))

#: ATR-based stop: exit when price falls more than ATR_STOP_MULT × ATR below entry.
#: A multiplier of 2.0 gives the trade enough room to breathe on BTC 1-min noise
#: without letting a real move run too far against us.
ATR_STOP_MULT: float = float(os.getenv("ATR_STOP_MULT", "2.0"))
ATR_STOP_WINDOW: int = int(os.getenv("ATR_STOP_WINDOW", "14"))

#: Minimum stop distance as a fraction of entry price (floor for low-volatility periods).
STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.005"))

#: ATR-based take profit: exit when price rises more than TAKE_PROFIT_MULT × ATR above entry.
TAKE_PROFIT_MULT: float = float(os.getenv("TAKE_PROFIT_MULT", "4.0"))

#: Target profit as a fraction of entry price (floor for high-conviction trades).
TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.01"))

#: Minimum expected reward/risk ratio after estimated costs to allow a new entry.
MIN_EXPECTED_RR: float = float(os.getenv("MIN_EXPECTED_RR", "1.20"))
#: Estimated round-trip execution cost as fraction of notional (spread+slippage+fees).
ESTIMATED_ROUND_TRIP_COST_PCT: float = float(os.getenv("ESTIMATED_ROUND_TRIP_COST_PCT", "0.0015"))

#: Promote stop to break-even once unrealized PnL reaches this R multiple.
BREAK_EVEN_R_MULT: float = float(os.getenv("BREAK_EVEN_R_MULT", "1.0"))
#: Take partial profits at this R multiple (fraction configured below).
PARTIAL_TP1_R: float = float(os.getenv("PARTIAL_TP1_R", "1.5"))
#: Fraction of open size to close at PARTIAL_TP1_R.
PARTIAL_TP1_FRACTION: float = float(os.getenv("PARTIAL_TP1_FRACTION", "0.50"))
#: Optional second partial take-profit R multiple.
PARTIAL_TP2_R: float = float(os.getenv("PARTIAL_TP2_R", "2.0"))
#: Fraction of open size to close at PARTIAL_TP2_R.
PARTIAL_TP2_FRACTION: float = float(os.getenv("PARTIAL_TP2_FRACTION", "0.50"))

#: Halt trading for the day once intraday drawdown exceeds this threshold.
MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))

# ── Backtest Realism ──────────────────────────────────────────────────────────
#: Fraction of the close price added (buys) or subtracted (sells) to simulate spread.
SLIPPAGE_PCT: float = float(os.getenv("SLIPPAGE_PCT", "0.0005"))

#: Flat fee in USD deducted from each fill.
COMMISSION_PER_TRADE: float = float(os.getenv("COMMISSION_PER_TRADE", "0.10"))

#: Additional slippage applied in backtests as a function of intrabar range.
#: Effective slippage ~= SLIPPAGE_PCT + BACKTEST_DYNAMIC_SLIPPAGE_K * ((h-l)/c).
BACKTEST_DYNAMIC_SLIPPAGE_K: float = float(os.getenv("BACKTEST_DYNAMIC_SLIPPAGE_K", "0.0"))

#: Simulated order latency in bars for backtests.
BACKTEST_LATENCY_BARS: int = int(os.getenv("BACKTEST_LATENCY_BARS", "0"))

#: Minimum fraction of requested quantity filled in backtests.
#: 1.0 means full fills (legacy behaviour); lower values simulate partial fills.
BACKTEST_PARTIAL_FILL_MIN: float = float(os.getenv("BACKTEST_PARTIAL_FILL_MIN", "1.0"))
MIN_EVAL_TRADES: int = int(os.getenv("MIN_EVAL_TRADES", "100"))

# ── Timing ────────────────────────────────────────────────────────────────────
#: Seconds to sleep between each ``run_once`` cycle.
CHECK_INTERVAL_SECONDS: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))

#: Hard cap on exponential-backoff sleep after repeated errors.
MAX_BACKOFF_SECONDS: int = 600

#: Number of consecutive cycle errors allowed before cooling down.
MAX_CONSECUTIVE_ERRORS: int = int(os.getenv("MAX_CONSECUTIVE_ERRORS", "5"))

#: Cooldown applied after the consecutive-error circuit breaker trips.
ERROR_COOLDOWN_SECONDS: int = int(os.getenv("ERROR_COOLDOWN_SECONDS", "900"))

#: Warn when open working orders persist for this many consecutive cycles.
OPEN_ORDER_STALE_CYCLES: int = int(os.getenv("OPEN_ORDER_STALE_CYCLES", "5"))

# ── Filters ───────────────────────────────────────────────────────────────────
#: Enable session-hour-based filter (suppress signals outside trading hours).
SESSION_FILTER_ENABLED: bool = os.getenv("SESSION_FILTER_ENABLED", "false").lower() in {"1", "true", "yes"}

#: Start of trading session in UTC hours (0-23).
SESSION_START_HOUR_UTC: int = int(os.getenv("SESSION_START_HOUR_UTC", "0"))

#: End of trading session in UTC hours (0-23).
SESSION_END_HOUR_UTC: int = int(os.getenv("SESSION_END_HOUR_UTC", "23"))

#: Enable volume-based filter (suppress signals on low volume bars).
VOLUME_FILTER_ENABLED: bool = os.getenv("VOLUME_FILTER_ENABLED", "false").lower() in {"1", "true", "yes"}

#: Enable price structure filter (suppress counter-trend entries).
STRUCTURE_FILTER_ENABLED: bool = os.getenv("STRUCTURE_FILTER_ENABLED", "false").lower() in {"1", "true", "yes"}

#: Lookback bars for structure analysis (higher lows/lower highs detection).
STRUCTURE_LOOKBACK: int = int(os.getenv("STRUCTURE_LOOKBACK", "20"))

# ── Backtest ──────────────────────────────────────────────────────────────────
#: Path to the CSV file used by BacktestBroker.
CSV_PATH: str = os.getenv("BACKTEST_CSV", "historical_data.csv")

#: Starting cash for the simulated backtest account.
STARTING_CASH: float = float(os.getenv("BACKTEST_STARTING_CASH", "100000"))

# ── Portfolio / execution safety ──────────────────────────────────────────────
#: Hard cap on gross position notional as a fraction of equity.
MAX_GROSS_EXPOSURE_PCT: float = float(os.getenv("MAX_GROSS_EXPOSURE_PCT", "1.0"))

#: Consecutive order submission errors before cooldown.
MAX_ORDER_ERRORS: int = int(os.getenv("MAX_ORDER_ERRORS", "3"))
ORDER_ERROR_COOLDOWN_SECONDS: int = int(os.getenv("ORDER_ERROR_COOLDOWN_SECONDS", "300"))

# ── State persistence ─────────────────────────────────────────────────────────
#: Path where TradingState is persisted between cycles so the bot can recover
#: trailing stop watermarks after a crash or restart.
STATE_FILE: str = os.getenv("STATE_FILE", "bot_state.json")


# ── Validation ────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def validate_config() -> None:
    """
    Validate configuration parameter constraints.

    Raises ``ValueError`` if any parameter violates its expected constraint.
    Called automatically on module load so misconfiguration is caught early.
    """
    if FAST_WINDOW >= SLOW_WINDOW:
        raise ValueError(
            f"FAST_WINDOW ({FAST_WINDOW}) must be < SLOW_WINDOW ({SLOW_WINDOW})"
        )
    if SLOW_WINDOW >= TREND_WINDOW:
        raise ValueError(
            f"SLOW_WINDOW ({SLOW_WINDOW}) must be < TREND_WINDOW ({TREND_WINDOW})"
        )
    if not (0 < RSI_OVERSOLD < RSI_OVERBOUGHT < 100):
        raise ValueError(
            f"RSI thresholds must satisfy 0 < RSI_OVERSOLD ({RSI_OVERSOLD}) "
            f"< RSI_OVERBOUGHT ({RSI_OVERBOUGHT}) < 100"
        )
    if not (0 <= MIN_ATR_PCT < 1):
        raise ValueError(
            f"MIN_ATR_PCT ({MIN_ATR_PCT}) must be in [0, 1)"
        )
    if ADAPTIVE_LOOKBACK_DAYS <= 0:
        raise ValueError(f"ADAPTIVE_LOOKBACK_DAYS ({ADAPTIVE_LOOKBACK_DAYS}) must be > 0")
    if not (0 < ADAPTIVE_ADX_PERCENTILE < 100):
        raise ValueError(
            f"ADAPTIVE_ADX_PERCENTILE ({ADAPTIVE_ADX_PERCENTILE}) must be in (0, 100)"
        )
    if not (0 < ADAPTIVE_ATR_PERCENTILE < 100):
        raise ValueError(
            f"ADAPTIVE_ATR_PERCENTILE ({ADAPTIVE_ATR_PERCENTILE}) must be in (0, 100)"
        )
    if not (0 <= SESSION_START_HOUR_UTC <= 23 and 1 <= SESSION_END_HOUR_UTC <= 24):
        raise ValueError("SESSION_START_HOUR_UTC/SESSION_END_HOUR_UTC must be valid UTC hours")
    if SESSION_START_HOUR_UTC >= SESSION_END_HOUR_UTC:
        raise ValueError("SESSION_START_HOUR_UTC must be < SESSION_END_HOUR_UTC")
    if ATR_ACCEL_WINDOW <= 1:
        raise ValueError(f"ATR_ACCEL_WINDOW ({ATR_ACCEL_WINDOW}) must be > 1")
    if MIN_ATR_ACCEL < 0:
        raise ValueError(f"MIN_ATR_ACCEL ({MIN_ATR_ACCEL}) must be >= 0")
    if STRUCTURE_LOOKBACK < 2:
        raise ValueError(f"STRUCTURE_LOOKBACK ({STRUCTURE_LOOKBACK}) must be >= 2")
    if MIN_VOLUME < 0:
        raise ValueError(f"MIN_VOLUME ({MIN_VOLUME}) must be >= 0")
    if not (0 < RISK_PER_TRADE <= 0.10):
        raise ValueError(
            f"RISK_PER_TRADE ({RISK_PER_TRADE}) must be in (0, 0.10] "
            f"(hard cap at 10%); got {RISK_PER_TRADE * 100:.1f}%"
        )
    if MAX_DAILY_LOSS_PCT <= 0 or MAX_DAILY_LOSS_PCT > 1:
        raise ValueError(
            f"MAX_DAILY_LOSS_PCT ({MAX_DAILY_LOSS_PCT}) must be in (0, 1]"
        )
    if STOP_LOSS_PCT < 0:
        raise ValueError(f"STOP_LOSS_PCT ({STOP_LOSS_PCT}) must be >= 0")
    if TAKE_PROFIT_PCT < 0:
        raise ValueError(f"TAKE_PROFIT_PCT ({TAKE_PROFIT_PCT}) must be >= 0")
    if TAKE_PROFIT_MULT <= 0:
        raise ValueError(f"TAKE_PROFIT_MULT ({TAKE_PROFIT_MULT}) must be > 0")
    if MIN_EXPECTED_RR <= 0:
        raise ValueError(f"MIN_EXPECTED_RR ({MIN_EXPECTED_RR}) must be > 0")
    if ESTIMATED_ROUND_TRIP_COST_PCT < 0:
        raise ValueError(
            f"ESTIMATED_ROUND_TRIP_COST_PCT ({ESTIMATED_ROUND_TRIP_COST_PCT}) must be >= 0"
        )
    if BREAK_EVEN_R_MULT <= 0:
        raise ValueError(f"BREAK_EVEN_R_MULT ({BREAK_EVEN_R_MULT}) must be > 0")
    if PARTIAL_TP1_R <= 0 or PARTIAL_TP2_R <= 0:
        raise ValueError("PARTIAL_TP1_R and PARTIAL_TP2_R must be > 0")
    if not (0 < PARTIAL_TP1_FRACTION <= 1):
        raise ValueError(
            f"PARTIAL_TP1_FRACTION ({PARTIAL_TP1_FRACTION}) must be in (0, 1]"
        )
    if not (0 < PARTIAL_TP2_FRACTION <= 1):
        raise ValueError(
            f"PARTIAL_TP2_FRACTION ({PARTIAL_TP2_FRACTION}) must be in (0, 1]"
        )
    if ATR_STOP_MULT <= 0:
        raise ValueError(f"ATR_STOP_MULT ({ATR_STOP_MULT}) must be > 0")
    if STARTING_CASH <= 0:
        raise ValueError(f"BACKTEST_STARTING_CASH ({STARTING_CASH}) must be > 0")
    if BACKTEST_DYNAMIC_SLIPPAGE_K < 0:
        raise ValueError(
            f"BACKTEST_DYNAMIC_SLIPPAGE_K ({BACKTEST_DYNAMIC_SLIPPAGE_K}) must be >= 0"
        )
    if BACKTEST_LATENCY_BARS < 0:
        raise ValueError(
            f"BACKTEST_LATENCY_BARS ({BACKTEST_LATENCY_BARS}) must be >= 0"
        )
    if not (0 < BACKTEST_PARTIAL_FILL_MIN <= 1):
        raise ValueError(
            f"BACKTEST_PARTIAL_FILL_MIN ({BACKTEST_PARTIAL_FILL_MIN}) must be in (0, 1]"
        )
    if MIN_EVAL_TRADES < 1:
        raise ValueError(f"MIN_EVAL_TRADES ({MIN_EVAL_TRADES}) must be >= 1")
    if not (0 < MAX_GROSS_EXPOSURE_PCT <= 1):
        raise ValueError(
            f"MAX_GROSS_EXPOSURE_PCT ({MAX_GROSS_EXPOSURE_PCT}) must be in (0, 1]"
        )
    if MAX_ORDER_ERRORS <= 0:
        raise ValueError(
            f"MAX_ORDER_ERRORS ({MAX_ORDER_ERRORS}) must be > 0"
        )
    if ORDER_ERROR_COOLDOWN_SECONDS <= 0:
        raise ValueError(
            f"ORDER_ERROR_COOLDOWN_SECONDS ({ORDER_ERROR_COOLDOWN_SECONDS}) must be > 0"
        )
    if MAX_CONSECUTIVE_ERRORS <= 0:
        raise ValueError(
            f"MAX_CONSECUTIVE_ERRORS ({MAX_CONSECUTIVE_ERRORS}) must be > 0"
        )
    if ERROR_COOLDOWN_SECONDS <= 0:
        raise ValueError(
            f"ERROR_COOLDOWN_SECONDS ({ERROR_COOLDOWN_SECONDS}) must be > 0"
        )
    if OPEN_ORDER_STALE_CYCLES <= 0:
        raise ValueError(
            f"OPEN_ORDER_STALE_CYCLES ({OPEN_ORDER_STALE_CYCLES}) must be > 0"
        )
    logger.debug("Configuration validated successfully.")


validate_config()
