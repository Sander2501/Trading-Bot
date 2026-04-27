"""
main
~~~~
Live trading entry-point.

Run with::

    python main.py           # live/demo trading
    python main.py --dry-run # log orders without sending them

The bot connects to the Capital.com API, fetches recent price bars,
generates an EMA-crossover + RSI signal, and submits market orders according
to the configured risk parameters.  It runs in an infinite loop, sleeping
``CHECK_INTERVAL_SECONDS`` between cycles, with exponential back-off on errors.
"""

import argparse
import json
import logging
import time
from datetime import date, datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from brokers import CapitalBroker, BaseBroker
from config import (
    ADX_THRESHOLD,
    ADX_WINDOW,
    ALLOW_SHORTS,
    ATR_STOP_MULT,
    ATR_STOP_WINDOW,
    CHECK_INTERVAL_SECONDS,
    CONFIRM_BARS,
    ERROR_COOLDOWN_SECONDS,
    FAST_WINDOW,
    MACD_FAST,
    MACD_SIGNAL_WINDOW,
    MACD_SLOW,
    MAX_BACKOFF_SECONDS,
    MIN_ATR_PCT,
    MAX_CONSECUTIVE_ERRORS,
    MAX_GROSS_EXPOSURE_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_GROSS_EXPOSURE_PCT,
    MAX_ORDER_ERRORS,
    OPEN_ORDER_STALE_CYCLES,
    ORDER_ERROR_COOLDOWN_SECONDS,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_WINDOW,
    RISK_PER_TRADE,
    SLIPPAGE_PCT,
    SLOW_WINDOW,
    STATE_FILE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_MULT,
    TAKE_PROFIT_PCT,
    SYMBOL,
    TIMEFRAME,
    TREND_WINDOW,
)
from strategy import atr_stop_distance, moving_average_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        RotatingFileHandler("bot.log", maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Metrics logging ──────────────────────────────────────────────────────────
#: File path where JSON-line metrics records are appended.
METRICS_LOG_PATH: str = "bot_metrics.jsonl"

#: Print a heartbeat summary to the console every N cycles.
METRICS_HEARTBEAT_CYCLES: int = 10


# ------------------------------------------------------------------
# State container
# ------------------------------------------------------------------


class TradingState:
    """
    Encapsulates mutable bot state that must persist across ``run_once`` cycles.

    Attributes
    ----------
    position_high:
        Highest price seen since the current long position was opened.
        Used to trail the ATR stop upward as the trade moves in our favour.
    position_low:
        Lowest price seen since the current short position was opened.
        Used to trail the ATR stop downward for short trades.
    last_snapshot_date:
        Calendar date on which ``snapshot_day()`` was last called so the
        daily-loss baseline is refreshed exactly once per day.
    """

    def __init__(self) -> None:
        self.position_high: float = 0.0
        self.position_low: float = float("inf")
        self.last_snapshot_date: date | None = None
        self.open_order_streak: int = 0
        self.cycles: int = 0
        self.order_error_streak: int = 0

    def reset_watermarks(self, price: float) -> None:
        """Seed both watermarks to *price* when a new position is opened."""
        self.position_high = price
        self.position_low = price

    def save(self, path: str = STATE_FILE) -> None:
        """Persist state to *path* so the bot can recover after a crash."""
        data = {
            "position_high": self.position_high,
            "position_low": self.position_low if self.position_low != float("inf") else None,
            "last_snapshot_date": self.last_snapshot_date.isoformat() if self.last_snapshot_date else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            Path(path).write_text(json.dumps(data))
        except OSError as exc:
            logger.warning("Could not save state: %s", exc)

    @classmethod
    def load(cls, path: str = STATE_FILE) -> "TradingState":
        """Load state from *path*, returning a fresh state if the file is missing."""
        state = cls()
        try:
            data = json.loads(Path(path).read_text())
            state.position_high = float(data.get("position_high") or 0.0)
            low = data.get("position_low")
            state.position_low = float(low) if low is not None else float("inf")
            d = data.get("last_snapshot_date")
            state.last_snapshot_date = date.fromisoformat(d) if d else None
            logger.info("Restored trading state from %s.", path)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        return state


# ------------------------------------------------------------------
# Dry-run broker wrapper
# ------------------------------------------------------------------


class DryRunBroker(BaseBroker):
    """
    Wraps any live broker and replaces order submission with log-only calls.
    All read operations (prices, positions, equity) pass through to the
    underlying broker so the signal logic runs on real data.
    """

    def __init__(self, inner: BaseBroker) -> None:
        self._inner = inner

    def get_recent_bars(self, symbol, limit, timeframe="1Min"):
        return self._inner.get_recent_bars(symbol, limit, timeframe)

    def get_position_qty(self, symbol):
        return self._inner.get_position_qty(symbol)

    def get_entry_price(self, symbol):
        return self._inner.get_entry_price(symbol)

    def has_open_order(self, symbol):
        return self._inner.has_open_order(symbol)

    def submit_buy(self, symbol, qty, sl=None, tp=None):
        logger.info("[DRY-RUN] Would BUY  %.6f %s (SL=%s, TP=%s)", qty, symbol, sl, tp)

    def submit_sell(self, symbol, qty, sl=None, tp=None):
        logger.info("[DRY-RUN] Would SELL %.6f %s (SL=%s, TP=%s)", qty, symbol, sl, tp)

    def get_market_status(self):
        return self._inner.get_market_status()

    def get_buying_power(self):
        return self._inner.get_buying_power()

    def get_equity(self):
        return self._inner.get_equity()

    def snapshot_day(self):
        return self._inner.snapshot_day()

    def flush_position_cache(self):
        return self._inner.flush_position_cache()

    @property
    def supports_shorting(self):
        return self._inner.supports_shorting


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------




def position_size(
    equity: float,
    buying_power: float,
    stop_distance: float,
    price: float,
) -> float:
    """
    Calculate order quantity using true risk-based sizing, capped by affordability.

    ``RISK_PER_TRADE`` is the fraction of equity we are willing to *lose* if
    stopped out, so: qty = (equity × RISK_PER_TRADE) / stop_distance.

    This keeps dollar risk constant regardless of price level or volatility —
    wider stops produce smaller positions, tighter stops produce larger ones.
    The result is further capped so the total position cost never exceeds equity.
    """
    if stop_distance <= 0 or price <= 0:
        return 0.0
    qty_by_risk = (equity * RISK_PER_TRADE) / stop_distance
    effective_buying_power = max(0.0, min(equity, buying_power))
    qty_by_exposure = equity * MAX_GROSS_EXPOSURE_PCT / (price * (1 + SLIPPAGE_PCT))
    # Cap at 99.9% of equity divided by worst-case fill price to ensure the
    # order cost stays within available cash after slippage and float rounding.
    qty_by_funds = effective_buying_power * 0.999 / (price * (1 + SLIPPAGE_PCT))
    return round(max(0.0, min(qty_by_risk, qty_by_funds, qty_by_exposure)), 6)


def _submit_with_guard(
    submit_fn,
    state: TradingState,
    description: str,
    sleep_enabled: bool,
) -> bool:
    """Submit an order and apply an error circuit-breaker on repeated failures."""
    try:
        submit_fn()
        state.order_error_streak = 0
        return True
    except Exception as exc:
        state.order_error_streak += 1
        logger.exception("Order submission failed (%s): %s", description, exc)
        if state.order_error_streak >= MAX_ORDER_ERRORS:
            logger.error(
                "Order error circuit-breaker tripped (%d consecutive errors). Cooling down %ds.",
                state.order_error_streak,
                ORDER_ERROR_COOLDOWN_SECONDS,
            )
            if sleep_enabled:
                time.sleep(ORDER_ERROR_COOLDOWN_SECONDS)
        return False


def daily_loss_exceeded(broker: BaseBroker) -> bool:
    """Return True when today's drawdown has reached ``MAX_DAILY_LOSS_PCT``."""
    equity, last_equity = broker.get_equity()
    if last_equity <= 0:
        return False
    return (last_equity - equity) / last_equity >= MAX_DAILY_LOSS_PCT


def _maybe_snapshot_day(broker: BaseBroker, state: TradingState) -> None:
    """Refresh the daily-loss baseline once per calendar day.

    Uses ``broker.current_date()`` rather than ``date.today()`` so that in
    backtests the "day" rolls over based on the bar timestamp, not wall clock.
    """
    today = broker.current_date()
    if state.last_snapshot_date != today:
        broker.snapshot_day()
        state.last_snapshot_date = today
        logger.info("Daily equity snapshot taken (baseline date: %s).", today)


def _emit_cycle_metrics(
    state: TradingState,
    latest_price: float,
    signal: str,
    current_qty: float,
    action: str,
    open_order_exists: bool,
    equity: float,
) -> None:
    """Append a structured per-cycle JSONL record for observability."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle": state.cycles,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "price": round(latest_price, 8),
        "signal": signal,
        "action": action,
        "position_qty": round(current_qty, 8),
        "equity": round(equity, 8),
        "open_order": bool(open_order_exists),
        "open_order_streak": state.open_order_streak,
    }
    try:
        with open(METRICS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError as exc:
        logger.warning("Failed to write metrics record to %s: %s", METRICS_LOG_PATH, exc)

    if state.cycles % METRICS_HEARTBEAT_CYCLES == 0:
        logger.info(
            "HEARTBEAT cycle=%d signal=%s action=%s qty=%.6f equity=%.2f open_order=%s",
            state.cycles,
            signal,
            action,
            current_qty,
            equity,
            open_order_exists,
        )


# ------------------------------------------------------------------
# Core trading cycle
# ------------------------------------------------------------------


def run_once(
    broker: BaseBroker,
    state: TradingState | None = None,
    sleep_enabled: bool = True,
) -> None:
    """
    Execute a single trading cycle:

    1. Refresh the daily-loss baseline if the calendar day has rolled over.
    2. Flush any stale per-cycle position cache.
    3. Skip if market is closed.
    4. Halt for the day if the daily-loss limit has been reached.
    5. Generate an EMA-crossover + RSI signal.
    6. Apply stop-loss, then act on BUY / SELL signals.

    Parameters
    ----------
    broker:
        Broker instance (live or backtest).
    state:
        Mutable trading state for watermarks and daily snapshot tracking.
        A fresh ``TradingState`` is created automatically if *None* is passed,
        which preserves backward compatibility with callers that omit the argument.
    sleep_enabled:
        When ``False`` the function returns immediately instead of sleeping
        (useful for backtests and unit tests).
    """
    if state is None:
        state = TradingState()

    _maybe_snapshot_day(broker, state)
    state.cycles += 1
    broker.flush_position_cache()

    is_open, wait = broker.get_market_status()
    if not is_open:
        logger.info(f"Market is closed. Sleeping {wait:.0f}s until open.")
        if sleep_enabled:
            time.sleep(wait if wait > 0 else CHECK_INTERVAL_SECONDS)
        return

    if daily_loss_exceeded(broker):
        logger.warning(
            f"Daily loss limit ({MAX_DAILY_LOSS_PCT:.1%}) reached. "
            "Halting for the day."
        )
        if sleep_enabled:
            time.sleep(3600)
        return

    # Fetch enough bars for all indicators to warm up
    bars_needed = max(TREND_WINDOW, MACD_SLOW, RSI_WINDOW) * 2 + CONFIRM_BARS + 5
    bars = broker.get_recent_bars(SYMBOL, limit=bars_needed, timeframe=TIMEFRAME)

    latest_price = float(bars["c"].iloc[-1])
    latest_high = float(bars["h"].iloc[-1])
    latest_low = float(bars["l"].iloc[-1])

    # Compute ATR once — used for both the trailing stop check and position sizing.
    atr = atr_stop_distance(bars["h"], bars["l"], bars["c"], window=ATR_STOP_WINDOW, multiplier=1.0)
    stop_dist = max(atr * ATR_STOP_MULT, latest_price * STOP_LOSS_PCT)
    tp_dist   = max(atr * TAKE_PROFIT_MULT, latest_price * TAKE_PROFIT_PCT)

    signal = moving_average_signal(
        bars,
        fast_window=FAST_WINDOW,
        slow_window=SLOW_WINDOW,
        trend_window=TREND_WINDOW,
        confirm_bars=CONFIRM_BARS,
        rsi_window=RSI_WINDOW,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
        macd_fast=MACD_FAST,
        macd_slow=MACD_SLOW,
        macd_signal=MACD_SIGNAL_WINDOW,
        adx_window=ADX_WINDOW,
        adx_threshold=ADX_THRESHOLD,
        min_atr_pct=MIN_ATR_PCT,
    )
    current_qty = broker.get_position_qty(SYMBOL)
    open_order_exists = broker.has_open_order(SYMBOL)
    buying_power = broker.get_buying_power()
    equity, _ = broker.get_equity()

    logger.info(
        f"{SYMBOL} | price={latest_price:.2f} | signal={signal} "
        f"| held={current_qty:.6f} | open_order={open_order_exists}"
    )

    if open_order_exists:
        state.open_order_streak += 1
        if state.open_order_streak >= OPEN_ORDER_STALE_CYCLES:
            logger.warning(
                "Open order has persisted for %d consecutive cycles; "
                "verify broker order state / cancel stale order if needed.",
                state.open_order_streak,
            )
        logger.info("Open order already exists. Skipping cycle.")
        _emit_cycle_metrics(
            state, latest_price, signal, current_qty, "SKIP_OPEN_ORDER", open_order_exists, equity
        )
        if sleep_enabled:
            time.sleep(CHECK_INTERVAL_SECONDS)
        return
    state.open_order_streak = 0

    # --- Watermark Tracking ---
    if current_qty > 0:
        state.position_high = max(state.position_high, latest_price)
    elif current_qty < 0:
        state.position_low = min(state.position_low, latest_price)
    else:
        state.reset_watermarks(latest_price)

    # --- Position Management (Trailing Stop + Take Profit) ---
    if current_qty != 0:
        entry_price = broker.get_entry_price(SYMBOL)
        if entry_price:
            # Use intrabar extremes (h/l) for trigger checks so exits are
            # consistent with how backtests evaluate standing stops/targets.
            long_stop_level = state.position_high - stop_dist
            short_stop_level = state.position_low + stop_dist
            long_tp_level = entry_price + tp_dist
            short_tp_level = entry_price - tp_dist

            long_stopped = current_qty > 0 and latest_low <= long_stop_level
            short_stopped = current_qty < 0 and latest_high >= short_stop_level
            long_tp = current_qty > 0 and latest_high >= long_tp_level
            short_tp = current_qty < 0 and latest_low <= short_tp_level

            exit_reason = None
            if long_stopped or short_stopped:
                exit_reason = "TRAILING STOP"
            elif long_tp or short_tp:
                exit_reason = "TAKE PROFIT"

            if exit_reason:
                if current_qty > 0:
                    ok = _submit_with_guard(
                        lambda: broker.submit_sell(SYMBOL, current_qty),
                        state,
                        "exit long",
                        sleep_enabled,
                    )
                    ref = state.position_high
                else:
                    ok = _submit_with_guard(
                        lambda: broker.submit_buy(SYMBOL, abs(current_qty)),
                        state,
                        "exit short",
                        sleep_enabled,
                    )
                    ref = state.position_low
                if not ok:
                    _emit_cycle_metrics(
                        state, latest_price, signal, current_qty, "ORDER_ERROR_EXIT",
                        open_order_exists, equity
                    )
                    return

                state.reset_watermarks(latest_price)
                logger.warning(
                    f"{exit_reason}: closed {abs(current_qty):.6f} {SYMBOL} "
                    f"at {latest_price:.2f} (entry {entry_price:.2f}, ref {ref:.2f}, "
                    f"stop_dist={stop_dist:.2f}, tp_dist={tp_dist:.2f})"
                )
                _emit_cycle_metrics(
                    state, latest_price, signal, current_qty, f"EXIT_{exit_reason.replace(' ', '_')}",
                    open_order_exists, equity
                )
                if sleep_enabled:
                    time.sleep(CHECK_INTERVAL_SECONDS)
                return

    # --- Signal execution ---
    qty = position_size(equity, buying_power, stop_dist, latest_price)
    action = "HOLD"

    if signal == "BUY":
        if current_qty > 0:
            logger.info("Already long — holding.")
        elif current_qty < 0:
            # Cover short first; next cycle opens the long if signal persists
            if not _submit_with_guard(
                lambda: broker.submit_buy(SYMBOL, abs(current_qty)),
                state,
                "cover short",
                sleep_enabled,
            ):
                _emit_cycle_metrics(
                    state, latest_price, signal, current_qty, "ORDER_ERROR_COVER", open_order_exists, equity
                )
                return
            state.reset_watermarks(latest_price)
            logger.info(f"COVER {abs(current_qty):.6f} {SYMBOL} @ ~{latest_price:.2f}")
            action = "COVER_SHORT"
        else:
            if qty <= 0:
                logger.info("Insufficient buying power to open long.")
                action = "SKIP_NO_BUYING_POWER"
            else:
                sl_price = round(latest_price - stop_dist, 8)
                tp_price = round(latest_price + tp_dist, 8)
                if not _submit_with_guard(
                    lambda: broker.submit_buy(SYMBOL, qty, sl=sl_price, tp=tp_price),
                    state,
                    "open long",
                    sleep_enabled,
                ):
                    _emit_cycle_metrics(
                        state, latest_price, signal, current_qty, "ORDER_ERROR_OPEN_LONG", open_order_exists, equity
                    )
                    return
                state.position_high = latest_price
                logger.info(
                    f"BUY   {qty:.6f} {SYMBOL} @ ~{latest_price:.2f} "
                    f"(SL={sl_price:.2f}, TP={tp_price:.2f})"
                )
                action = "OPEN_LONG"

    elif signal == "SELL":
        if current_qty < 0:
            logger.info("Already short — holding.")
        elif current_qty > 0:
            # Close long first; next cycle opens the short if signal persists
            if not _submit_with_guard(
                lambda: broker.submit_sell(SYMBOL, current_qty),
                state,
                "close long",
                sleep_enabled,
            ):
                _emit_cycle_metrics(
                    state, latest_price, signal, current_qty, "ORDER_ERROR_CLOSE_LONG", open_order_exists, equity
                )
                return
            state.reset_watermarks(latest_price)
            logger.info(f"SELL  {current_qty:.6f} {SYMBOL} @ ~{latest_price:.2f}")
            action = "CLOSE_LONG"
        else:
            if not ALLOW_SHORTS:
                logger.info("SELL signal — ALLOW_SHORTS=False, staying flat.")
                action = "SKIP_SHORTS_DISABLED"
            elif not broker.supports_shorting:
                logger.info("SELL signal — broker does not support shorting, staying flat.")
                action = "SKIP_NO_SHORTING"
            elif qty <= 0:
                logger.info("Insufficient equity to open short.")
                action = "SKIP_NO_EQUITY"
            else:
                sl_price = round(latest_price + stop_dist, 8)
                tp_price = round(latest_price - tp_dist, 8)
                if not _submit_with_guard(
                    lambda: broker.submit_sell(SYMBOL, qty, sl=sl_price, tp=tp_price),
                    state,
                    "open short",
                    sleep_enabled,
                ):
                    _emit_cycle_metrics(
                        state, latest_price, signal, current_qty, "ORDER_ERROR_OPEN_SHORT", open_order_exists, equity
                    )
                    return
                state.position_low = latest_price
                logger.info(
                    f"SHORT {qty:.6f} {SYMBOL} @ ~{latest_price:.2f} "
                    f"(SL={sl_price:.2f}, TP={tp_price:.2f})"
                )
                action = "OPEN_SHORT"

    else:
        logger.info("No action.")
        action = "NO_ACTION"

    _emit_cycle_metrics(
        state, latest_price, signal, current_qty, action, open_order_exists, equity
    )

    if sleep_enabled:
        time.sleep(CHECK_INTERVAL_SECONDS)


def _update_error_state(
    consecutive_errors: int,
    error_backoff: int,
) -> tuple[int, int, bool]:
    """
    Update error counters/backoff and report whether the circuit breaker tripped.
    """
    next_errors = consecutive_errors + 1
    tripped = next_errors >= MAX_CONSECUTIVE_ERRORS
    next_backoff = min(error_backoff * 2, MAX_BACKOFF_SECONDS)
    return next_errors, next_backoff, tripped


# ------------------------------------------------------------------
# Bot runner
# ------------------------------------------------------------------


def run_bot(broker: BaseBroker | None = None, dry_run: bool = False) -> None:
    """
    Start the live trading loop.

    Accepts an optional ``broker`` argument to allow dependency injection
    (useful for testing with a BacktestBroker).  Pass ``dry_run=True`` to log
    orders without submitting them.
    """
    if broker is None:
        broker = CapitalBroker()

    if dry_run:
        broker = DryRunBroker(broker)
        logger.info("DRY-RUN mode — orders will be logged but not sent.")

    logger.info(f"Crypto bot started using {broker.__class__.__name__}.")

    state = TradingState.load()
    error_backoff = CHECK_INTERVAL_SECONDS
    consecutive_errors = 0

    while True:
        try:
            run_once(broker, state, sleep_enabled=True)
            state.save()
        except Exception as exc:
            logger.exception(f"Bot error: {exc}")
            consecutive_errors, error_backoff, tripped = _update_error_state(
                consecutive_errors, error_backoff
            )
            time.sleep(error_backoff)
            if tripped:
                logger.critical(
                    "Circuit breaker tripped after %d consecutive errors; "
                    "cooling down for %ds before resuming.",
                    consecutive_errors,
                    ERROR_COOLDOWN_SECONDS,
                )
                time.sleep(ERROR_COOLDOWN_SECONDS)
                consecutive_errors = 0
                error_backoff = CHECK_INTERVAL_SECONDS
            continue

        consecutive_errors = 0
        error_backoff = CHECK_INTERVAL_SECONDS  # reset on successful cycle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capital.com trading bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log orders without submitting them (safe for testing against live data)",
    )
    args = parser.parse_args()
    run_bot(dry_run=args.dry_run)
