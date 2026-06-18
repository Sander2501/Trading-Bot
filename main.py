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
from pathlib import Path

from brokers import CapitalBroker, BaseBroker
from config import (
    ADX_THRESHOLD,
    ADX_TRENDING_THRESHOLD,
    ADX_WINDOW,
    ALLOW_SHORTS,
    ATR_STOP_MULT,
    ATR_STOP_WINDOW,
    ADAPTIVE_ADX_PERCENTILE,
    ADAPTIVE_ATR_PERCENTILE,
    ADAPTIVE_LOOKBACK_DAYS,
    ADAPTIVE_THRESHOLDS_ENABLED,
    ATR_ACCEL_WINDOW,
    BACKTEST_DYNAMIC_SLIPPAGE_K,
    BACKTEST_LATENCY_BARS,
    BACKTEST_PARTIAL_FILL_MIN,
    BREAK_EVEN_R_MULT,
    CHECK_INTERVAL_SECONDS,
    COMMISSION_PER_TRADE,
    CONFIRM_BARS,
    DATA_INTEGRITY_GATE_ENABLED,
    DATA_MAX_BAR_AGE_SECONDS,
    ERROR_COOLDOWN_SECONDS,
    KILL_SWITCH_FILE,
    MAX_SLIP_BPS,
    MAX_WEEKLY_LOSS_PCT,
    SLIPPAGE_HALT_SECONDS,
    TELEMETRY_ENABLED,
    TELEMETRY_FILE,
    FAST_WINDOW,
    MACD_FAST,
    MACD_SIGNAL_WINDOW,
    MACD_SLOW,
    MAX_BACKOFF_SECONDS,
    MIN_ATR_PCT,
    MAX_CONSECUTIVE_ERRORS,
    MAX_DAILY_LOSS_PCT,
    MAX_GROSS_EXPOSURE_PCT,
    MAX_ORDER_ERRORS,
    MIN_ATR_ACCEL,
    MIN_EXPECTED_RR,
    ESTIMATED_ROUND_TRIP_COST_PCT,
    MIN_VOLUME,
    OPEN_ORDER_STALE_CYCLES,
    ORDER_ERROR_COOLDOWN_SECONDS,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_WINDOW,
    RISK_PER_TRADE,
    SESSION_END_HOUR_UTC,
    SESSION_FILTER_ENABLED,
    SESSION_START_HOUR_UTC,
    SLIPPAGE_PCT,
    SLOW_WINDOW,
    STATE_FILE,
    STOP_LOSS_PCT,
    STRUCTURE_FILTER_ENABLED,
    STRUCTURE_LOOKBACK,
    TAKE_PROFIT_MULT,
    PARTIAL_TP1_FRACTION,
    PARTIAL_TP1_R,
    PARTIAL_TP2_FRACTION,
    PARTIAL_TP2_R,
    TAKE_PROFIT_PCT,
    SYMBOL,
    TIMEFRAME,
    TREND_WINDOW,
    VOLUME_FILTER_ENABLED,
)
import risk
import telemetry
from data_integrity import check_bars, interval_seconds_for
from strategy import atr_stop_distance, moving_average_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

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
        self.entry_price: float = 0.0
        self.initial_stop_dist: float = 0.0
        self.break_even_armed: bool = False
        self.partial_tp1_taken: bool = False
        self.partial_tp2_taken: bool = False
        self.events: list[dict] = []
        self.last_snapshot_date: date | None = None
        self.open_order_streak: int = 0
        self.cycles: int = 0
        self.order_error_streak: int = 0
        # Risk firewall
        self.week_anchor_date: date | None = None
        self.week_baseline_equity: float = 0.0
        self.slippage_halt_until: datetime | None = None

    def reset_watermarks(self, price: float) -> None:
        """Seed both watermarks to *price* when a new position is opened."""
        self.position_high = price
        self.position_low = price

    def seed_trade_context(self, entry_price: float, stop_distance: float) -> None:
        """Record entry metadata for break-even and partial-TP logic."""
        self.entry_price = entry_price
        self.initial_stop_dist = max(0.0, stop_distance)
        self.break_even_armed = False
        self.partial_tp1_taken = False
        self.partial_tp2_taken = False

    def log_event(self, cycle: int, event: str, details: dict | None = None) -> None:
        payload = {"cycle": cycle, "event": event, **(details or {})}
        self.events.append(payload)
        if len(self.events) > 1000:
            self.events = self.events[-1000:]


    def save(self, path: str = STATE_FILE) -> None:
        """Persist state to *path* so the bot can recover after a crash."""
        data = {
            "position_high": self.position_high,
            "position_low": self.position_low if self.position_low != float("inf") else None,
            "entry_price": self.entry_price,
            "initial_stop_dist": self.initial_stop_dist,
            "break_even_armed": self.break_even_armed,
            "partial_tp1_taken": self.partial_tp1_taken,
            "partial_tp2_taken": self.partial_tp2_taken,
            "events": self.events[-200:],
            "last_snapshot_date": self.last_snapshot_date.isoformat() if self.last_snapshot_date else None,
            "week_anchor_date": self.week_anchor_date.isoformat() if self.week_anchor_date else None,
            "week_baseline_equity": self.week_baseline_equity,
            "slippage_halt_until": self.slippage_halt_until.isoformat() if self.slippage_halt_until else None,
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
            state.entry_price = float(data.get("entry_price") or 0.0)
            state.initial_stop_dist = float(data.get("initial_stop_dist") or 0.0)
            state.break_even_armed = bool(data.get("break_even_armed", False))
            state.partial_tp1_taken = bool(data.get("partial_tp1_taken", False))
            state.partial_tp2_taken = bool(data.get("partial_tp2_taken", False))
            state.events = list(data.get("events") or [])
            d = data.get("last_snapshot_date")
            state.last_snapshot_date = date.fromisoformat(d) if d else None
            wd = data.get("week_anchor_date")
            state.week_anchor_date = date.fromisoformat(wd) if wd else None
            state.week_baseline_equity = float(data.get("week_baseline_equity") or 0.0)
            sh = data.get("slippage_halt_until")
            state.slippage_halt_until = datetime.fromisoformat(sh) if sh else None
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
    # Cap at 95% of equity divided by worst-case fill price to ensure the
    # order cost stays within available cash after slippage and float rounding.
    qty_by_funds = effective_buying_power * 0.95 / (price * (1 + SLIPPAGE_PCT))
    return round(max(0.0, min(qty_by_risk, qty_by_funds, qty_by_exposure)), 6)


def _submit_with_guard(
    submit_fn,
    state: TradingState,
    description: str,
    sleep_enabled: bool,
    *,
    order_meta: dict | None = None,
) -> bool:
    """Submit an order and apply an error circuit-breaker on repeated failures.

    ``order_meta`` is recorded by telemetry on both submit and reject so
    post-trade analytics can pair attempts with outcomes.
    """
    if order_meta is not None:
        telemetry.record("ORDER_SUBMIT", description=description, **order_meta)
    try:
        submit_fn()
        state.order_error_streak = 0
        return True
    except Exception as exc:
        state.order_error_streak += 1
        logger.exception("Order submission failed (%s): %s", description, exc)
        if order_meta is not None:
            telemetry.record(
                "ORDER_REJECT", description=description, reason=str(exc), **order_meta
            )
        if state.order_error_streak >= MAX_ORDER_ERRORS:
            logger.error(
                "Order error circuit-breaker tripped (%d consecutive errors). Cooling down %ds.",
                state.order_error_streak,
                ORDER_ERROR_COOLDOWN_SECONDS,
            )
            if sleep_enabled:
                time.sleep(ORDER_ERROR_COOLDOWN_SECONDS)
        return False


def _record_fill(broker: BaseBroker, requested_price: float, state: TradingState) -> None:
    """Emit ORDER_FILL telemetry and arm the slippage halt if abused.

    Live brokers without a sync ``last_fill`` are no-ops here; for them
    fill telemetry needs an async polling hook (separate change).
    """
    fill = getattr(broker, "last_fill", None)
    if not fill:
        return
    slip = telemetry.slippage_bps(requested_price, fill["price"], side=fill["side"])
    telemetry.record(
        "ORDER_FILL",
        side=fill["side"],
        qty=fill["qty"],
        fill_price=fill["price"],
        requested_price=requested_price,
        slip_bps=slip,
    )
    if MAX_SLIP_BPS > 0 and abs(slip) > MAX_SLIP_BPS:
        state.slippage_halt_until = risk.slippage_halt_until(
            datetime.now(timezone.utc), SLIPPAGE_HALT_SECONDS
        )
        logger.error(
            "MAX_SLIP_BPS tripped: %.1f bps on %s fill (limit %.1f). "
            "Halting new entries until %s.",
            slip, fill["side"], MAX_SLIP_BPS, state.slippage_halt_until.isoformat(),
        )
        telemetry.record(
            "RISK_HALT",
            kind_detail="MAX_SLIP_BPS",
            slip_bps=slip,
            halt_until=state.slippage_halt_until.isoformat(),
        )


def daily_loss_exceeded(broker: BaseBroker) -> bool:
    """Return True when today's drawdown has reached ``MAX_DAILY_LOSS_PCT``."""
    equity, last_equity = broker.get_equity()
    if last_equity <= 0:
        return False
    return (last_equity - equity) / last_equity >= MAX_DAILY_LOSS_PCT


def _maybe_snapshot_week(broker: BaseBroker, state: TradingState) -> None:
    """Re-anchor the weekly-loss baseline on each new ISO week (Monday rollover)."""
    today = broker.current_date()
    anchor = risk.week_anchor(today)
    if state.week_anchor_date != anchor:
        equity, _ = broker.get_equity()
        state.week_anchor_date = anchor
        state.week_baseline_equity = equity
        logger.info("Weekly equity baseline re-anchored to %s ($%.2f).", anchor, equity)




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


def _maybe_take_partial_tp(
    broker: BaseBroker,
    state: TradingState,
    current_qty: float,
    r_multiple: float,
    latest_price: float,
    sleep_enabled: bool,
) -> bool:
    """Take partial profits at TP1/TP2 R-multiples; return True if a partial fired."""
    if current_qty == 0:
        return False
    is_long = current_qty > 0
    abs_qty = abs(current_qty)
    side = "LONG" if is_long else "SHORT"
    verb = "closed" if is_long else "covered"
    submit = broker.submit_sell if is_long else broker.submit_buy

    triggers = [
        ("partial_tp1_taken", PARTIAL_TP1_R, PARTIAL_TP1_FRACTION, "TP1"),
        ("partial_tp2_taken", PARTIAL_TP2_R, PARTIAL_TP2_FRACTION, "TP2"),
    ]
    for attr, r_mult, frac, label in triggers:
        if getattr(state, attr) or r_multiple < r_mult:
            continue
        qty_to_close = round(max(0.0, abs_qty * frac), 6)
        if not (0 < qty_to_close < abs_qty):
            continue
        order_side = "SELL" if is_long else "BUY"
        if _submit_with_guard(
            lambda q=qty_to_close: submit(SYMBOL, q),
            state,
            f"partial {label.lower()} {side.lower()}",
            sleep_enabled,
            order_meta={"side": order_side, "qty": qty_to_close, "requested_price": latest_price, "order_kind": f"PARTIAL_{label}"},
        ):
            setattr(state, attr, True)
            logger.info("PARTIAL %s %s: %s %.6f @ ~%.2f", label, side, verb, qty_to_close, latest_price)
            _record_fill(broker, latest_price, state)
            return True
    return False


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
    _maybe_snapshot_week(broker, state)
    state.cycles += 1
    broker.flush_position_cache()

    # --- Risk firewall (operator-controlled and automatic halts) ---
    if risk.kill_switch_active(KILL_SWITCH_FILE):
        logger.error("KILL SWITCH ACTIVE (%s) — halting cycle.", KILL_SWITCH_FILE)
        telemetry.record("RISK_HALT", kind_detail="KILL_SWITCH", path=KILL_SWITCH_FILE)
        if sleep_enabled:
            time.sleep(CHECK_INTERVAL_SECONDS)
        return

    if risk.is_slippage_halted(state.slippage_halt_until):
        logger.warning(
            "Slippage cooldown active until %s — skipping cycle.",
            state.slippage_halt_until.isoformat() if state.slippage_halt_until else "?",
        )
        if sleep_enabled:
            time.sleep(CHECK_INTERVAL_SECONDS)
        return

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
        telemetry.record("RISK_HALT", kind_detail="DAILY_LOSS")
        if sleep_enabled:
            time.sleep(3600)
        return

    equity_now, _ = broker.get_equity()
    if risk.weekly_loss_exceeded(equity_now, state.week_baseline_equity, MAX_WEEKLY_LOSS_PCT):
        logger.warning(
            "Weekly loss limit (%.1f%%) reached — halting until next week.",
            MAX_WEEKLY_LOSS_PCT * 100,
        )
        telemetry.record(
            "RISK_HALT",
            kind_detail="WEEKLY_LOSS",
            equity=equity_now,
            week_baseline=state.week_baseline_equity,
        )
        if sleep_enabled:
            time.sleep(3600)
        return

    # Fetch enough bars for all indicators to warm up
    bars_needed = max(TREND_WINDOW, MACD_SLOW, RSI_WINDOW) * 2 + CONFIRM_BARS + 5
    bars = broker.get_recent_bars(SYMBOL, limit=bars_needed, timeframe=TIMEFRAME)

    expected_interval = interval_seconds_for(TIMEFRAME)
    # Default freshness budget: 4 bars worth, only when broker provides timestamps
    # and live trading (sleep_enabled=True). Backtests must not be freshness-gated.
    max_age = (
        DATA_MAX_BAR_AGE_SECONDS or (expected_interval * 4 if expected_interval else None)
    ) if sleep_enabled else None
    integrity = check_bars(
        bars,
        expected_interval_seconds=expected_interval,
        max_age_seconds=max_age,
    )
    if integrity.issues:
        for msg in integrity.issues:
            level = logger.error if msg in integrity.fatal else logger.warning
            level("Data integrity: %s", msg)
        if integrity.fatal and DATA_INTEGRITY_GATE_ENABLED:
            logger.error("Data integrity gate tripped — skipping cycle.")
            if sleep_enabled:
                time.sleep(CHECK_INTERVAL_SECONDS)
            return

    latest_price = float(bars["c"].iloc[-1])
    latest_high = float(bars["h"].iloc[-1])
    latest_low = float(bars["l"].iloc[-1])

    # Compute ATR once — used for both the trailing stop check and position sizing.
    atr = atr_stop_distance(bars["h"], bars["l"], bars["c"], window=ATR_STOP_WINDOW, multiplier=1.0)
    stop_dist = max(atr * ATR_STOP_MULT, latest_price * STOP_LOSS_PCT)
    tp_dist   = max(atr * TAKE_PROFIT_MULT, latest_price * TAKE_PROFIT_PCT)

    bars_per_day = {
        "1Min": 24 * 60,
        "5Min": 24 * 12,
        "15Min": 24 * 4,
        "30Min": 24 * 2,
        "1H": 24,
        "4H": 6,
        "1D": 1,
    }.get(TIMEFRAME, 24 * 4)
    adaptive_lookback_bars = ADAPTIVE_LOOKBACK_DAYS * bars_per_day if ADAPTIVE_THRESHOLDS_ENABLED else 0

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
        adx_trending_threshold=ADX_TRENDING_THRESHOLD,
        min_atr_pct=MIN_ATR_PCT,
        adaptive_lookback_bars=adaptive_lookback_bars,
        adx_threshold_percentile=ADAPTIVE_ADX_PERCENTILE,
        min_atr_pct_percentile=ADAPTIVE_ATR_PERCENTILE,
        session_filter_enabled=SESSION_FILTER_ENABLED,
        session_start_hour_utc=SESSION_START_HOUR_UTC,
        session_end_hour_utc=SESSION_END_HOUR_UTC,
        atr_accel_window=ATR_ACCEL_WINDOW,
        min_atr_accel=MIN_ATR_ACCEL,
        structure_filter_enabled=STRUCTURE_FILTER_ENABLED,
        structure_lookback=STRUCTURE_LOOKBACK,
        volume_filter_enabled=VOLUME_FILTER_ENABLED,
        min_volume=MIN_VOLUME,
    )
    current_qty = broker.get_position_qty(SYMBOL)
    open_order_exists = broker.has_open_order(SYMBOL)
    buying_power = broker.get_buying_power()
    equity, _ = broker.get_equity()

    logger.info(
        f"{SYMBOL} | price={latest_price:.2f} | signal={signal} "
        f"| held={current_qty:.6f} | open_order={open_order_exists}"
    )
    telemetry.record(
        "SIGNAL",
        symbol=SYMBOL,
        signal=signal,
        price=latest_price,
        qty_held=current_qty,
        equity=equity,
        open_order=open_order_exists,
        cycle=state.cycles,
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
        state.seed_trade_context(0.0, 0.0)

    # --- Position Management (Trailing Stop + Take Profit) ---
    if current_qty != 0:
        entry_price = broker.get_entry_price(SYMBOL)
        if entry_price:
            if state.initial_stop_dist <= 0:
                state.seed_trade_context(float(entry_price), stop_dist)

            init_stop = max(state.initial_stop_dist, 1e-9)
            if current_qty > 0:
                r_multiple = (latest_high - state.entry_price) / init_stop
            else:
                r_multiple = (state.entry_price - latest_low) / init_stop

            if not state.break_even_armed and r_multiple >= BREAK_EVEN_R_MULT:
                state.break_even_armed = True
                state.log_event(state.cycles, "BREAK_EVEN_ARMED", {"r_multiple": round(r_multiple, 4)})
                logger.info("BREAK-EVEN armed at %.2fR", r_multiple)

            # Priority order: stop > take-profit > partial. Stops must win when the
            # same bar shows both a stop hit and a partial trigger so a sudden
            # reversal can't be ignored just because the high also touched the partial.
            long_stop_level = state.position_high - stop_dist
            short_stop_level = state.position_low + stop_dist
            if state.break_even_armed:
                long_stop_level = max(long_stop_level, state.entry_price)
                short_stop_level = min(short_stop_level, state.entry_price)
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
                exit_kind = "STOP" if exit_reason == "TRAILING STOP" else "TAKE_PROFIT"
                if current_qty > 0:
                    ok = _submit_with_guard(
                        lambda: broker.submit_sell(SYMBOL, current_qty),
                        state,
                        "exit long",
                        sleep_enabled,
                        order_meta={"side": "SELL", "qty": current_qty, "requested_price": latest_price, "order_kind": f"EXIT_LONG_{exit_kind}"},
                    )
                    ref = state.position_high
                else:
                    ok = _submit_with_guard(
                        lambda: broker.submit_buy(SYMBOL, abs(current_qty)),
                        state,
                        "exit short",
                        sleep_enabled,
                        order_meta={"side": "BUY", "qty": abs(current_qty), "requested_price": latest_price, "order_kind": f"EXIT_SHORT_{exit_kind}"},
                    )
                    ref = state.position_low
                if not ok:
                    return
                _record_fill(broker, latest_price, state)

                state.reset_watermarks(latest_price)
                state.seed_trade_context(0.0, 0.0)
                state.log_event(state.cycles, f"EXIT_{exit_reason.replace(' ', '_')}", {"price": latest_price})
                logger.warning(
                    f"{exit_reason}: closed {abs(current_qty):.6f} {SYMBOL} "
                    f"at {latest_price:.2f} (entry {entry_price:.2f}, ref {ref:.2f}, "
                    f"stop_dist={stop_dist:.2f}, tp_dist={tp_dist:.2f})"
                )
                if sleep_enabled:
                    time.sleep(CHECK_INTERVAL_SECONDS)
                return

            if _maybe_take_partial_tp(broker, state, current_qty, r_multiple, latest_price, sleep_enabled):
                return

    # --- Signal execution ---
    qty = position_size(equity, buying_power, stop_dist, latest_price)
    bar_range_pct = max(0.0, (latest_high - latest_low) / max(latest_price, 1e-9))
    latency_penalty = 0.0001 * BACKTEST_LATENCY_BARS
    partial_fill_penalty = (1.0 - BACKTEST_PARTIAL_FILL_MIN) * 0.0005
    est_slippage_pct = SLIPPAGE_PCT + BACKTEST_DYNAMIC_SLIPPAGE_K * bar_range_pct + latency_penalty + partial_fill_penalty
    commission_per_unit = (COMMISSION_PER_TRADE / max(qty, 1e-9)) if qty > 0 else 0.0
    est_cost = 2.0 * (latest_price * est_slippage_pct + commission_per_unit) + latest_price * ESTIMATED_ROUND_TRIP_COST_PCT
    rr_after_cost = (tp_dist - est_cost) / max(1e-9, stop_dist + est_cost)
    allow_new_entry = rr_after_cost >= MIN_EXPECTED_RR

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
                order_meta={"side": "BUY", "qty": abs(current_qty), "requested_price": latest_price, "order_kind": "COVER_SHORT"},
            ):
                return
            _record_fill(broker, latest_price, state)
            state.reset_watermarks(latest_price)
            logger.info(f"COVER {abs(current_qty):.6f} {SYMBOL} @ ~{latest_price:.2f}")
        else:
            if qty <= 0:
                logger.info("Insufficient buying power to open long.")
            elif not allow_new_entry:
                logger.info(
                    "Skipping long: expected RR after costs %.2f < min %.2f",
                    rr_after_cost,
                    MIN_EXPECTED_RR,
                )
            else:
                sl_price = round(latest_price - stop_dist, 8)
                tp_price = round(latest_price + tp_dist, 8)
                if not _submit_with_guard(
                    lambda: broker.submit_buy(SYMBOL, qty, sl=sl_price, tp=tp_price),
                    state,
                    "open long",
                    sleep_enabled,
                    order_meta={"side": "BUY", "qty": qty, "requested_price": latest_price, "sl": sl_price, "tp": tp_price, "order_kind": "OPEN_LONG"},
                ):
                    return
                _record_fill(broker, latest_price, state)
                state.position_high = latest_price
                state.seed_trade_context(latest_price, stop_dist)
                state.log_event(state.cycles, "OPEN_LONG", {"qty": qty, "price": latest_price})
                logger.info(
                    f"BUY   {qty:.6f} {SYMBOL} @ ~{latest_price:.2f} "
                    f"(SL={sl_price:.2f}, TP={tp_price:.2f})"
                )

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
                order_meta={"side": "SELL", "qty": current_qty, "requested_price": latest_price, "order_kind": "CLOSE_LONG"},
            ):
                return
            _record_fill(broker, latest_price, state)
            state.reset_watermarks(latest_price)
            logger.info(f"SELL  {current_qty:.6f} {SYMBOL} @ ~{latest_price:.2f}")
        else:
            if not ALLOW_SHORTS:
                logger.info("SELL signal — ALLOW_SHORTS=False, staying flat.")
            elif not broker.supports_shorting:
                logger.info("SELL signal — broker does not support shorting, staying flat.")
            elif qty <= 0:
                logger.info("Insufficient equity to open short.")
            elif not allow_new_entry:
                logger.info(
                    "Skipping short: expected RR after costs %.2f < min %.2f",
                    rr_after_cost,
                    MIN_EXPECTED_RR,
                )
            else:
                sl_price = round(latest_price + stop_dist, 8)
                tp_price = round(latest_price - tp_dist, 8)
                if not _submit_with_guard(
                    lambda: broker.submit_sell(SYMBOL, qty, sl=sl_price, tp=tp_price),
                    state,
                    "open short",
                    sleep_enabled,
                    order_meta={"side": "SELL", "qty": qty, "requested_price": latest_price, "sl": sl_price, "tp": tp_price, "order_kind": "OPEN_SHORT"},
                ):
                    return
                _record_fill(broker, latest_price, state)
                state.position_low = latest_price
                state.seed_trade_context(latest_price, stop_dist)
                state.log_event(state.cycles, "OPEN_SHORT", {"qty": qty, "price": latest_price})
                logger.info(
                    f"SHORT {qty:.6f} {SYMBOL} @ ~{latest_price:.2f} "
                    f"(SL={sl_price:.2f}, TP={tp_price:.2f})"
                )

    else:
        logger.info("No action.")

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

    telemetry.configure(TELEMETRY_FILE if TELEMETRY_ENABLED else None, enabled=TELEMETRY_ENABLED)
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
