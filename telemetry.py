"""
telemetry
~~~~~~~~~
Lightweight JSONL event logger for execution diagnostics.

Records signal timing, order submits, fills, rejects, and slippage so that
post-trade analytics can answer questions like "where did our pnl leak?"
and "what fraction of cycles produced an actionable signal?".

The logger is process-wide and append-only. All file errors are swallowed
so a broken telemetry file never crashes the bot — telemetry is observability,
not a hard dependency.

Usage::

    from telemetry import record
    record("ORDER_SUBMIT", side="BUY", qty=0.1, requested_price=70_000)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_path: Path | None = None
_enabled: bool = False
_warned_disabled: bool = False


def configure(path: str | os.PathLike[str] | None, *, enabled: bool = True) -> None:
    """Set the telemetry sink. Called once at startup from :mod:`config`."""
    global _path, _enabled, _warned_disabled
    _enabled = bool(enabled and path)
    _path = Path(path) if path else None
    _warned_disabled = False


def record(kind: str, **fields: object) -> None:
    """Append one JSON event to the telemetry file. Never raises."""
    global _warned_disabled
    if not _enabled or _path is None:
        if not _warned_disabled:
            logger.debug("Telemetry disabled or unconfigured; events will be dropped.")
            _warned_disabled = True
        return
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        **_serializable(fields),
    }
    try:
        line = json.dumps(event)
    except (TypeError, ValueError) as exc:
        logger.warning("telemetry.record(%s): could not serialize event (%s)", kind, exc)
        return
    try:
        with _lock:
            with _path.open("a") as f:
                f.write(line + "\n")
    except OSError as exc:
        logger.warning("telemetry.record(%s): file write failed (%s)", kind, exc)


def _serializable(fields: dict[str, object]) -> dict[str, object]:
    """Coerce common non-JSON types to strings/floats so json.dumps doesn't fail."""
    out: dict[str, object] = {}
    for k, v in fields.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, datetime):
            out[k] = v.astimezone(timezone.utc).isoformat() if v.tzinfo else v.isoformat()
        else:
            out[k] = repr(v)
    return out


def slippage_bps(requested: float, filled: float, *, side: str) -> float:
    """Return signed slippage in basis points (positive = worse than requested)."""
    if requested <= 0:
        return 0.0
    diff = filled - requested if side.upper() in {"BUY", "COVER"} else requested - filled
    return diff / requested * 10_000.0
