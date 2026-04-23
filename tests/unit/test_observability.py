import json

import main


def test_emit_cycle_metrics_writes_jsonl(tmp_path, monkeypatch):
    out = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(main, "METRICS_LOG_PATH", str(out))
    monkeypatch.setattr(main, "SYMBOL", "BTC/USD")
    monkeypatch.setattr(main, "TIMEFRAME", "1Min")

    state = main.TradingState()
    state.cycles = 3

    main._emit_cycle_metrics(
        state=state,
        latest_price=123.45,
        signal="HOLD",
        current_qty=0.0,
        action="NO_ACTION",
        open_order_exists=False,
        equity=1000.0,
    )

    line = out.read_text(encoding="utf-8").strip()
    payload = json.loads(line)

    assert payload["cycle"] == 3
    assert payload["symbol"] == "BTC/USD"
    assert payload["timeframe"] == "1Min"
    assert payload["action"] == "NO_ACTION"
    assert payload["equity"] == 1000.0


def test_emit_cycle_metrics_appends_records(tmp_path, monkeypatch):
    out = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(main, "METRICS_LOG_PATH", str(out))

    state = main.TradingState()
    state.cycles = 1
    main._emit_cycle_metrics(state, 100.0, "BUY", 0.1, "OPEN_LONG", False, 10000.0)

    state.cycles = 2
    main._emit_cycle_metrics(state, 101.0, "SELL", 0.0, "CLOSE_LONG", False, 10050.0)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
