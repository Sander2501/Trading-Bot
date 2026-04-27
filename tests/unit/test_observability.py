import main


def test_emit_cycle_metrics_noop_does_not_create_file(tmp_path):
    state = main.TradingState()
    state.cycles = 3

    result = main._emit_cycle_metrics(
        state=state,
        latest_price=123.45,
        signal="HOLD",
        current_qty=0.0,
        action="NO_ACTION",
        open_order_exists=False,
        equity=1000.0,
    )

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_emit_cycle_metrics_is_repeatable_noop(tmp_path):
    state = main.TradingState()
    state.cycles = 1
    assert main._emit_cycle_metrics(state, 100.0, "BUY", 0.1, "OPEN_LONG", False, 10000.0) is None

    state.cycles = 2
    assert main._emit_cycle_metrics(state, 101.0, "SELL", 0.0, "CLOSE_LONG", False, 10050.0) is None
    assert list(tmp_path.iterdir()) == []
