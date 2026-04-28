import json

import pytest

import telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    telemetry.configure(None, enabled=False)
    yield
    telemetry.configure(None, enabled=False)


def test_disabled_record_writes_nothing(tmp_path):
    path = tmp_path / "events.jsonl"
    telemetry.configure(path, enabled=False)
    telemetry.record("SIGNAL", price=100.0)
    assert not path.exists()


def test_enabled_record_writes_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    telemetry.configure(path, enabled=True)
    telemetry.record("SIGNAL", price=100.0, signal="BUY", qty_held=0.0)
    telemetry.record("ORDER_SUBMIT", side="BUY", qty=0.5, requested_price=100.0)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    sig = json.loads(lines[0])
    assert sig["kind"] == "SIGNAL"
    assert sig["signal"] == "BUY"
    assert sig["price"] == 100.0
    assert "ts" in sig
    submit = json.loads(lines[1])
    assert submit["kind"] == "ORDER_SUBMIT"
    assert submit["qty"] == 0.5


def test_record_swallows_unserializable_fields(tmp_path):
    path = tmp_path / "events.jsonl"
    telemetry.configure(path, enabled=True)
    telemetry.record("WEIRD", obj=object())
    line = json.loads(path.read_text().strip())
    assert "obj" in line  # repr fallback applied
    assert isinstance(line["obj"], str)


def test_record_swallows_file_errors(tmp_path):
    # Configure to a path under a non-existent directory so writes fail.
    telemetry.configure(tmp_path / "nope" / "events.jsonl", enabled=True)
    # Should not raise
    telemetry.record("SIGNAL", price=1.0)


@pytest.mark.parametrize(
    "side,requested,filled,expected_bps",
    [
        ("BUY", 100.0, 100.5, 50.0),       # paid more than requested = positive slip
        ("BUY", 100.0, 99.5, -50.0),       # paid less = negative slip
        ("SELL", 100.0, 99.5, 50.0),       # got less than requested = positive slip
        ("SELL", 100.0, 100.5, -50.0),     # got more than requested = negative slip
        ("COVER", 100.0, 100.0, 0.0),
    ],
)
def test_slippage_bps(side, requested, filled, expected_bps):
    assert telemetry.slippage_bps(requested, filled, side=side) == expected_bps


def test_slippage_bps_zero_requested():
    assert telemetry.slippage_bps(0.0, 100.0, side="BUY") == 0.0
