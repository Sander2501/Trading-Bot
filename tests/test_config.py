"""
tests.test_config
~~~~~~~~~~~~~~~~~
Unit tests for config.py validation logic.
"""

import pytest


# ---------------------------------------------------------------------------
# validate_config – direct function tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Test validate_config() with various invalid parameter combinations."""

    def _patched_validate(self, overrides: dict):
        """
        Import and call validate_config() with patched constant values.
        All defaults are valid; only keys in *overrides* are replaced.
        """
        import config as cfg_module
        from config import validate_config

        defaults = {
            "FAST_WINDOW":       9,
            "SLOW_WINDOW":       21,
            "TREND_WINDOW":      50,
            "RSI_OVERSOLD":      25.0,
            "RSI_OVERBOUGHT":    75.0,
            "RISK_PER_TRADE":    0.02,
            "MAX_DAILY_LOSS_PCT": 0.03,
            "STOP_LOSS_PCT":     0.005,
            "ATR_STOP_MULT":     2.0,
            "STARTING_CASH":     100_000.0,
        }
        merged = {**defaults, **overrides}

        # Temporarily patch module constants and call validate_config
        original = {k: getattr(cfg_module, k) for k in merged}
        try:
            for k, v in merged.items():
                setattr(cfg_module, k, v)
            validate_config()
        finally:
            for k, v in original.items():
                setattr(cfg_module, k, v)

    def test_valid_defaults_pass(self):
        """Default configuration should pass validation without error."""
        self._patched_validate({})

    def test_fast_equal_slow_raises(self):
        with pytest.raises(ValueError, match="FAST_WINDOW"):
            self._patched_validate({"FAST_WINDOW": 21, "SLOW_WINDOW": 21})

    def test_fast_greater_than_slow_raises(self):
        with pytest.raises(ValueError, match="FAST_WINDOW"):
            self._patched_validate({"FAST_WINDOW": 30, "SLOW_WINDOW": 21})

    def test_slow_equal_trend_raises(self):
        with pytest.raises(ValueError, match="SLOW_WINDOW"):
            self._patched_validate({"SLOW_WINDOW": 50, "TREND_WINDOW": 50})

    def test_slow_greater_than_trend_raises(self):
        with pytest.raises(ValueError, match="SLOW_WINDOW"):
            self._patched_validate({"SLOW_WINDOW": 60, "TREND_WINDOW": 50})

    def test_rsi_oversold_equal_overbought_raises(self):
        with pytest.raises(ValueError, match="RSI"):
            self._patched_validate({"RSI_OVERSOLD": 50.0, "RSI_OVERBOUGHT": 50.0})

    def test_rsi_oversold_greater_than_overbought_raises(self):
        with pytest.raises(ValueError, match="RSI"):
            self._patched_validate({"RSI_OVERSOLD": 80.0, "RSI_OVERBOUGHT": 20.0})

    def test_rsi_oversold_zero_raises(self):
        with pytest.raises(ValueError, match="RSI"):
            self._patched_validate({"RSI_OVERSOLD": 0.0})

    def test_rsi_overbought_100_raises(self):
        with pytest.raises(ValueError, match="RSI"):
            self._patched_validate({"RSI_OVERBOUGHT": 100.0})

    def test_risk_per_trade_zero_raises(self):
        with pytest.raises(ValueError, match="RISK_PER_TRADE"):
            self._patched_validate({"RISK_PER_TRADE": 0.0})

    def test_risk_per_trade_too_high_raises(self):
        with pytest.raises(ValueError, match="RISK_PER_TRADE"):
            self._patched_validate({"RISK_PER_TRADE": 0.6})

    def test_risk_per_trade_negative_raises(self):
        with pytest.raises(ValueError, match="RISK_PER_TRADE"):
            self._patched_validate({"RISK_PER_TRADE": -0.01})

    def test_max_daily_loss_zero_raises(self):
        with pytest.raises(ValueError, match="MAX_DAILY_LOSS_PCT"):
            self._patched_validate({"MAX_DAILY_LOSS_PCT": 0.0})

    def test_max_daily_loss_above_one_raises(self):
        with pytest.raises(ValueError, match="MAX_DAILY_LOSS_PCT"):
            self._patched_validate({"MAX_DAILY_LOSS_PCT": 1.1})

    def test_stop_loss_negative_raises(self):
        with pytest.raises(ValueError, match="STOP_LOSS_PCT"):
            self._patched_validate({"STOP_LOSS_PCT": -0.01})

    def test_atr_stop_mult_zero_raises(self):
        with pytest.raises(ValueError, match="ATR_STOP_MULT"):
            self._patched_validate({"ATR_STOP_MULT": 0.0})

    def test_atr_stop_mult_negative_raises(self):
        with pytest.raises(ValueError, match="ATR_STOP_MULT"):
            self._patched_validate({"ATR_STOP_MULT": -1.0})

    def test_starting_cash_zero_raises(self):
        with pytest.raises(ValueError, match="BACKTEST_STARTING_CASH"):
            self._patched_validate({"STARTING_CASH": 0.0})

    def test_starting_cash_negative_raises(self):
        with pytest.raises(ValueError, match="BACKTEST_STARTING_CASH"):
            self._patched_validate({"STARTING_CASH": -500.0})

    def test_boundary_risk_per_trade_max_passes(self):
        """RISK_PER_TRADE of exactly 0.10 (hard cap) should be allowed."""
        self._patched_validate({"RISK_PER_TRADE": 0.10})

    def test_boundary_max_daily_loss_one_passes(self):
        """MAX_DAILY_LOSS_PCT of exactly 1.0 should be allowed."""
        self._patched_validate({"MAX_DAILY_LOSS_PCT": 1.0})
