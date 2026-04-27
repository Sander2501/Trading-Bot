# Trading Bot

[![Tests](https://github.com/Sander2501/Trading-Bot/actions/workflows/tests.yml/badge.svg)](https://github.com/Sander2501/Trading-Bot/actions/workflows/tests.yml)

A modular Python trading bot for **paper trading** with Capital.com.  
It implements a triple-EMA trend-following strategy with MACD, ADX, and RSI confirmation.

> **Important:** This project is intended for **paper/demo trading only**.  
> No real money is ever at risk. See [Known Assumptions](#known-assumptions).

---

## Project Structure

```text
Trading-Bot/
├── .env.example                  ← Copy to .env and fill credentials
├── .github/
│   └── workflows/
│       └── tests.yml             ← CI: runs tests on every push & PR
├── brokers/
│   ├── __init__.py
│   ├── base.py                   ← Abstract BaseBroker interface
│   ├── backtest.py               ← Offline CSV replay broker
│   └── capital.py                ← Capital.com live/demo API broker
├── tests/
│   ├── __init__.py
│   ├── conftest.py               ← Shared pytest fixtures
│   ├── unit/
│   │   └── test_strategy_scenarios.py  ← Deterministic strategy tests
│   ├── test_backtest_broker.py
│   ├── test_config.py
│   └── test_strategy.py
├── config.py                     ← All configuration (env vars + validation)
├── main.py                       ← Live trading entry-point
├── metrics.py                    ← TradeMetrics dataclass
├── pytest.ini                    ← Test configuration
├── requirements.txt
├── run_backtest.py               ← Backtest entry-point (generates report)
└── strategy.py                   ← Signal generation logic
```

---

## Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/Sander2501/Trading-Bot.git
cd Trading-Bot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. One-command project setup (installs all dependencies, including pytest)
make setup
# (Alternative without make: pip install -r requirements.txt)

# 4. Copy the example environment file and add your credentials
cp .env.example .env
# Edit .env — see the Environment Variables section below

# 5. Run the test suite to verify everything works
pytest -q

# 6. Run a backtest
python run_backtest.py
```

### One-command setup (recommended)

Use the helper script to create `.venv`, install pinned dependencies, and verify
core imports:

```bash
./scripts/dev_setup.sh
```

---

## Environment Variables

All configuration is loaded from a `.env` file (or real environment variables).  
Copy `.env.example` to `.env` and edit the values:

| Variable | Default | Description |
|---|---|---|
| `CAPITAL_API_KEY` | — | Capital.com API key (required for live/demo) |
| `CAPITAL_IDENTIFIER` | — | Your Capital.com email |
| `CAPITAL_PASSWORD` | — | Your Capital.com password |
| `CAPITAL_USE_DEMO` | `true` | Use demo environment (recommended) |
| `SYMBOL` | `BTC/USD` | Trading symbol |
| `TIMEFRAME` | `15Min` | Bar timeframe |
| `RISK_PER_TRADE` | `0.02` | Fraction of equity per trade (see Risk Profiles) |
| `MAX_CONSECUTIVE_ERRORS` | `5` | Consecutive bot-cycle errors before cooldown circuit breaker |
| `ERROR_COOLDOWN_SECONDS` | `900` | Cooldown duration after circuit breaker trips |
| `OPEN_ORDER_STALE_CYCLES` | `5` | Warn if open orders persist for N consecutive cycles |
| `BACKTEST_CSV` | `historical_data.csv` | Path to historical CSV for backtesting |
| `BACKTEST_STARTING_CASH` | `100000` | Starting account balance for backtests |

See `config.py` for the full list and default values.

---

## Running the Bot

### Backtest (recommended first step)

```bash
python run_backtest.py
```

Replays `historical_data.csv` and prints a performance report:

```
========================================
       BACKTEST PERFORMANCE
========================================
Starting Cash    :  $100,000.00
Final Equity     :  $102,450.25
Return %         :       +2.45%
Win Rate         :        58.3%
Profit Factor    :         1.34
Max Drawdown     :        3.21%
Sharpe Ratio     :        0.892
----------------------------------------
Avg Win          :      $125.40
Avg Loss         :      $-83.20
Total Fills      :           48
========================================
```

### Live / Demo Trading

```bash
python main.py
```

The bot connects to Capital.com, fetches recent bars, generates a signal, and places orders.  
Press `Ctrl+C` to stop.

---

## Strategy Explanation

The strategy uses a **triple-EMA trend filter** with **MACD**, **ADX**, and **RSI** confirmation:

| Indicator | Role | Default |
|---|---|---|
| Fast EMA (9) | Captures recent momentum | `FAST_WINDOW=9` |
| Slow EMA (21) | Medium-term direction | `SLOW_WINDOW=21` |
| Trend EMA (50) | Long-term bias | `TREND_WINDOW=50` |
| MACD histogram | Confirms momentum direction | Fast=12, Slow=26, Signal=9 |
| ADX | Regime filter — suppresses choppy signals | `ADX_THRESHOLD=25` |
| RSI | Entry timing (buy dips, sell spikes) | `RSI_WINDOW=14` |

**Signal logic:**

1. **ADX < threshold** → `HOLD` (sideways/choppy market, all signals suppressed)
2. **TRENDING + BULL** (price > Trend EMA, ADX > 25):  
   - Fast > Slow + MACD positive + RSI < 45 → `BUY`  
   - Fresh EMA crossover → `BUY` (early entry, can trigger before `CONFIRM_BARS`)
3. **TRENDING + BEAR**:  
   - Fast < Slow + MACD negative + RSI > 55 → `SELL`
4. **RANGING** (ADX 20–25):  
   - RSI oversold → `BUY`; RSI overbought → `SELL`

**Exit logic:** ATR-based trailing stop + fixed take-profit target.

> Regime note: the ADX gate is configurable via `ADX_THRESHOLD`; "TRENDING" is
> currently tagged when `ADX > 25` in logs.

---

## Risk Profiles

Risk is controlled by `RISK_PER_TRADE` (fraction of equity per trade).  
Position size formula: `qty = (equity × RISK_PER_TRADE) / price`

| Profile | `RISK_PER_TRADE` | Max daily loss | Suitable for |
|---|---|---|---|
| Conservative | `0.01` (1%) | ~3% | Beginners ✅ |
| **Moderate** | **`0.02` (2%)** | **~5%** | **Default** |
| Aggressive | `0.05` (5%) | ~10% | Experienced traders only |

**Hard cap:** `RISK_PER_TRADE` cannot exceed `0.10` (10%) — the bot will refuse to start.

**Recommendation for beginners:** Start with `RISK_PER_TRADE=0.01` and paper-trade for at least 30 days before adjusting.

---

## Parameter Tuning Guide

Run → analyze report → adjust → re-run:

```bash
# 1. Run baseline backtest
python run_backtest.py

# 2. Adjust parameters in .env (e.g. tighten stop)
echo "ATR_STOP_MULT=1.5" >> .env

# 3. Re-run and compare
python run_backtest.py
```

**Safe tuning ranges:**

| Parameter | Conservative | Default | Aggressive |
|---|---|---|---|
| `ADX_THRESHOLD` | 30 | 25 | 20 |
| `CONFIRM_BARS` | 3 | 2 | 1 |
| `ATR_STOP_MULT` | 3.0 | 2.0 | 1.5 |
| `RISK_PER_TRADE` | 0.01 | 0.02 | 0.05 |

> **Note on `CONFIRM_BARS`:** the strategy also has an "early crossover" path for strong trend reversals.  
> In practice, this means `CONFIRM_BARS` still reduces noise, but some fresh crossovers can enter earlier by design.

**Key metrics to watch:**

- **Win Rate > 45%** with **Profit Factor > 1.2** = viable strategy
- **Max Drawdown < 10%** = acceptable risk
- **Sharpe Ratio > 0.5** = risk-adjusted returns are positive

---

## Known Assumptions

- **Paper trading only:** This bot is designed for demo/paper accounts. It has never been tested with real money.
- **Market hours:** The bot skips cycles when the market is closed (Capital.com CFD hours). Crypto markets run 24/7, but the broker may have weekend gaps.
- **Shorting support:** Short selling (`SELL` signals opening new shorts) requires broker support. Capital.com CFDs support shorting; other brokers may not. Check `BaseBroker.supports_shorting`.
- **Backtest vs live differences:** Backtests use a fixed slippage estimate (`SLIPPAGE_PCT=0.0005`). Live trading may experience higher slippage during volatile periods. Commission defaults to $0.10/trade in backtest.
- **15-minute bars:** Strategy defaults to 15-minute timeframe. Shorter bars are noisier and can increase overtrading.
- **No guarantee of profit:** Past backtest performance does not guarantee future live results.

---


### Windows PowerShell quick commands

If you are on PowerShell (like `PS C:\...`), use PowerShell-native commands instead of Unix ones:

```powershell
# from repo root (folder that contains requirements.txt)
cd C:\path\to\Trading-Bot

# activate venv
.\.venv\Scripts\Activate.ps1

# inspect proxy-related env vars
Get-ChildItem Env: | Where-Object { $_.Name -match 'PIP|HTTP_PROXY|HTTPS_PROXY|NO_PROXY' }

# clear proxy vars for current shell session (optional)
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue

# install deps + run tests
python -m pip install -r requirements.txt
pytest -q
```

> If you see `Could not open requirements file`, you are in the wrong directory.
> Run `Get-ChildItem` and confirm `requirements.txt` is present before installing.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Run `pip install -r requirements.txt` |
| `Could not open requirements file: requirements.txt` | `cd` into the repo root first (the folder containing `requirements.txt`) and retry |
| `Missing CAPITAL_API_KEY` | Create `.env` from `.env.example` and fill credentials |
| `FileNotFoundError: historical_data.csv` | Provide a CSV with at least a `c` (close) column |
| Signal stays `HOLD` forever | ADX may be low (choppy market). Lower `ADX_THRESHOLD` or wait for a trending period |
| `ValueError: RISK_PER_TRADE … must be in (0, 0.10]` | Reduce `RISK_PER_TRADE` in `.env` to 0.02 or less |
| Tests fail on collection | Run `pip install -r requirements.txt` to install `pytest` |

---

## Running Tests

```bash
# Ensure env + deps first (safe to re-run)
./scripts/dev_setup.sh

# One-command test runner (auto-creates .venv if missing)
./scripts/run_tests.sh

# Run all tests (fast)
pytest -q

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/ -v

# Run a specific test file
pytest tests/test_strategy.py -v
```

Tests run automatically on every push and pull request via GitHub Actions.

---

## Advanced Validation & Optimization

### Walk-forward validation

```bash
python scripts/walk_forward.py --csv historical_data.csv --train-bars 8000 --test-bars 4000
```

Produces `walk_forward_report.json` with per-fold ROI/PF/Sharpe/MDD plus averages.

### Parameter robustness sweep

```bash
python scripts/sweep_params.py
```

Produces `sweep_report.json` with ranked parameter combinations.
