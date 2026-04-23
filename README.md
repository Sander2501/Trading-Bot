# Trading Bot

A modular Python trading bot built for paper trading with Alpaca.

## Overview

This project implements a simple automated trading bot that:
- connects to Alpaca paper trading
- retrieves recent market data
- generates trading signals using a moving average strategy
- places paper buy/sell orders
- avoids duplicate open orders
- logs actions for debugging and review

This project is intended for learning, experimentation, and team development. It does **not** use real money.

---

## Current Project Structure

```text
Trading-Bot/
├── .env.example
├── .gitignore
├── broker.py
├── main.py
├── requirements.txt
└── strategy.py

# Run the backtest with current CSV — should complete without errors
python backtest.py
# After changes, run again and compare output metrics
python backtest.py