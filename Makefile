PYTHON ?= python
PIP ?= pip

.PHONY: setup test backtest

setup:
	$(PIP) install -r requirements.txt

test:
	pytest -q

backtest:
	$(PYTHON) run_backtest.py
