import pandas as pd


def moving_average_signal(closes: pd.Series, window: int = 20) -> str:
    """
    Returns BUY, SELL, or HOLD based on price/MA crossover:
    - BUY:  price crossed above the moving average
    - SELL: price crossed below the moving average
    - HOLD: no crossover, or not enough data
    """
    if len(closes) < window + 1:
        return "HOLD"

    ma = float(closes.tail(window).mean())
    prev_price = float(closes.iloc[-2])
    curr_price = float(closes.iloc[-1])

    if prev_price <= ma and curr_price > ma:
        return "BUY"
    elif prev_price >= ma and curr_price < ma:
        return "SELL"
    return "HOLD"