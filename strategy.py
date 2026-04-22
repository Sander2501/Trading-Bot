import pandas as pd


def moving_average_signal(closes: pd.Series, window: int = 20) -> str:
    """
    Returns BUY, SELL, or HOLD based on:
    - BUY: latest price > moving average
    - SELL: latest price < moving average
    - HOLD: not enough data or equal
    """
    if len(closes) < window:
        return "HOLD"

    latest_price = float(closes.iloc[-1])
    moving_average = float(closes.tail(window).mean())

    if latest_price > moving_average:
        return "BUY"
    elif latest_price < moving_average:
        return "SELL"
    else:
        return "HOLD"