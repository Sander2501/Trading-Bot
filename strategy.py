import pandas as pd


def moving_average_signal(
    closes: pd.Series,
    window: int = 3,
    confirm_bars: int = 1,
) -> str:
    if len(closes) < window:
        return "HOLD"

    ma = float(closes.tail(window).mean())
    latest = float(closes.iloc[-1])

    print(f"latest={latest:.2f}, ma={ma:.2f}")

    if latest > ma:
        return "BUY"
    elif latest < ma:
        return "SELL"
    else:
        return "HOLD"