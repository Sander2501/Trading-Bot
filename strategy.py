import pandas as pd


def moving_average_signal(
    closes: pd.Series,
    window: int = 20,
    confirm_bars: int = 1,
) -> str:
    """
    Returns BUY, SELL, or HOLD based on price vs. moving average.

    BUY  if the last `confirm_bars` closes are all above the MA.
    SELL if the last `confirm_bars` closes are all below the MA.
    HOLD otherwise.

    The MA is computed over the `window` bars preceding the confirmation
    window so the MA and the comparison prices are independent.
    """
    if confirm_bars < 1:
        raise ValueError("confirm_bars must be >= 1")

    required = window + confirm_bars
    if len(closes) < required:
        return "HOLD"

    ma = float(closes.iloc[-required:-confirm_bars].mean())
    recent = closes.iloc[-confirm_bars:].astype(float)

    if (recent > ma).all():
        return "BUY"
    if (recent < ma).all():
        return "SELL"
    return "HOLD"
