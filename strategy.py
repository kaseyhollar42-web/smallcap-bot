"""
The ranking rules. Used by BOTH the live bot and the backtester, so what you
test is exactly what trades.

Every day, each stock gets a momentum score:
    score = 0.3 x (1-month gain) + 0.5 x (3-month gain) + 0.2 x (6-month gain)
(weights are set in config.yaml; newer stocks without 6 months of history use
their 3-month gain in its place).

A stock can only be bought if it passes every filter: price >= $3, enough
daily trading volume, ~3 months of history, and (by default) up over the
last 3 months. The bot buys the top-ranked names and sells a holding once it
slips out of the top `hold_rank`, or if it hits the stop loss.
"""

import pandas as pd

from common import IEX_VOLUME_SHARE


def momentum_panel(close: pd.DataFrame, volume: pd.DataFrame, cfg: dict, feed: str = "sip") -> dict:
    """Compute scores, filters and ranks for every stock on every day."""
    m, f = cfg["momentum"], cfg["filters"]
    w = m["weights"]
    r21 = close / close.shift(21) - 1
    r63 = close / close.shift(63) - 1
    r126 = close / close.shift(126) - 1
    score = w["r21"] * r21 + w["r63"] * r63 + w["r126"] * r126.fillna(r63)

    dollar_vol = (close * volume).rolling(20, min_periods=15).mean()
    if feed == "iex":
        dollar_vol = dollar_vol / IEX_VOLUME_SHARE  # estimate whole-market volume
    history = close.notna().cumsum()

    eligible = (
        (close >= f["min_price"])
        & (dollar_vol >= f["min_dollar_volume"])
        & (history >= f["min_history_days"])
        & score.notna()
    )
    if m.get("require_positive", True):
        eligible &= r63 > 0

    rank = score.where(eligible).rank(axis=1, ascending=False, method="first")
    return {"score": score, "rank": rank, "eligible": eligible,
            "r21": r21, "r63": r63, "r126": r126, "dollar_vol": dollar_vol}


def latest_table(close: pd.DataFrame, volume: pd.DataFrame, cfg: dict, feed: str = "sip") -> pd.DataFrame:
    """Today's ranking table (based on the last completed trading day)."""
    p = momentum_panel(close, volume, cfg, feed)
    last = close.index[-1]
    t = pd.DataFrame({
        "price": close.loc[last],
        "r21_%": p["r21"].loc[last] * 100,
        "r63_%": p["r63"].loc[last] * 100,
        "r126_%": p["r126"].loc[last] * 100,
        "score": p["score"].loc[last],
        "dollar_vol_M": p["dollar_vol"].loc[last] / 1e6,
        "eligible": p["eligible"].loc[last].fillna(False).astype(bool),
        "rank": p["rank"].loc[last],
    })
    return t.sort_values(["rank", "score"], ascending=[True, False], na_position="last")


def history_days_needed() -> int:
    return 126 + 10
