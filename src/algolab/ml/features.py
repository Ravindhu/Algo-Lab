"""Features for the trade filter. Every value at bar t uses only bars up to and including t (no look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["rsi", "rsi_slope", "atr_pct", "ema_gap", "body_ratio", "range_pct",
            "ret5", "ret20", "hour", "dow"]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Standard Wilder-style RSI used by the ML feature set."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(df: pd.DataFrame, rsi_n: int = 14, atr_n: int = 14) -> pd.DataFrame:
    # Imported here, not at module top: backtest.engine -> framework -> strategies.scalp
    # -> ml.features would otherwise re-enter backtest.engine while it is still loading.
    from algolab.backtest.engine import atr

    c = df["close"]
    r = rsi(c, rsi_n)
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    f = pd.DataFrame({
        "rsi": r,
        "rsi_slope": r - r.shift(3),
        "atr_pct": atr(df, atr_n) / c,
        "ema_gap": (e20 - e50) / c,
        "body_ratio": body / rng,
        "range_pct": rng / c,
        "ret5": c.pct_change(5),
        "ret20": c.pct_change(20),
        "hour": df.index.hour,
        "dow": df.index.dayofweek,
    }, index=df.index)
    return f.replace([np.inf, -np.inf], np.nan)
