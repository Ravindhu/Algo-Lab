"""Hilega-Milega: simple-average RSI with an EMA over it. Same convention as framework/strategies/indicators/hm.py.

Every function reads only the closes it is given, so a caller that passes a closed-bar window gets the
same result the live code gets for that window (the EMA is seeded at the start of the window)."""
from __future__ import annotations

import numpy as np


def rsi_simple(closes, period: int) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    n, p = len(c), int(period)
    out = np.full(n, np.nan)
    if p <= 0 or n <= p:
        return out
    d = np.diff(c)
    g = np.concatenate([[0.0], np.cumsum(np.clip(d, 0, None))])
    lo = np.concatenate([[0.0], np.cumsum(np.clip(-d, 0, None))])
    idx = np.arange(p, n)
    gains = g[idx] - g[idx - p]
    losses = lo[idx] - lo[idx - p]
    safe = np.where(losses > 0, losses, 1.0)
    rs = np.where(losses > 0, gains / safe, np.inf)
    out[idx] = np.where(gains + losses == 0, 50.0, 100.0 - 100.0 / (1.0 + rs))
    return out


def ema_of(values, period: int) -> np.ndarray:
    """EMA seeded with the first defined value; defined once `period` values exist."""
    v = np.asarray(values, dtype=float)
    k = 2.0 / (int(period) + 1.0)
    out = np.full(len(v), np.nan)
    prev, defined = None, 0
    for i, x in enumerate(v):
        if np.isnan(x):
            continue
        defined += 1
        prev = float(x) if prev is None else k * float(x) + (1 - k) * prev
        if defined >= period:
            out[i] = prev
    return out


def direction(closes, rsi_period: int, ema_period: int) -> np.ndarray:
    """+1 where RSI is above EMA(RSI), -1 below, 0 where undefined or equal."""
    rsi = rsi_simple(closes, rsi_period)
    line = ema_of(rsi, ema_period)
    out = np.zeros(len(rsi), dtype=int)
    ok = ~np.isnan(rsi) & ~np.isnan(line)
    out[ok & (rsi > line)] = 1
    out[ok & (rsi < line)] = -1
    return out


def bullish_last(closes, rsi_period: int, ema_period: int, oversold: float | None = None) -> bool:
    """Latest bar: RSI above EMA(RSI). With `oversold`, RSI must also be at or below it."""
    rsi = rsi_simple(closes, rsi_period)
    line = ema_of(rsi, ema_period)
    if len(rsi) == 0 or np.isnan(rsi[-1]) or np.isnan(line[-1]):
        return False
    if rsi[-1] <= line[-1]:
        return False
    return True if oversold is None else bool(rsi[-1] <= oversold)
