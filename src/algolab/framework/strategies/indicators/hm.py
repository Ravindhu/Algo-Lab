"""hm.py - the ONE Hilega-Milega indicator used by every strategy.

Reference: the "HILEGA_MILEGA" TradingView indicator (Nitish Kumar):
RSI (default 9) with an EMA (default 3) plotted over it. Bullish = RSI above
EMA(RSI); bearish = RSI below. RSI uses the SIMPLE-average convention
(mean gain / mean loss over the last `rsi_period` changes), matching
algolab/hilega_milega.py.

All functions are causal: reading bar j uses closes[0..j] only.

Consumers:
  - strategies/ocl.py            hm_bullish (HTF gate no cap; LTF gate + oversold cap)
  - strategies/engulfing_strategy.py  hm_direction series aligned with pattern bars
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def hm_rsi_series(closes: Sequence[float], rsi_period: int = 9) -> np.ndarray:
    """Simple-average RSI per bar; NaN until rsi_period bars are available."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    p = int(rsi_period)
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


def hm_ema_series(values: Sequence[Optional[float]], period: int = 3) -> np.ndarray:
    """EMA over a series that may start with NaNs; NaN until `period` defined points exist.

    The EMA is seeded with the first defined value (same as algolab.hilega_milega.ema)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    k = 2.0 / (int(period) + 1.0)
    out = np.full(n, np.nan)
    prev: Optional[float] = None
    defined = 0
    for i in range(n):
        x = v[i]
        if np.isnan(x):
            continue
        defined += 1
        prev = float(x) if prev is None else k * float(x) + (1 - k) * prev
        if defined >= period:
            out[i] = prev
    return out


def hm_direction(closes: Sequence[float], rsi_period: int = 9,
                 ema_period: int = 3) -> np.ndarray:
    """Per-bar Hilega-Milega direction: +1 RSI > EMA(RSI), -1 RSI < EMA(RSI), 0 undefined/equal."""
    rsi = hm_rsi_series(closes, rsi_period)
    line = hm_ema_series(rsi, ema_period)
    out = np.zeros(len(rsi), dtype=int)
    ok = ~np.isnan(rsi) & ~np.isnan(line)
    out[ok & (rsi > line)] = 1
    out[ok & (rsi < line)] = -1
    return out


def hm_bullish(closes: Sequence[float], rsi_period: int = 9, ema_period: int = 3,
               oversold: Optional[float] = None) -> bool:
    """Latest-bar Hilega-Milega gate.

    True when RSI is above EMA(RSI). With `oversold` given, RSI must ALSO be at or
    below it (the LTF recovery rule: bull only while still near oversold)."""
    rsi = hm_rsi_series(closes, rsi_period)
    line = hm_ema_series(rsi, ema_period)
    if len(rsi) == 0 or np.isnan(rsi[-1]) or np.isnan(line[-1]):
        return False
    if rsi[-1] <= line[-1]:
        return False
    return True if oversold is None else bool(rsi[-1] <= oversold)


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
        period: int = 14) -> list[Optional[float]]:
    """Average True Range (Wilder-style simple average), same numbers as
    algolab.hilega_milega.atr: list of None until `period` TR values exist."""
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("atr: OHLC series length mismatch")
    trs: list[Optional[float]] = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[i] - lows[i])
            continue
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    out: list[Optional[float]] = []
    for i in range(len(trs)):
        if i < period - 1:
            out.append(None)
            continue
        w = [t for t in trs[i - period + 1:i + 1] if t is not None]
        out.append(sum(w) / len(w) if w else None)
    return out
