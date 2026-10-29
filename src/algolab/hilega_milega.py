"""hilega_milega.py - Hilega-Milega indicator math (simplified NK rule).
======================================================

Reference: the "HILEGA_MILEGA" TradingView indicator (Nitish Kumar):
RSI (default 9) with an EMA (default 3) plotted over it. This module
provides the series math only; the strategy gates live in
algolab.signals.engulfing (engulfing + HM) and algolab.framework.strategies.ocl.

Simplified operator rule: the WMA(RSI) leg is NOT required. Bullish is
RSI above EMA(RSI) AND RSI at/below the oversold level (default 30) -
the recovery cross FROM oversold, on both the HTF and the LTF. """
from __future__ import annotations

from typing import List, Optional, Sequence


def _clean(closes: Sequence[Optional[float]]) -> List[float]:
    """Drop trailing Nones is NOT done here; None values abort the series."""
    if any(v is None for v in closes):
        raise ValueError("hilega_milega: None in close series not supported")
    return [float(v) for v in closes]


def hilega_milega_rsi(closes: Sequence[Optional[float]], *,
                      rsi_period: int = 9) -> List[Optional[float]]:
    """Simple-average RSI (same convention as ml.features.engineering.rsi).

    Returns a list the same length as ``closes``; ``None`` until the
    first ``rsi_period`` bars are available.
    """
    vals = _clean(closes)
    out: List[Optional[float]] = []
    for i in range(len(vals)):
        if i < rsi_period:
            out.append(None)
            continue
        gains = losses = 0.0
        for a, b in zip(vals[i - rsi_period:i], vals[i - rsi_period + 1:i + 1]):
            d = b - a
            if d >= 0:
                gains += d
            else:
                losses -= d
        if gains + losses == 0:
            out.append(50.0)
        else:
            rs = gains / losses if losses > 0 else float("inf")
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out


def ema(values: Sequence[Optional[float]], period: int) -> List[Optional[float]]:
    """EMA of a series; ``None`` until ``period`` points are defined."""
    out: List[Optional[float]] = []
    k = 2.0 / (period + 1.0)
    prev: Optional[float] = None
    defined = 0
    for v in values:
        if v is None:
            out.append(None)
            continue
        defined += 1
        prev = v if prev is None else (k * v + (1 - k) * prev)
        out.append(None if defined < period else prev)
    return out


def atr(highs: Sequence[float], lows: Sequence[float],
        closes: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """Average True Range (Wilder-style simple average).

    TR = max(high-low, |high-prev_close|, |low-prev_close|). Returns a
    list the same length as the inputs; ``None`` until ``period`` TR
    values are available. Used by the ATR trailing stop in the OCL
    strategy (parameters: atr_period, atr_multiplier).
    """
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("atr: OHLC series length mismatch")
    trs: List[Optional[float]] = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[i] - lows[i])
            continue
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    out: List[Optional[float]] = []
    for i in range(len(trs)):
        if i < period - 1:
            out.append(None)
            continue
        window = [t for t in trs[i - period + 1:i + 1] if t is not None]
        out.append(sum(window) / len(window) if window else None)
    return out


def wma(values: Sequence[Optional[float]],
        period: int) -> List[Optional[float]]:
    """Weighted moving average; ``None`` until the window is full."""
    out: List[Optional[float]] = []
    n = period
    weights = list(range(1, n + 1))
    denom = n * (n + 1) / 2.0
    for i in range(len(values)):
        if i < n - 1:
            out.append(None)
            continue
        window = values[i - n + 1:i + 1]
        if any(v is None for v in window):
            out.append(None)
            continue
        out.append(sum(w * float(v) for w, v in zip(weights, window)) / denom)
    return out
