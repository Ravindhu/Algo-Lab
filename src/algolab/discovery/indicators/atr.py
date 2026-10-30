"""Average True Range: simple average of true range over `period` bars. Same numbers as the live atr()."""
from __future__ import annotations

import numpy as np


def atr_series(high, low, close, period: int) -> np.ndarray:
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    out = np.full(len(c), np.nan)
    if len(c) >= period:
        csum = np.cumsum(np.concatenate([[0.0], tr]))
        idx = np.arange(period - 1, len(c))
        out[idx] = (csum[idx + 1] - csum[idx + 1 - period]) / period
    return out
