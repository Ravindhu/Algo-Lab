# Ready-made strategy classes for the framework.
from __future__ import annotations

import pandas as pd

from algolab.framework.strategies.base import Strategy
from algolab.hilega_milega import atr as atr_series
from algolab.signals.engulfing import DEFAULT_MIN_BODY_PCT, engulfing, hm_direction_series

FIB_ENTRY = 0.886   # entry sits 88.6% of the way back into the engulfing candle's range


def fib_entry_level(high: float, low: float, direction: int, fib: float = FIB_ENTRY) -> float:
    """Entry level inside the engulfing (2nd) candle, measured from its extreme in the trade direction.
    BUY : high - fib * range (a deep pullback from the high).
    SELL: low  + fib * range (a deep pullback from the low)."""
    rng = high - low
    return high - fib * rng if direction > 0 else low + fib * rng


# Engulfing entries are confirmed by the Hilega-Milega RSI/EMA rule on the pattern bar:
# BUY needs a bullish engulfing with RSI above EMA(RSI); SELL mirrors it.
# Entry rule: after the pattern, the trade fires on the FIRST closed bar whose range reaches the fib
# entry level, within max_wait bars of the pattern. Earlier touches void that pattern.
class EngulfingHmStrategy(Strategy):
    name = "engulfing_hm"

    def __init__(self, symbol, timeframe="H4", count=300, rsi_period=9, ema_period=3,
                 atr_period=14, sl_mult=1.5, tp_mult=3.0, fib_pct=FIB_ENTRY,
                 min_body_pct=DEFAULT_MIN_BODY_PCT, max_wait=5, **extra):
        super().__init__(symbol, timeframe, count, rsi_period=rsi_period,
                         ema_period=ema_period, atr_period=atr_period, sl_mult=sl_mult,
                         tp_mult=tp_mult, fib_pct=fib_pct, min_body_pct=min_body_pct,
                         max_wait=max_wait, **extra)
        if not 0.0 < fib_pct < 1.0:
            raise ValueError("fib_pct must be within (0, 1)")
        if max_wait < 1:
            raise ValueError("max_wait must be >= 1 bar")
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.fib_pct = fib_pct
        self.min_body_pct = min_body_pct
        self.max_wait = int(max_wait)

    def signal(self, df: pd.DataFrame) -> dict | None:
        if len(df) < max(self.rsi_period + self.ema_period + 3, self.atr_period + 2):
            return None
        last = len(df) - 2            # newest CLOSED candle; the final row is still forming
        eng = engulfing(df, self.min_body_pct).to_numpy()
        hm = hm_direction_series([float(x) for x in df["close"].iloc[: last + 1]],
                                 self.rsi_period, self.ema_period)
        lows = df["low"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)

        hit = None
        # Newest pattern first. The pattern bar j must be before the touch bar and inside the wait window.
        for j in range(last - 1, max(0, last - self.max_wait) - 1, -1):
            direction = int(eng[j])
            if direction == 0 or int(hm[j]) != direction:
                continue
            level = fib_entry_level(highs[j], lows[j], direction, self.fib_pct)
            touched = [k for k in range(j + 1, last + 1)
                       if (lows[k] <= level if direction > 0 else highs[k] >= level)]
            if touched and touched[0] == last:          # first touch is the newest closed bar
                hit = (j, direction, level)
                break
        if hit is None:
            return None

        j, direction, level = hit
        closes = [float(x) for x in df["close"].iloc[: last + 1]]
        highs_l = [float(x) for x in highs[: last + 1]]
        lows_l = [float(x) for x in lows[: last + 1]]
        atr_value = atr_series(highs_l, lows_l, closes, period=self.atr_period)[-1]
        if atr_value is None or atr_value <= 0:
            return None
        return {"side": "BUY" if direction > 0 else "SELL",
                "entry": float(level), "fib_pct": self.fib_pct,
                "pattern_time": str(df.index[j]),
                "atr": float(atr_value),
                "sl_dist": self.sl_mult * float(atr_value),
                "tp_dist": self.tp_mult * float(atr_value),
                "time": str(df.index[last])}


# Plugin registry export (framework_runner loads strategies by name through this).
STRATEGY = EngulfingHmStrategy
