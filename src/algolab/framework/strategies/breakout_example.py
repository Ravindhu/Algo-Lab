# EXAMPLE plug-and-play strategy: 3-bar momentum breakout.
# Copy this file, rename the class + `name`, tweak signal(), add one line to
# config/framework.json — nothing else to change anywhere else.
from __future__ import annotations

import pandas as pd

from algolab.framework.strategies.base import Strategy
from algolab.hilega_milega import atr as atr_series


class BreakoutExampleStrategy(Strategy):
    # BUY when the last closed bar closes above the highest close of the lookback before it;
    # SELL mirrors it. ATR-multiple stop/target like the other distance-based strategies.
    name = "breakout_example"

    def __init__(self, symbol, timeframe="M15", count=300, lookback=20,
                 atr_period=14, sl_mult=1.5, tp_mult=3.0, **extra):
        super().__init__(symbol, timeframe, count, lookback=lookback, atr_period=atr_period,
                         sl_mult=sl_mult, tp_mult=tp_mult, **extra)
        self.lookback = lookback
        self.atr_period = atr_period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def signal(self, df: pd.DataFrame) -> dict | None:
        if len(df) < max(self.lookback + 2, self.atr_period + 2):
            return None
        i = len(df) - 2                       # last CLOSED bar; the final row is still forming
        # Compare the last closed bar against the lookback before it. A close above the prior
        # highs is a breakout; note the still-forming final row is never the breakout bar.
        close = float(df["close"].iloc[-1])
        prior_high = float(df["high"].iloc[i - self.lookback + 1:i + 1].max())
        prior_low = float(df["low"].iloc[i - self.lookback + 1:i + 1].min())
        if close > prior_high:
            side = "BUY"
        elif close < prior_low:
            side = "SELL"
        else:
            return None
        highs = [float(x) for x in df["high"].iloc[: i + 1]]
        lows = [float(x) for x in df["low"].iloc[: i + 1]]
        closes = [float(x) for x in df["close"].iloc[: i + 1]]
        atr_v = atr_series(highs, lows, closes, period=self.atr_period)[-1]
        if atr_v is None or atr_v <= 0:
            return None
        return {"side": side, "atr": float(atr_v),
                "sl_dist": self.sl_mult * float(atr_v),
                "tp_dist": self.tp_mult * float(atr_v),
                "time": str(df.index[i])}


# Plugin registry export — this one line is what makes the strategy discoverable.
STRATEGY = BreakoutExampleStrategy
