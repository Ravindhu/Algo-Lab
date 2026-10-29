# Ready-made strategy classes for the framework.
from __future__ import annotations

import pandas as pd

from algolab.framework.strategies.base import Strategy
from algolab.hilega_milega import atr as atr_series
from algolab.hilega_milega import ema, hilega_milega_rsi
from algolab.signals.engulfing import engulfing


# Engulfing entries are confirmed by the Hilega-Milega RSI/EMA rule:
# BUY needs a bullish engulfing with RSI above EMA(RSI); SELL mirrors it.
class EngulfingHmStrategy(Strategy):
    # BUY on bullish engulfing + RSI above EMA(RSI); SELL on bearish engulfing + RSI below EMA(RSI).
    name = "engulfing_hm"

    def __init__(self, symbol, timeframe="H4", count=300, rsi_period=9, ema_period=3,
                 atr_period=14, sl_mult=1.5, tp_mult=3.0, **extra):
        super().__init__(symbol, timeframe, count, rsi_period=rsi_period,
                         ema_period=ema_period, atr_period=atr_period, sl_mult=sl_mult,
                         tp_mult=tp_mult, **extra)
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def signal(self, df: pd.DataFrame) -> dict | None:
        if len(df) < max(self.rsi_period + self.ema_period + 3, self.atr_period + 2):
            return None
        i = len(df) - 2  # last CLOSED candle; the final row is still forming
        direction = int(engulfing(df).iloc[i])
        if direction == 0:
            return None
        closes = [float(x) for x in df["close"].iloc[: i + 1]]
        rsi_values = hilega_milega_rsi(closes, rsi_period=self.rsi_period)
        hm_line = ema(rsi_values, self.ema_period)
        rsi_value, hm_value = rsi_values[-1], hm_line[-1]
        if rsi_value is None or hm_value is None:
            return None
        if (direction > 0 and rsi_value <= hm_value) or (direction < 0 and rsi_value >= hm_value):
            return None
        highs = [float(x) for x in df["high"].iloc[: i + 1]]
        lows = [float(x) for x in df["low"].iloc[: i + 1]]
        atr_values = atr_series(highs, lows, closes, period=self.atr_period)
        atr_value = atr_values[-1]
        if atr_value is None or atr_value <= 0:
            return None
        return {"side": "BUY" if direction > 0 else "SELL",
                "rsi": float(rsi_value), "hm": float(hm_value), "atr": float(atr_value),
                "sl_dist": self.sl_mult * float(atr_value),
                "tp_dist": self.tp_mult * float(atr_value),
                "time": str(df.index[i])}


# Plugin registry export (framework_runner loads strategies by name through this).
STRATEGY = EngulfingHmStrategy
