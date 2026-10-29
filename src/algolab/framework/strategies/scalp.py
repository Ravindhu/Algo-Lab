"""ScalpTrader: short-timeframe trend-pullback scalper.

Idea: when the fast EMA is above the slow EMA (uptrend), wait for a bullish bar whose low dips to the
fast EMA and closes back above it, with RSI in a neutral band (not stretched). SELL mirrors this.
Stops and targets are ATR multiples, so they scale with volatility.

STATUS: not yet validated. Run a backtest before enabling it live (config/framework.json, "enabled").
"""
from __future__ import annotations

import pandas as pd

from algolab.framework.strategies.base import Strategy
from algolab.ml.features import rsi


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


class ScalpTrader(Strategy):
    name = "scalp_trend_pullback"

    def __init__(self, symbol, timeframe="M5", count=300, ema_fast=8, ema_slow=21,
                 rsi_n=14, atr_n=14, sl_mult=1.0, tp_mult=1.5, rsi_low=40.0, rsi_high=60.0,
                 **extra):
        super().__init__(symbol, timeframe, count, ema_fast=ema_fast, ema_slow=ema_slow,
                         rsi_n=rsi_n, atr_n=atr_n, sl_mult=sl_mult, tp_mult=tp_mult,
                         rsi_low=rsi_low, rsi_high=rsi_high, **extra)

    def signal(self, df: pd.DataFrame) -> dict | None:
        p = self.params
        if len(df) < max(p["ema_slow"], p["atr_n"], p["rsi_n"]) + 5:
            return None
        c = df["close"]
        ef = c.ewm(span=p["ema_fast"], adjust=False).mean()
        es = c.ewm(span=p["ema_slow"], adjust=False).mean()
        r = rsi(c, p["rsi_n"])
        a = _atr(df, p["atr_n"])

        i = -2                                       # last CLOSED bar
        bar = df.iloc[i]
        atr_v = float(a.iloc[i])
        rsi_v = float(r.iloc[i])
        if not (atr_v > 0 and pd.notna(rsi_v)):
            return None
        neutral = p["rsi_low"] <= rsi_v <= p["rsi_high"]
        fast = float(ef.iloc[i])

        side = None
        if (ef.iloc[i] > es.iloc[i] and neutral and bar["close"] > bar["open"]
                and bar["low"] <= fast < bar["close"]):
            side = "BUY"
        elif (ef.iloc[i] < es.iloc[i] and neutral and bar["close"] < bar["open"]
                and bar["high"] >= fast > bar["close"]):
            side = "SELL"
        if side is None:
            return None
        return {"side": side, "rsi": round(rsi_v, 2), "atr": atr_v,
                "sl_dist": p["sl_mult"] * atr_v, "tp_dist": p["tp_mult"] * atr_v,
                "time": str(df.index[i])}


# Plugin registry export (framework_runner loads strategies by name through this).
STRATEGY = ScalpTrader
