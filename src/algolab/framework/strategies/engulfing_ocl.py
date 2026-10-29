"""Engulfing HTF setup with the OCL lower-timeframe entry model.

HTF (closed bars): a BUY engulfing on the newest closed bar, with both candle bodies above min_body_pct of their
range and a Hilega-Milega confirmation (RSI above EMA(RSI)) on that bar. The setup arms one level: the fib level
inside the engulfing candle (fib_pct, default 88.6%). The floor is the engulfing candle's low.

LTF (closed bars): the OCL entry model is used unchanged. Touch of the level, red-red-green on the last three
LTF bars, the LTF Hilega-Milega gate with the oversold cap, the retry ladder, and the protection rules
(breakeven, ATR trail, disaster stop, HTF floor). Only the HTF setup differs from OclKeylevel.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from algolab.framework.strategies.ocl import OclKeylevel
from algolab.framework.strategies.engulfing_strategy import engulfing, fib_entry_level
from algolab.framework.strategies.indicators import hm_direction


class EngulfingOcl(OclKeylevel):
    name = "engulfing_ocl"

    def _htf_setup(self, htf: pd.DataFrame) -> dict[str, Any] | None:
        p = self.p
        if len(htf) < 2:
            return None
        direction = int(engulfing(htf, p.min_body_pct).iloc[-1])
        if direction <= 0:                                  # BUY side only; the inverse wrapper gives SELL
            return None
        closes = [float(x) for x in htf["close"]]
        if p.hm_enabled and int(hm_direction(closes, p.rsi_period, p.ema_period)[-1]) != 1:
            return None
        high, low = float(htf["high"].iloc[-1]), float(htf["low"].iloc[-1])
        level = fib_entry_level(high, low, 1, p.fib_pct)
        return {"htf_sl": low, "aggressive": level, "conservative": level,
                "confirm_bar": "ENG", "key": str(htf.index[-1])}


STRATEGY = EngulfingOcl
