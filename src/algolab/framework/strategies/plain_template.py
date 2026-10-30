# TEMPLATE strategy: the standard shape every new framework strategy follows.
#
# Copy this file, then change, in this order:
#   1. class name and `name`            (the config "type")
#   2. __init__ parameters               (one default per knob, passed to super)
#   3. signal(): the ENTRY rule          (closed bars only, pure, no orders)
#   4. the return dict                   (side + atr + sl_dist + tp_dist + time)
#   5. STRATEGY export at the bottom     (makes it discoverable)
#   6. one line in registry.py           (built-in list) and one config file, enabled=false
#
# Exit rules (trailing stop, breakeven) are NOT part of this template yet. The Bot's
# position manager handles exits. Only sl_dist and tp_dist are returned here.
#
# Entry rule of this sample ("plain logic" + HM):
#   BUY  when the Hilega-Milega RSI crosses UP through its own EMA on the last closed bar.
#   SELL when it crosses DOWN. Same rule as the discovery family `hm_cross`.
from __future__ import annotations

import numpy as np
import pandas as pd

from algolab.framework.strategies.base import Strategy
from algolab.framework.strategies.indicators import atr, hm_direction


class PlainTemplateStrategy(Strategy):
    name = "plain_template"

    def __init__(self, symbol, timeframe="M15", count=300,
                 rsi_period=9, ema_period=3, atr_period=14, sl_mult=1.5, tp_mult=3.0, **extra):
        super().__init__(symbol, timeframe, count, rsi_period=rsi_period, ema_period=ema_period,
                         atr_period=atr_period, sl_mult=sl_mult, tp_mult=tp_mult, **extra)
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def signal(self, df: pd.DataFrame) -> dict | None:
        # Enough history for the RSI, its EMA, the ATR, and one bar before the signal bar.
        need = max(self.rsi_period, self.ema_period, self.atr_period) * 3 + 2
        if len(df) < need:
            return None
        i = len(df) - 2                                   # last CLOSED bar; the final row is forming
        closes = [float(x) for x in df["close"].iloc[: i + 1]]
        highs = [float(x) for x in df["high"].iloc[: i + 1]]
        lows = [float(x) for x in df["low"].iloc[: i + 1]]

        # ENTRY RULE: direction flips into +1 (bullish) or -1 (bearish) on this bar.
        d = hm_direction(closes, self.rsi_period, self.ema_period)
        if d[-1] == 1 and d[-2] != 1:
            side = "BUY"
        elif d[-1] == -1 and d[-2] != -1:
            side = "SELL"
        else:
            return None

        atr_v = atr(highs, lows, closes, period=self.atr_period)[-1]
        if atr_v is None or not np.isfinite(atr_v) or atr_v <= 0:
            return None
        return {"side": side, "atr": float(atr_v),
                "sl_dist": self.sl_mult * float(atr_v),
                "tp_dist": self.tp_mult * float(atr_v),
                "time": str(df.index[i])}


# Plugin registry export. This line makes the strategy discoverable.
STRATEGY = PlainTemplateStrategy
