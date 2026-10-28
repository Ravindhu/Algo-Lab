"""Ready-made strategies. EngulfingRsiStrategy wraps the project's proven scanner rule, so the
Bot and the live pipeline use the same signal logic."""
from __future__ import annotations

import pandas as pd

from algolab.framework.strategy import Strategy
from algolab.scanner.scanner import scan_frame


class EngulfingRsiStrategy(Strategy):
    """BUY on a bullish engulfing with RSI below oversold; SELL on a bearish engulfing with RSI above overbought.
    Params: rsi_n, oversold, overbought (defaults match config/strategy.json)."""
    name = "engulfing_rsi"

    def __init__(self, symbol, timeframe="H4", count=300, rsi_n=14, oversold=30.0,
                 overbought=70.0, **extra):
        super().__init__(symbol, timeframe, count, rsi_n=rsi_n, oversold=oversold,
                         overbought=overbought, **extra)
        self._cfg = {"rsi_n": rsi_n, "oversold": oversold, "overbought": overbought}

    def signal(self, df: pd.DataFrame) -> dict | None:
        hit = scan_frame(self.symbol.name, df, self._cfg)
        if not hit:
            return None
        return {"side": hit["side"], "rsi": hit.get("rsi"), "time": str(hit.get("time"))}
