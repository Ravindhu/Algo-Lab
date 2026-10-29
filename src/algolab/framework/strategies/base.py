"""Strategy base class. Subclass it, implement signal(), and hand it to a Bot.

Rules for every strategy:
  - signal() is pure: it reads the DataFrame and returns a dict or None. No orders, no network.
  - Only the last CLOSED bar counts. The last row of df is still forming; use df.iloc[-2].
  - A signal dict must contain "side" ("BUY" or "SELL"). For live trading it must also contain
    "lots", "sl" and "tp". The Bot refuses to place an order without them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from algolab.framework.symbol import ForexSymbol
from algolab.framework.timeframe import TimeFrame


class Strategy(ABC):
    name: str = "strategy"

    def __init__(self, symbol: ForexSymbol | str, timeframe: TimeFrame | str = TimeFrame.H4,
                 count: int = 300, **params):
        self.symbol = symbol if isinstance(symbol, ForexSymbol) else ForexSymbol(symbol)
        self.timeframe = TimeFrame(timeframe)
        self.count = count
        self.params = params

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> dict | None:
        """Return {"side": "BUY"|"SELL", ...} for the closed bar, or None for no trade."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.symbol.name}, {self.timeframe.value}, {self.params})"
