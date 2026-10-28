"""ForexSymbol: a ready-made wrapper around one instrument. It talks to the broker only through an
adapter (Mt5Adapter or any object with the same methods), so it can be tested without MT5."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algolab.framework.timeframe import TimeFrame


@dataclass(frozen=True)
class ForexSymbol:
    name: str

    @property
    def pip(self) -> float:
        """Price size of one pip: 0.01 for JPY pairs, 0.0001 for the rest."""
        return 0.01 if "JPY" in self.name[:6] else 0.0001

    async def candles(self, adapter, timeframe: TimeFrame = TimeFrame.H4,
                      count: int = 300) -> pd.DataFrame:
        """Oldest first. The LAST row is the bar still forming; the one before it is the last closed bar."""
        return await adapter.candles(self.name, TimeFrame(timeframe).value, count)

    def info(self, adapter) -> dict:
        return adapter.symbol_info(self.name)

    def tick(self, adapter) -> dict:
        return adapter.tick(self.name)
