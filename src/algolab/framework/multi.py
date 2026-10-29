"""Multi-timeframe strategy base.

The Bot fetches every timeframe in ``timeframes`` and calls ``signal_frames({tf: df})``.
Same rules as Strategy: signal_frames() is pure (no orders, no network, no file writes) and it only
looks at CLOSED bars. The last row of every frame is still forming, so ``closed()`` drops it.
"""
from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from algolab.framework.strategy import Strategy


def closed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the still-forming last row."""
    return df.iloc[:-1]


class MultiTimeframeStrategy(Strategy):
    """Subclasses set ``self.timeframes`` to (higher_tf, lower_tf) and implement signal_frames.
    The LAST name is the timeframe the Bot uses for bar identity (dedupe key)."""

    timeframes: tuple[str, ...] = ()

    def signal(self, df: pd.DataFrame) -> dict | None:
        raise TypeError(f"{type(self).__name__} needs several timeframes; the Bot calls signal_frames()")

    @abstractmethod
    def signal_frames(self, frames: dict[str, pd.DataFrame]) -> dict | None:
        """Return the trade dict for the newest closed lower-timeframe bar, or None."""
