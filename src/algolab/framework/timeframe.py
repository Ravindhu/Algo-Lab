"""Timeframe names used across the project. Maps to MetaTrader5 TIMEFRAME_* names in adapter.py."""
from enum import Enum


class TimeFrame(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def seconds(self) -> int:
        return {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                "H1": 3600, "H4": 14400, "D1": 86400}[self.value]


# Timeframe names used in the strategy configs (ocl_keylevel.py style) -> algolab names.
SOURCE_TF = {"1minute": "M1", "5minute": "M5", "15minute": "M15", "30minute": "M30",
             "60minute": "H1", "day": "D1"}


def normalize_tf(name) -> str:
    """Canonical algolab timeframe name ("M15", "H4", ...) for any name a config may use: "H4", "h4",
    "15minute", "day". Raises ValueError for anything else, so a typo in framework.json fails loudly."""
    key = str(name).strip()
    if key in SOURCE_TF:
        return SOURCE_TF[key]
    try:
        return TimeFrame(key.upper()).value
    except ValueError as e:
        raise ValueError(f"unknown timeframe {name!r}; use {sorted(SOURCE_TF)} or M1..D1") from e
