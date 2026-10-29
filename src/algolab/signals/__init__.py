"""Shared signal-rule exports (back-compat shim).

The implementations live in algolab/framework/strategies now:
  engulfing / fib_entry_level  -> framework/strategies/engulfing_strategy.py
  HM indicator (RSI/EMA gate)  -> framework/strategies/indicators (hm.py)
  scanner (scan_frame/scan_all)-> framework/scanner.py
"""
from algolab.framework.scanner import scan_all, scan_frame
from algolab.framework.strategies.engulfing_strategy import (DEFAULT_MIN_BODY_PCT,
                                                             FIB_ENTRY, engulfing,
                                                             fib_entry_level)
from algolab.framework.strategies.indicators import hm_direction as hm_direction_series

__all__ = ["engulfing", "fib_entry_level", "hm_direction_series",
           "DEFAULT_MIN_BODY_PCT", "FIB_ENTRY", "scan_all", "scan_frame"]
