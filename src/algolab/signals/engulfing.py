"""Scanner helpers (moved to the framework) + back-compat shim.

The scanner lives in algolab/framework/scanner.py now. The signal math it uses
(engulfing, Hilega-Milega, fib entry) lives in algolab/framework/strategies/
indicators and algolab/framework/strategies/engulfing_strategy.py.

This module keeps re-exporting the old names so imports like
`from algolab.signals.engulfing import scan_all` keep working, but the ONE
implementation is in the framework.
"""
from algolab.framework.scanner import scan_all, scan_frame
from algolab.framework.strategies.engulfing_strategy import (DEFAULT_MIN_BODY_PCT,
                                                             FIB_ENTRY, engulfing,
                                                             fib_entry_level)
from algolab.framework.strategies.indicators import hm_direction as hm_direction_series

__all__ = ["engulfing", "fib_entry_level", "hm_direction_series", "hm_confirmed",
           "DEFAULT_MIN_BODY_PCT", "FIB_ENTRY", "scan_all", "scan_frame"]
