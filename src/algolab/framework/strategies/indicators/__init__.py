"""Shared indicators for framework strategies.

One home for indicator math so every strategy reads the SAME definition:

    from algolab.framework.strategies.indicators import hm_direction, hm_bullish, atr

Currently: the Hilega-Milega indicator (RSI with an EMA over it), used by both
ocl.py and engulfing_strategy.py.
"""
from algolab.framework.strategies.indicators.hm import (
    atr, hm_bullish, hm_direction, hm_ema_series, hm_rsi_series)
