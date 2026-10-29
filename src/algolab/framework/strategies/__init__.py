"""Strategy implementations, grouped under one package.

    from algolab.framework.strategies import (
        EngulfingHmStrategy, OclKeylevel, ScalpTrader, Strategy, MultiTimeframeStrategy)

Each module holds one strategy (or its base class):
    base.py               Strategy (single-timeframe base)
    multi.py              MultiTimeframeStrategy (HTF + LTF base)
    scalp.py              ScalpTrader        -> "scalp" in config/framework.json
    engulfing_strategy.py EngulfingHmStrategy -> "engulfing" in config/framework.json
    ocl.py                OclKeylevel        -> "ocl" in config/framework.json
"""
from algolab.framework.strategies.base import Strategy
from algolab.framework.strategies.engulfing_strategy import EngulfingHmStrategy
from algolab.framework.strategies.multi import MultiTimeframeStrategy
from algolab.framework.strategies.ocl import OclKeylevel, OclParams, SetupState
from algolab.framework.strategies.ocl_sell_fail import OclSellFailBottom, OclSellFailTop
from algolab.framework.strategies.scalp import ScalpTrader

__all__ = ["Strategy", "MultiTimeframeStrategy", "ScalpTrader",
           "EngulfingHmStrategy", "OclKeylevel", "OclParams", "SetupState",
           "OclSellFailTop", "OclSellFailBottom"]
