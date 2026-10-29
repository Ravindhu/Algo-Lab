"""Bot framework and shared building blocks (Bot, symbol, timeframe, prop rules).

Strategies live in the strategies/ subpackage:

    from algolab.framework import Bot, ForexSymbol
    from algolab.framework.strategies import EngulfingHmStrategy

    bot = Bot(adapter=ad, dry_run=True).add(EngulfingHmStrategy(ForexSymbol("EURUSD")))
    await bot.run_once()
"""
from algolab.framework.bot import Bot
from algolab.framework.strategies import (
    EngulfingHmStrategy, MultiTimeframeStrategy, OclKeylevel, ScalpTrader, Strategy)
from algolab.framework.symbol import ForexSymbol
from algolab.framework.timeframe import TimeFrame

__all__ = ["Bot", "Strategy", "MultiTimeframeStrategy", "ForexSymbol", "TimeFrame",
           "ScalpTrader", "EngulfingHmStrategy", "OclKeylevel"]
