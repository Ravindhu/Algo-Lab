"""Strategy and Bot framework, plus ready-made classes (TimeFrame, ForexSymbol, strategies).

    from algolab.framework import Bot, ForexSymbol, TimeFrame, EngulfingRsiStrategy
    bot = Bot(adapter=ad, dry_run=True).add(EngulfingRsiStrategy(ForexSymbol("EURUSD"), TimeFrame.H4))
    await bot.run_once()
"""
from algolab.framework.bot import Bot
from algolab.framework.scalp import ScalpTrader
from algolab.framework.strategies import EngulfingRsiStrategy
from algolab.framework.strategy import Strategy
from algolab.framework.symbol import ForexSymbol
from algolab.framework.timeframe import TimeFrame

__all__ = ["Bot", "Strategy", "ForexSymbol", "TimeFrame", "EngulfingRsiStrategy", "ScalpTrader"]
