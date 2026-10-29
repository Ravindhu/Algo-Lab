"""Framework tests: no broker needed. A fake adapter stands in for Mt5Adapter."""
import asyncio

import numpy as np
import pandas as pd

from algolab.framework import Bot, ForexSymbol, Strategy, TimeFrame


class FakeAdapter:
    # adapter surface added for the live safety checks (margin, broker P&L, reconciliation)
    def margin_for(self, symbol, side, lots):
        return 0.0

    def free_margin(self):
        return 1e9

    def realized_pnl(self, ticket):
        return getattr(self, "realized", {}).get(int(ticket))

    def position_tickets(self, magic):
        return {int(p["ticket"]) for p in self.positions}

    def __init__(self, open_positions=0, df=None):
        self._open = open_positions
        self._df = df if df is not None else frame()
        self.orders = []

    async def candles(self, symbol, tf_name, count):
        return self._df

    async def open_positions(self):
        return self._open

    def market_order(self, symbol, side, lots, sl, tp, magic, deviation):
        self.orders.append((symbol, side, lots))
        return {"ticket": 1, "fill_price": 1.1}


def frame(n=50):
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    close = np.linspace(1.10, 1.12, n)
    return pd.DataFrame({"open": close, "high": close + 0.001,
                         "low": close - 0.001, "close": close}, index=idx)


class AlwaysBuy(Strategy):
    name = "always_buy"

    def signal(self, df):
        return {"side": "BUY", "lots": 0.1, "sl": 1.09, "tp": 1.13}


class Broken(Strategy):
    name = "broken"

    def signal(self, df):
        raise ValueError("boom")


def test_timeframe_and_symbol_helpers():
    assert TimeFrame.H4.seconds == 14400
    assert ForexSymbol("EURUSD").pip == 0.0001
    assert ForexSymbol("USDJPY").pip == 0.01
    assert ForexSymbol("USDJPYm").pip == 0.01


def test_dry_run_reports_but_never_orders():
    ad = FakeAdapter()
    bot = Bot(adapter=ad, dry_run=True).add(AlwaysBuy(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert res[0]["side"] == "BUY" and res[0]["executed"] is False
    assert ad.orders == []


def test_live_places_order_once_per_bar():
    ad = FakeAdapter(open_positions=0)
    bot = Bot(adapter=ad, dry_run=False).add(AlwaysBuy(ForexSymbol("EURUSD")))
    asyncio.run(bot.run_once())
    asyncio.run(bot.run_once())            # same bar again: must not trade twice
    assert ad.orders == [("EURUSD", "BUY", 0.1)]


def test_unknown_positions_fail_closed():
    ad = FakeAdapter(open_positions=None)
    bot = Bot(adapter=ad, dry_run=False).add(AlwaysBuy(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert res[0]["executed"] is False and ad.orders == []


def test_max_open_blocks_orders():
    ad = FakeAdapter(open_positions=3)
    bot = Bot(adapter=ad, dry_run=False, max_open=3).add(AlwaysBuy(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert res[0]["executed"] is False and ad.orders == []


def test_broken_strategy_does_not_stop_others():
    bot = Bot(adapter=FakeAdapter(), dry_run=True)
    bot.add(Broken("EURUSD")).add(AlwaysBuy(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert any("error" in r for r in res)
    assert any(r.get("side") == "BUY" for r in res)
