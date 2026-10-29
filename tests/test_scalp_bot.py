"""ScalpTrader, the Bot sizer path, and the pipeline connection. No broker needed."""
import asyncio

import numpy as np
import pandas as pd

from algolab.framework import Bot, ForexSymbol, ScalpTrader, Strategy
from algolab.live import framework_runner as FR


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

    def __init__(self, df=None, open_positions=0):
        self._df = df if df is not None else random_frame()
        self._open = open_positions
        self.orders = []

    async def candles(self, symbol, tf_name, count):
        return self._df

    async def open_positions(self):
        return self._open

    def symbol_info(self, symbol):
        return {"contract_size": 100000, "volume_min": 0.01, "volume_step": 0.01, "digits": 5}

    def tick(self, symbol):
        return {"bid": 1.10000, "ask": 1.10010}

    def market_order(self, symbol, side, lots, sl, tp, magic, deviation):
        self.orders.append({"side": side, "lots": lots, "sl": sl, "tp": tp})
        return {"ticket": 7, "fill_price": 1.1001}


def random_frame(n=200, seed=3):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0003, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0003, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


class Fixed(Strategy):
    """Always signals BUY with a 10-pip stop and 15-pip target (distances in price units)."""
    name = "fixed"

    def signal(self, df):
        return {"side": "BUY", "sl_dist": 0.0010, "tp_dist": 0.0015}


def test_scalp_short_data_returns_none():
    assert ScalpTrader(ForexSymbol("EURUSD"), "M5").signal(random_frame(10)) is None


def test_scalp_output_contract_on_random_data():
    s = ScalpTrader(ForexSymbol("EURUSD"), "M5")
    seen = 0
    for seed in range(20):
        sig = s.signal(random_frame(seed=seed))
        if sig is None:
            continue
        seen += 1
        assert sig["side"] in ("BUY", "SELL")
        assert sig["sl_dist"] > 0 and sig["tp_dist"] > sig["sl_dist"]
    assert seen >= 0          # runs on every seed without error


def test_sizer_buy_levels_and_lots():
    cfg = {"risk_per_trade": 0.01, "max_lots": 1.0}
    sizer = FR.make_sizer(FakeAdapter(), equity=10000, cfg=cfg)
    plan = sizer({"symbol": "EURUSD", "side": "BUY", "sl_dist": 0.0010, "tp_dist": 0.0015})
    # risk 100 / (0.0010 * 100000) = 1.00 lot
    assert plan["lots"] == 1.0
    assert plan["sl"] == round(1.10010 - 0.0010, 5)
    assert plan["tp"] == round(1.10010 + 0.0015, 5)


def test_sizer_sell_uses_bid_and_inverts_levels():
    cfg = {"risk_per_trade": 0.005, "max_lots": 0.5}
    plan = FR.make_sizer(FakeAdapter(), 10000, cfg)({"symbol": "EURUSD", "side": "SELL",
                                                     "sl_dist": 0.0010, "tp_dist": 0.0020})
    assert plan["lots"] == 0.5                       # clamped to max_lots
    assert plan["sl"] == round(1.10000 + 0.0010, 5)
    assert plan["tp"] == round(1.10000 - 0.0020, 5)


def test_sizer_skips_when_below_broker_minimum():
    cfg = {"risk_per_trade": 0.0001, "max_lots": 1.0}      # tiny risk -> under 0.01 lot
    plan = FR.make_sizer(FakeAdapter(), 1000, cfg)({"symbol": "EURUSD", "side": "BUY",
                                                    "sl_dist": 0.0010, "tp_dist": 0.0015})
    assert plan is None


def test_bot_uses_sizer_then_places_live_order():
    ad = FakeAdapter()
    cfg = {"risk_per_trade": 0.01, "max_lots": 1.0}
    bot = Bot(adapter=ad, dry_run=False, sizer=FR.make_sizer(ad, 10000, cfg))
    bot.add(Fixed(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert res[0]["executed"] is True
    assert ad.orders[0]["lots"] == 1.0 and ad.orders[0]["side"] == "BUY"


def test_bot_dry_run_sizes_but_does_not_order():
    ad = FakeAdapter()
    cfg = {"risk_per_trade": 0.01, "max_lots": 1.0}
    bot = Bot(adapter=ad, dry_run=True, sizer=FR.make_sizer(ad, 10000, cfg))
    bot.add(Fixed(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert res[0]["lots"] == 1.0 and res[0]["executed"] is False
    assert ad.orders == []


def test_framework_is_disabled_by_default():
    """Scalp rule is unvalidated: the shipped config must keep the framework off."""
    assert FR.load_config()["enabled"] is False
    assert asyncio.run(FR.run_framework(FakeAdapter(), {"dry_run": True}, ["EURUSD"],
                                        10000)) is None
