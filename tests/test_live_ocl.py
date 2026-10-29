"""Live wiring for OCL: sizer (currency, broker stop, prop limits), position manager (close-based exits,
broker closures, HTF floor, retry on failed close, flatten) and the Bot path with the prop gate.
A fake adapter stands in for MT5. No broker, no network, no alerts (alerts.send is replaced)."""
import asyncio

import pandas as pd
import pytest

from algolab.framework.prop import FAILED, PropGuard, PropRules, PropState
from algolab.live import framework_runner as FR
from algolab.live import positions
from algolab.live.adapter import AdapterError
from algolab.ops import alerts
from test_ocl import FILL, FORMING, HTF_BREAK, LTF_TOUCH, frame, htf_frame, ltf_frame


@pytest.fixture(autouse=True)
def no_alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "send", lambda text, *a, **k: sent.append(text) or True)
    return sent


class FakeAdapter:
    def __init__(self, bid=1.0919, ask=1.0921):
        self.ticks = {"EURUSD": {"bid": bid, "ask": ask}}
        self.infos = {"EURUSD": {"contract_size": 100000.0, "volume_min": 0.01, "volume_step": 0.01,
                                 "digits": 5}}
        self.candles_by_tf = {}
        self.positions = []
        self.closed, self.orders = [], []
        self.order_ticket = 77
        self.fail_candles = False
        self.fail_close = 0            # number of close calls that fail before one succeeds

    def tick(self, symbol):
        if symbol not in self.ticks:
            raise AdapterError(f"tick({symbol}) failed")
        return self.ticks[symbol]

    def symbol_info(self, symbol):
        return self.infos[symbol]

    def account_currency(self):
        return "USD"

    def positions_by_magic(self, magic):
        return list(self.positions)

    async def candles(self, symbol, tf_name, count):
        if self.fail_candles:
            raise AdapterError("candles failed")
        return self.candles_by_tf[tf_name]

    async def open_positions(self):
        return len(self.positions)

    def close_position(self, ticket, symbol, side, volume, magic, deviation):
        if self.fail_close > 0:
            self.fail_close -= 1
            raise AdapterError("close rejected")
        self.closed.append((ticket, symbol))
        return {"ticket": ticket, "fill_price": self.tick(symbol)["bid"]}

    def market_order(self, symbol, side, lots, sl, tp, magic, deviation):
        self.orders.append((symbol, side, lots, sl, tp))
        return {"ticket": self.order_ticket, "fill_price": self.ticks[symbol]["ask"]}


CFG = {"timeframe": "M5", "max_open": 3, "magic": 20262002, "deviation_points": 20,
       "risk_per_trade": 0.005, "max_lots": 0.5, "max_spread_atr": 0.25, "enabled": True,
       "strategies": [{"type": "ocl", "params": {"hm_enabled": False, "ltf_pattern_enabled": False}}]}
PROTECT = {"atr_trail_enabled": True, "atr_multiplier": 2.0, "breakeven_rr": 1.5, "round_trip_cost_pct": 0.2}


def rec(**over):
    r = {"ticket": 1, "symbol": "EURUSD", "side": "BUY", "fill_price": 1.092, "lots": 0.05,
         "sl": 1.074, "tp": 1.134, "trigger": 1.082, "htf_sl": 1.078, "protect": PROTECT,
         "atr_period": 14, "bar": "2026-01-05 01:50:00"}
    r.update(over)
    return r


def tracked_one(**over):
    return {"1": positions.new_record(rec(**over), strategy="ocl_keylevel", ltf="M5", htf="M15")}


def open_position(ad):
    ad.positions = [{"ticket": 1, "symbol": "EURUSD", "side": 1, "price_open": 1.092,
                     "sl": 1.074, "tp": 1.134, "volume": 0.05}]


def ltf_closed_then_forming(*closed_bars):
    """LTF frame from 00:00. 23 FILL bars put the first given bar at 01:55 (the entry bar), then a forming row."""
    return frame([FILL] * 23 + list(closed_bars) + [FORMING], start="2026-01-05 00:00", freq="5min")


ENTRY_BAR = (1.092, 1.093, 1.091, 1.092)    # 01:55, the entry bar (index 23)


def htf_closed(close):
    return frame([(1.09, 1.10, 1.08, close), FORMING], start="2026-01-05 01:45", freq="15min")


# ---------- sizer ----------

def test_structural_stop_is_the_level_and_target_is_rr_times_risk_from_live_price():
    ad = FakeAdapter()
    plan = FR.make_sizer(ad, 10000, CFG)({"symbol": "EURUSD", "side": "BUY", "sl_price": 1.0740,
                                          "tp_dist": 0.042, "target_rr": 3.0, "htf_sl": 1.078,
                                          "atr": 0.0025, "max_spread_atr": 0.25})
    assert plan["sl"] == pytest.approx(1.074)
    assert plan["tp"] == pytest.approx(1.0921 + 3.0 * (1.0921 - 1.078), abs=1e-5)
    assert plan["lots"] > 0


def test_dynamic_exit_sends_no_target():
    plan = FR.make_sizer(FakeAdapter(), 10000, CFG)({"symbol": "EURUSD", "side": "BUY", "sl_dist": 0.02,
                                                     "tp_dist": None, "atr": 0.0025})
    assert plan["tp"] == 0.0


def test_jpy_pair_risk_is_converted_to_account_currency():
    ad = FakeAdapter()
    ad.ticks["USDJPY"] = {"bid": 149.99, "ask": 150.01}
    ad.infos["USDJPY"] = dict(ad.infos["EURUSD"], digits=3)
    plan = FR.make_sizer(ad, 10000, CFG)({"symbol": "USDJPY", "side": "BUY", "sl_dist": 0.5,
                                          "tp_dist": 1.5, "atr": 0.1})
    assert plan["lots"] == pytest.approx(0.15, abs=0.01)     # 50 USD risk / 333 USD per lot


def test_missing_conversion_pair_skips_with_a_reason():
    ad = FakeAdapter()
    ad.ticks["EURGBX"] = {"bid": 1.0919, "ask": 1.0920}
    ad.infos["EURGBX"] = dict(ad.infos["EURUSD"])
    plan = FR.make_sizer(ad, 10000, CFG)({"symbol": "EURGBX", "side": "BUY", "sl_dist": 0.001,
                                          "tp_dist": 0.003, "atr": 0.0025})
    assert plan == {"skip": "no USD conversion pair"}


def test_wide_spread_skips_with_a_reason():
    ad = FakeAdapter(bid=1.0919, ask=1.0929)
    plan = FR.make_sizer(ad, 10000, CFG)({"symbol": "EURUSD", "side": "BUY", "sl_price": 1.078,
                                          "atr": 0.0025, "max_spread_atr": 0.25, "htf_sl": 1.078,
                                          "tp_dist": 0.04, "target_rr": 3.0})
    assert plan == {"skip": "spread too wide for ATR"}


def test_prop_room_skips_the_order_with_the_prop_reason():
    guard = PropGuard(PropRules(starting_balance=10000), state=PropState(profile="t", day="2026-01-05",
                                                                         day_start_equity=10000.0))
    plan = FR.make_sizer(FakeAdapter(), 9450, CFG, "USD", guard, pd.Timestamp("2026-01-05 10:00"),
                         lambda: 0)({"symbol": "EURUSD", "side": "BUY", "sl_price": 1.074, "htf_sl": 1.078,
                                     "tp_dist": 0.04, "target_rr": 3.0, "atr": 0.0025,
                                     "max_spread_atr": 0.25})
    assert "skip" in plan and "prop" in plan["skip"]


def test_prop_caps_lots_by_risk_percent():
    guard = PropGuard(PropRules(starting_balance=10000, max_risk_per_trade_pct=1.0), profile="t")
    guard.update(pd.Timestamp("2026-01-05 10:00"), 10000)
    plan = FR.make_sizer(FakeAdapter(), 10000, CFG, "USD", guard, pd.Timestamp("2026-01-05 10:00"),
                         lambda: 0)({"symbol": "EURUSD", "side": "BUY", "sl_price": 1.074, "htf_sl": 1.078,
                                     "tp_dist": 0.04, "target_rr": 3.0, "atr": 0.0025,
                                     "max_spread_atr": 0.25})
    # Entry 1.0921, broker stop 1.074: 0.0181 per unit, 1810 per lot. 1% of 10000 = 100 -> at most 0.05 lots.
    assert 0 < plan["lots"] <= 0.05 + 1e-9


# ---------- position manager ----------

def test_new_record_carries_the_trade_levels():
    t = tracked_one()["1"]
    assert t["trade"]["stop"] == pytest.approx(1.078)           # LTF stop starts at the HTF floor
    assert t["trade"]["broker_sl"] == pytest.approx(1.074)
    assert t["last_bar"] == "2026-01-05 01:55:00"


def test_quiet_bars_keep_the_position_and_move_no_stops():
    ad = FakeAdapter()
    open_position(ad)
    ad.candles_by_tf = {"M5": ltf_closed_then_forming(ENTRY_BAR, (1.093, 1.094, 1.092, 1.093)),
                        "M15": htf_closed(1.09)}
    tracked = tracked_one()
    assert asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20)) == []
    assert ad.closed == [] and "1" in tracked


def test_close_below_the_ltf_stop_closes_at_market_and_reports_sl():
    ad = FakeAdapter()
    open_position(ad)
    ad.candles_by_tf = {"M5": ltf_closed_then_forming(ENTRY_BAR, (1.090, 1.091, 1.070, 1.077)),
                        "M15": htf_closed(1.09)}
    tracked = tracked_one()
    events = asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20))
    assert ad.closed == [(1, "EURUSD")]
    assert events[0]["reason"] == "SL" and events[0]["price"] == pytest.approx(1.077)
    assert tracked == {}


def test_broker_closed_position_is_classified_as_disaster_without_an_order():
    ad = FakeAdapter()                                          # broker no longer holds the position
    ad.candles_by_tf = {"M5": ltf_closed_then_forming(ENTRY_BAR, (1.090, 1.091, 1.060, 1.085)),
                        "M15": htf_closed(1.09)}
    tracked = tracked_one()
    events = asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20))
    assert events[0]["reason"] == "DISASTER" and ad.closed == [] and tracked == {}


def test_htf_close_below_the_floor_closes_with_htf_reason():
    ad = FakeAdapter()
    open_position(ad)
    ad.candles_by_tf = {"M5": ltf_closed_then_forming(ENTRY_BAR, (1.092, 1.093, 1.091, 1.092)),
                        "M15": htf_closed(1.075)}
    tracked = tracked_one()
    events = asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20))
    assert events[0]["reason"] == "HTF" and tracked == {}


def test_candle_error_keeps_tracking():
    ad = FakeAdapter()
    open_position(ad)
    ad.fail_candles = True
    tracked = tracked_one()
    assert asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20)) == []
    assert "1" in tracked


def test_failed_close_is_retried_on_the_same_bar_next_cycle():
    ad = FakeAdapter()
    open_position(ad)
    ad.candles_by_tf = {"M5": ltf_closed_then_forming(ENTRY_BAR, (1.090, 1.091, 1.070, 1.077)),
                        "M15": htf_closed(1.09)}
    tracked = tracked_one()
    ad.fail_close = 1
    assert asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20)) == []
    assert "1" in tracked and tracked["1"]["last_bar"] == "2026-01-05 01:55:00"   # not advanced
    events = asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20))
    assert events[0]["reason"] == "SL"


def test_flatten_closes_live_positions_and_drops_broker_closed_ones():
    ad = FakeAdapter()
    open_position(ad)
    tracked = tracked_one()
    tracked["2"] = dict(tracked["1"], ticket=2)                 # not held by the broker any more
    events = positions.flatten(ad, tracked, magic=20262002, deviation=20, reason="PROP")
    reasons = {e["ticket"]: e["reason"] for e in events}
    assert reasons == {1: "PROP", 2: "BROKER_CLOSE"} and tracked == {}


# ---------- Bot path ----------

def htf_ltf_frames(ad):
    ad.candles_by_tf = {"M15": htf_frame(HTF_BREAK), "M5": ltf_frame(LTF_TOUCH)}


def test_bot_fires_places_the_order_and_tracks_the_fill():
    ad = FakeAdapter()
    htf_ltf_frames(ad)
    tracked = {}
    bot, objs = FR.build_bot(ad, {"dry_run": False}, ["EURUSD"], 10000, CFG, "USD", tracked)
    results = asyncio.run(bot.run_once())
    fired = [r for r in results if r.get("executed")]
    assert len(fired) == 1 and ad.orders and ad.orders[0][1] == "BUY"
    assert ad.orders[0][3] == pytest.approx(fired[0]["sl"])
    obj = objs[("ocl_keylevel", "EURUSD")]
    assert obj.st.in_trade is True
    assert tracked["77"]["trade"]["htf_sl"] == pytest.approx(1.078)


def test_bot_dry_run_places_nothing():
    ad = FakeAdapter()
    htf_ltf_frames(ad)
    tracked = {}
    bot, objs = FR.build_bot(ad, {"dry_run": True}, ["EURUSD"], 10000, CFG, "USD", tracked)
    asyncio.run(bot.run_once())
    assert ad.orders == [] and tracked == {}
    assert objs[("ocl_keylevel", "EURUSD")].st.in_trade is False


def test_failed_prop_account_blocks_the_bot_with_a_reason():
    ad = FakeAdapter()
    htf_ltf_frames(ad)
    guard = PropGuard(PropRules(starting_balance=10000),
                      state=PropState(profile="t", status=FAILED, fail_reason="x"))
    bot, _ = FR.build_bot(ad, {"dry_run": False}, ["EURUSD"], 10000, CFG, "USD", {}, guard,
                          pd.Timestamp("2026-01-05 10:00"))
    results = asyncio.run(bot.run_once())
    assert ad.orders == []
    assert results and results[0]["why"].startswith("prop account FAILED")


def test_one_position_at_a_time_when_prop_allows_one():
    ad = FakeAdapter()
    htf_ltf_frames(ad)
    tracked = {"9": {"ticket": 9}}                               # a framework position already open
    guard = PropGuard(PropRules(starting_balance=10000, max_open_positions=1), profile="t")
    bot, _ = FR.build_bot(ad, {"dry_run": False}, ["EURUSD"], 10000, CFG, "USD", tracked, guard,
                          pd.Timestamp("2026-01-05 10:00"))
    asyncio.run(bot.run_once())
    assert ad.orders == []
