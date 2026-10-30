"""Parity: discovery ocl_keylevel must reproduce the LIVE OCL backtest, trade for trade.

Live reference: framework/ocl_backtest.py run_ocl_backtest (buy) and run_inverse_backtest with OclInv (sell).
Both drive the same strategy object and protect.Trade that the live bot uses.
"""
import numpy as np
import pandas as pd
import pytest

# Import the framework first: importing backtest.engine first hits a circular import in the live package.
from algolab.framework.ocl_backtest import run_inverse_backtest, run_ocl_backtest  # noqa: I001
from algolab.backtest.engine import Costs
from algolab.discovery.strategies import ocl as disc
from algolab.framework.strategies.ocl import OclKeylevel
from algolab.framework.strategies.ocl_inverse import OclInv

LIVE_PARAMS = dict(rsi_period=9, ema_period=3, oversold=30.0, hm_enabled=True, ltf_pattern_enabled=True,
                   allow_c4_confirmation=True, trigger_level_mode="CONSERVATIVE", touch_buffer=0.0,
                   target_rr=3.0, max_ltf_retries=3, atr_period=14, atr_multiplier=2.0,
                   atr_trail_enabled=True, breakeven_rr=1.5, round_trip_cost_pct=0.2,
                   round_trip_cost_pips=None, disaster_stop_atr=1.0, max_spread_atr=0.25, lookback=120,
                   exit_mode="AUTO")

# Small spread: the synthetic LTF bars are small, and the live spread filter would block almost every entry
# at a realistic spread. Both sides use the same value.
COSTS = Costs(spread=0.00003)


def _reflect(bar, C):
    o, h, l, c = bar
    return (2 * C - o, 2 * C - l, 2 * C - h, 2 * C - c)


def _buy_trigger_path(bar, dip, rng, per):
    """LTF path inside the trigger HTF bar: red bars down to the dip, one green rebound bar, then up to close."""
    o, h, l, c = bar
    rebound = dip + 0.00005                # small green step: RSI stays under the oversold cap
    # 9 red bars down (drives RSI under the oversold cap), one green rebound, then 6 bars up to the close
    path = np.concatenate([np.linspace(o, dip, 10)[1:], [rebound], np.linspace(rebound, c, 7)[1:]])
    return np.clip(path + rng.normal(0, 0.00003, per), l, h)


def _market_ocl(n_htf, seed, per=16, p_pat=0.3):
    """HTF bars with OCL setups (red, red, green break) followed by a trigger bar. The setup arms at the
    close of the green bar, so the LTF trigger path sits in the trigger bar. Sell setups are the buy setups
    reflected about the starting price (the same mirror the live inverse wrapper uses)."""
    rng = np.random.default_rng(seed)
    price = 1.10
    bars, marks = [], []
    while len(bars) < n_htf:
        if rng.random() < p_pat:
            p0 = price
            c1 = p0 - rng.uniform(0.002, 0.004)
            o2 = c1 + 0.0001
            c2 = o2 - rng.uniform(0.002, 0.004)
            o3 = c2 - 0.0001
            c3 = o2 + rng.uniform(0.0005, 0.0015)
            bar1 = (p0, p0 + 0.0002, c1 - 0.0002, c1)
            bar2 = (o2, o2 + 0.0002, c2 - 0.0002, c2)
            bar3 = (o3, max(o3, c3) + 0.0002, min(o3, c3) - 0.0002, c3)
            # With C4 confirmation the live setup is read from the last four closed HTF bars, so the
            # green C3 is recognised one bar after it closes. bar4 is that free bar (C4); the LTF trigger
            # sits in bar5, the first bar the setup can arm for.
            bar4 = (c3, c3 + 0.0001, c3 - 0.0001, c3 + 0.00005)
            bar5 = (c3 + 0.00005, max(c3, c3 + 0.0001) + 0.0002, o3 - 0.0003, c3 + 0.0001)
            dip = o3 - 0.0001
            side = "buy" if rng.random() < 0.5 else "sell"
            if side == "buy":
                pat = [bar1, bar2, bar3, bar4, bar5]
                trig = ("buy", dip)
            else:
                pat = [_reflect(b, p0) for b in (bar1, bar2, bar3, bar4, bar5)]
                trig = ("sell", 2 * p0 - dip, p0)
            bars += pat
            marks += [None, None, None, None, trig]
            price = pat[-1][3]
        else:
            o = price
            cl = o + rng.normal(0, 0.003)
            bars.append((o, max(o, cl) + 0.0002, min(o, cl) - 0.0002, cl))
            marks.append(None)
            price = cl
    bars, marks = bars[:n_htf], marks[:n_htf]
    htf = pd.DataFrame(bars, columns=["open", "high", "low", "close"],
                       index=pd.date_range("2024-01-01", periods=n_htf, freq="4h"))
    rows = []
    for k, bar in enumerate(bars):
        mk = marks[k]
        if mk is None:
            path = np.linspace(bar[0], bar[3], per + 1)[1:]
        elif mk[0] == "buy":
            path = _buy_trigger_path(bar, mk[1], rng, per)
        else:                                     # sell: buy path on the reflected bar, reflected back
            C = mk[2]
            path = 2 * C - _buy_trigger_path(_reflect(bar, C), 2 * C - mk[1], rng, per)
        prev = bar[0]
        for cl in path:
            hi = min(bar[1], max(prev, cl) + abs(rng.normal(0, 0.00003)))
            lo = max(bar[2], min(prev, cl) - abs(rng.normal(0, 0.00003)))
            rows.append((prev, hi, lo, cl))
            prev = cl
    ltf = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                       index=pd.date_range("2024-01-01", periods=n_htf * per, freq="15min"))
    return htf, ltf


@pytest.fixture(scope="module")
def frames():
    return _market_ocl(1500, seed=21)


def _key(df):
    return [(r.entry_time, r.exit_time, r.reason) for r in df.itertuples()]


def test_buy_side_matches_live_ocl(frames):
    htf, ltf = frames
    live = run_ocl_backtest(OclKeylevel("EURUSD", htf="H4", ltf="M15", **LIVE_PARAMS), htf, ltf, COSTS)
    ours = disc.run(htf, ltf, disc.OclParams(), spread=COSTS.spread,
                    pip=OclKeylevel("EURUSD").symbol.pip, side="BUY")
    assert len(live) >= 2, "market produced too few live trades to be a real test"
    assert _key(ours) == _key(live)
    np.testing.assert_allclose(ours["entry"].to_numpy(), live["entry"].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ours["exit"].to_numpy(), live["exit"].to_numpy(), rtol=0, atol=1e-12)


def test_sell_side_matches_live_ocl_inverse(frames):
    htf, ltf = frames
    live = run_inverse_backtest(OclInv("EURUSD", htf="H4", ltf="M15", **LIVE_PARAMS), htf, ltf, COSTS)
    ours = disc.run(htf, ltf, disc.OclParams(), spread=COSTS.spread,
                    pip=OclKeylevel("EURUSD").symbol.pip, side="SELL")
    assert len(live) >= 1, "market produced no live sell trades to compare"
    assert _key(ours) == _key(live)
    np.testing.assert_allclose(ours["entry"].to_numpy(), live["entry"].to_numpy(), rtol=0, atol=1e-12)
