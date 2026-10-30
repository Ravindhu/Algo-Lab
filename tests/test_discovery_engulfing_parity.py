"""Parity: discovery engulfing_ocl must reproduce the LIVE engulfing_ocl backtest, trade for trade.

Live reference: framework/ocl_backtest.py run_ocl_backtest (buy) and run_inverse_backtest (sell). Both
drive the same strategy object and protect.Trade that the live bot uses. The discovery version runs on
the same two timeframes and must give the same entries, exits and exit reasons.
"""
import numpy as np
import pandas as pd
import pytest

# Import the framework first: importing backtest.engine first hits a circular import in the live package.
from algolab.framework.ocl_backtest import run_inverse_backtest, run_ocl_backtest  # noqa: I001
from algolab.backtest.engine import Costs
from algolab.discovery.strategies import engulfing as disc
from algolab.framework.strategies.engulfing_ocl import EngulfingOcl
from algolab.framework.strategies.ocl_inverse import EngulfingOclInv

LIVE_PARAMS = dict(rsi_period=9, ema_period=3, min_body_pct=0.5, fib_pct=0.886, oversold=30.0,
                   hm_enabled=True, ltf_pattern_enabled=True, target_rr=3.0, max_ltf_retries=3,
                   atr_period=14, atr_multiplier=2.0, atr_trail_enabled=True, breakeven_rr=1.5,
                   round_trip_cost_pct=0.2, round_trip_cost_pips=None, disaster_stop_atr=1.0,
                   max_spread_atr=0.25, lookback=120)


def _market(n_htf, seed, per=16, p_pat=0.25):
    """Two timeframes with real setups in them.

    A bullish engulfing (red bar, then green engulfing it) or a bearish one (green, then red engulfing it) is
    placed in HTF bars. The setup only arms after the engulfing bar closes, so the LTF trigger path is
    placed in the NEXT HTF bar: a dip to the level, rebound, two counter bars, then a close in trade
    direction (dip-and-rebound for buy, rally-and-fade for sell). Everything else is a plain random bar."""
    rng = np.random.default_rng(seed)
    price = 1.10
    bars, fire = [], []
    while len(bars) < n_htf:
        if rng.random() < p_pat:
            if rng.random() < 0.5:                                  # bullish engulfing
                o1 = price
                c1 = o1 - rng.uniform(0.003, 0.006)
                o2 = c1 - 0.0001
                c2 = o1 + rng.uniform(0.0008, 0.002)
                pat = [(o1, o1 + 0.0002, c1 - 0.0002, c1), (o2, max(o2, c2) + 0.0002, min(o2, c2) - 0.0002, c2)]
                trig = "buy"
            else:                                                   # bearish engulfing
                o1 = price
                c1 = o1 + rng.uniform(0.003, 0.006)
                o2 = c1 + 0.0001
                c2 = o1 - rng.uniform(0.0008, 0.002)
                pat = [(o1, c1 + 0.0002, o1 - 0.0002, c1), (o2, max(o2, c2) + 0.0002, min(o2, c2) - 0.0002, c2)]
                trig = "sell"
            bars += pat
            fire += [None, trig]
            price = pat[-1][3]
        else:
            o = price
            c = o + rng.normal(0, 0.003)
            bars.append((o, max(o, c) + 0.0002, min(o, c) - 0.0002, c))
            fire.append(None)
            price = c
    bars, fire = bars[:n_htf], fire[:n_htf]

    htf = pd.DataFrame(bars, columns=["open", "high", "low", "close"],
                       index=pd.date_range("2024-01-01", periods=n_htf, freq="4h"))
    rows = []
    for k, (o, h, l, c) in enumerate(bars):
        r = h - l
        trig = fire[k - 1] if k > 0 else None                       # the setup that armed this bar
        if trig == "buy":
            # red bars down to the dip, then one green rebound bar whose low touches the level
            dip, up = l + 0.05 * r, l + 0.4 * r
            rebound = dip + 0.5 * (up - dip)
            path = np.concatenate([np.linspace(o, dip, 7)[1:], [rebound], np.linspace(rebound, c, 10)[1:]])
        elif trig == "sell":
            # green bars up to the top, then one red rejection bar whose high touches the level
            up, dn = h - 0.05 * r, h - 0.4 * r
            rejection = up - 0.5 * (up - dn)
            path = np.concatenate([np.linspace(o, up, 7)[1:], [rejection], np.linspace(rejection, c, 10)[1:]])
        else:
            path = np.linspace(o, c, per + 1)[1:]
        path = np.clip(path + rng.normal(0, 0.00005, per), l, h)
        prev = o
        for cl in path:
            hi = min(h, max(prev, cl) + abs(rng.normal(0, 0.00003)))
            lo = max(l, min(prev, cl) - abs(rng.normal(0, 0.00003)))
            rows.append((prev, hi, lo, cl))
            prev = cl
    ltf = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                       index=pd.date_range("2024-01-01", periods=n_htf * per, freq="15min"))
    return htf, ltf


@pytest.fixture(scope="module")
def frames():
    return _market(1500, seed=5)


# The synthetic LTF bars are small, and the live spread filter (spread > max_spread_atr * ATR skips the
# trade) would block almost every entry at a realistic spread. Both sides use the same value.
COSTS = Costs(spread=0.00003)


def _key(df):
    return [(r.entry_time, r.exit_time, r.reason) for r in df.itertuples()]


def test_buy_side_matches_live_engulfing_ocl(frames):
    htf, ltf = frames
    live = run_ocl_backtest(EngulfingOcl("EURUSD", htf="H4", ltf="M15", **LIVE_PARAMS), htf, ltf, COSTS)
    ours = disc.run(htf, ltf, disc.EngulfingParams(), spread=COSTS.spread,
                    pip=EngulfingOcl("EURUSD").symbol.pip, side="BUY")
    assert len(live) >= 2, "market produced too few live trades to be a real test"
    assert _key(ours) == _key(live)
    np.testing.assert_allclose(ours["entry"].to_numpy(), live["entry"].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ours["exit"].to_numpy(), live["exit"].to_numpy(), rtol=0, atol=1e-12)


def test_sell_side_matches_live_engulfing_ocl_inverse(frames):
    htf, ltf = frames
    live = run_inverse_backtest(EngulfingOclInv("EURUSD", htf="H4", ltf="M15", **LIVE_PARAMS), htf, ltf, COSTS)
    ours = disc.run(htf, ltf, disc.EngulfingParams(), spread=COSTS.spread,
                    pip=EngulfingOcl("EURUSD").symbol.pip, side="SELL")
    assert _key(ours) == _key(live)
    np.testing.assert_allclose(ours["entry"].to_numpy(), live["entry"].to_numpy(), rtol=0, atol=1e-12)


def test_trades_run_to_close_with_breakeven_stop_and_trail(frames):
    """Every trade must close with a reason, and the exit reasons must come from the shared trade code."""
    htf, ltf = frames
    ours = disc.run(htf, ltf, disc.EngulfingParams(), spread=COSTS.spread, pip=0.0001, side="BOTH")
    assert len(ours) > 0
    assert set(ours["reason"]) <= {"SL", "TRAIL", "TP", "DISASTER", "HTF", "EOD"}
    assert ours["exit_time"].notna().all()
