"""Parity: discovery sell_fail_top must reproduce the LIVE OclSellFailTop backtest, trade for trade.

Live reference: framework/ocl_backtest.py run_ocl_backtest (buy) and run_inverse_backtest (sell), driving the
live OclSellFailTop object and protect.Trade. TOP has no hand-off gate, so the live oracle needs no stub.
The hand-off store is replaced by an in-memory log so the run never writes the state file.
"""
import numpy as np
import pandas as pd
import pytest

# Import the framework first: importing backtest.engine first hits a circular import in the live package.
from algolab.framework.ocl_backtest import run_inverse_backtest, run_ocl_backtest  # noqa: I001
from algolab.backtest.engine import Costs
from algolab.discovery.strategies import sell_fail_top as disc
from algolab.framework.strategies.ocl_sell_fail import MemoryLog, OclSellFailTop

LIVE_PARAMS = dict(rsi_period=9, ema_period=3, oversold=30.0, hm_enabled=True, ltf_pattern_enabled=True,
                   touch_buffer=0.0, target_rr=3.0, max_ltf_retries=3, atr_period=14, atr_multiplier=2.0,
                   atr_trail_enabled=True, breakeven_rr=1.5, round_trip_cost_pct=0.2,
                   round_trip_cost_pips=None, disaster_stop_atr=1.0, max_spread_atr=0.25, lookback=120,
                   min_body_pct_c23=0.5, max_window=30, exit_mode="AUTO")

# Small spread: the synthetic LTF bars are small, so the live spread filter would block almost every entry
# at a realistic spread. Both sides use the same value.
COSTS = Costs(spread=0.00003)


def _live(htf, ltf, side):
    strat = OclSellFailTop("EURUSD", htf="H4", ltf="M15", **LIVE_PARAMS)
    strat.use_memory_handoff(MemoryLog())          # never write the live state file in a test
    # Framework gap: run_ocl_backtest reads sig["confirm_bar"], which the sell-fail signal never sets (KeyError).
    # The test adds the field without touching the live code. Its value is not used for sell-fail.
    signal = strat.signal_frames

    def _with_confirm(frames):
        sig = signal(frames)
        if sig is not None:
            sig = {**sig, "confirm_bar": None}
        return sig

    strat.signal_frames = _with_confirm
    if side == "BUY":
        return run_ocl_backtest(strat, htf, ltf, COSTS)
    return run_inverse_backtest(strat, htf, ltf, COSTS)


def _reflect(bar, C):
    o, h, l, c = bar
    return (2 * C - o, 2 * C - l, 2 * C - h, 2 * C - c)


def _buy_trigger_path(bar, dip, rng, per):
    """LTF path inside the trigger HTF bar: red bars down to the dip, one green rebound, then up to close."""
    o, h, l, c = bar
    rebound = dip + 0.00005                        # small green step: RSI stays under the oversold cap
    path = np.concatenate([np.linspace(o, dip, 10)[1:], [rebound], np.linspace(rebound, c, 7)[1:]])
    return np.clip(path + rng.normal(0, 0.00003, per), l, h)


def _market_sell_fail(n_htf, seed, per=16, p_pat=0.3):
    """HTF bars with failed-short cores: C1 any, C2 green, C3 red (both bodies > 50%), then a break bar that
    closes above C3's open. The trigger path sits in the bar after the break, when the setup is armed.
    Sell setups are the buy setups reflected about the starting price (the mirror the live inverse uses).
    Buy-side level is the C3 open (TOP tier 1), so the trigger dips to it."""
    rng = np.random.default_rng(seed)
    price = 1.10
    bars, marks = [], []
    while len(bars) < n_htf:
        if rng.random() < p_pat:
            p0 = price
            side = "buy" if rng.random() < 0.5 else "sell"
            c1 = p0 - rng.uniform(0.002, 0.004)
            bar1 = (p0, p0 + 0.0002, c1 - 0.0002, c1)                       # C1 red
            o2 = c1 + 0.0001
            c2 = o2 + rng.uniform(0.002, 0.004)
            bar2 = (o2, c2 + 0.0002, o2 - 0.0002, c2)                       # C2 green
            o3 = c2
            c3 = o3 - rng.uniform(0.002, 0.004)
            bar3 = (o3, o3 + 0.0002, c3 - 0.0002, c3)                       # C3 red, aggressive level = o3
            c4 = o3 + rng.uniform(0.0005, 0.0015)
            bar4 = (c3, max(c3, c4) + 0.0002, min(c3, c4) - 0.0002, c4)    # break: close above o3
            dip = o3 - 0.0001
            c5 = dip + 0.0005
            bar5 = (c4, max(c4, c5) + 0.0002, min(c4, dip) - 0.0002, c5)   # trigger bar
            pat = [bar1, bar2, bar3, bar4, bar5]
            if side == "sell":
                pat = [_reflect(b, p0) for b in pat]
            bars += pat
            marks += [None, None, None, None, (side, dip, p0, bar5)]
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
        else:
            side, dip, p0, bar_buy = mk
            if side == "buy":
                path = _buy_trigger_path(bar_buy, dip, rng, per)
            else:                                   # sell: the buy path on the un-reflected bar, reflected back
                path = 2 * p0 - _buy_trigger_path(bar_buy, dip, rng, per)
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
    return _market_sell_fail(1500, seed=31)


def _key(df):
    return [(r.entry_time, r.exit_time, r.reason) for r in df.itertuples()]


def test_buy_side_matches_live_sell_fail_top(frames):
    htf, ltf = frames
    live = _live(htf, ltf, "BUY")
    ours = disc.run(htf, ltf, disc.SellFailParams(), spread=COSTS.spread,
                    pip=OclSellFailTop("EURUSD").symbol.pip, side="BUY")
    assert len(live) >= 2, "market produced too few live trades to be a real test"
    assert _key(ours) == _key(live)
    np.testing.assert_allclose(ours["entry"].to_numpy(), live["entry"].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ours["exit"].to_numpy(), live["exit"].to_numpy(), rtol=0, atol=1e-12)


def test_sell_side_matches_live_sell_fail_top_inverse(frames):
    htf, ltf = frames
    live = _live(htf, ltf, "SELL")
    ours = disc.run(htf, ltf, disc.SellFailParams(), spread=COSTS.spread,
                    pip=OclSellFailTop("EURUSD").symbol.pip, side="SELL")
    assert len(live) >= 1, "market produced no live sell trades to compare"
    assert _key(ours) == _key(live)
    np.testing.assert_allclose(ours["entry"].to_numpy(), live["entry"].to_numpy(), rtol=0, atol=1e-12)


def test_trades_close_with_reasons_from_shared_trade_code(frames):
    htf, ltf = frames
    ours = disc.run(htf, ltf, disc.SellFailParams(), spread=COSTS.spread, pip=0.0001, side="BOTH")
    assert len(ours) > 0
    assert set(ours["reason"]) <= {"SL", "TRAIL", "TP", "DISASTER", "HTF", "EOD"}
    assert ours["exit_time"].notna().all()
