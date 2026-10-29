"""OCL SELL_FAILURE_TOP and SELL_FAILURE_BOTTOM: the failed-short blueprints.

All OCL blueprints are BUY setups. The failing pattern is a SHORT: C2 green
(its open = conservative sell level), C3 red (its open = aggressive sell
level), >50% bodies; a later candle closes above the aggressive sell level and
the levels flip to BUY levels on the retest from above. TOP has two tiers
(broken C3 open, then C2 open); BOTTOM has one (the lowest swing candle's open).
No broker needed.
"""
import numpy as np
import pandas as pd

from algolab.framework.strategies.ocl_sell_fail import (
    OclSellFailBottom, OclSellFailTop, _break_setup)


def htf_frame(bars, start="2026-01-05", freq="15min"):
    idx = pd.date_range(start, periods=len(bars), freq=freq)
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)


def htf_window(extra=0):
    """HTF frame: warm-up bars before the failed-short core, then the core plus
    the failure candle. The last row of a frame is treated as still forming, so
    the frame carries one extra row (a forming bar) after the failure candle."""
    warm = [(130.0 - i, 130.5 - i, 129.5 - i, 130.0 - i) for i in range(extra)]
    return htf_frame(warm + list(FAIL_CORE) + [(138, 138.5, 137.5, 138)])


def ltf_frame(rows, start="2026-01-05 01:00", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def _bar(o, h, l, c):
    from algolab.framework.strategies.ocl import Bar
    return Bar(o, h, l, c)


def _bars_of(rows):
    return [_bar(*r) for r in rows]


# ---- HTF window: failed short --------------------------------------------
# C1 red, C2 green with big body (open 120), C3 red with big body (open 135).
# The levels: aggressive SELL = 135 (C3 open), conservative SELL = 120 (C2 open).
# A later candle closes above 135 -> the short failed.
FAIL_CORE = [
    (125, 126, 124, 125),               # C1 (colour unconstrained in the blueprint)
    (120, 134, 119, 133),               # C2 GREEN, open 120 = conservative SELL
    (135, 136, 125, 126),               # C3 RED, open 135 = aggressive SELL
    (130, 139, 129, 138),               # failure candle: closes 138 > 135
]

# ---- LTF window: retest from above into tier 1, with a red-red-green ------
LTF_TOUCH = [
    (139.0 + i * 0.2, 139.5 + i * 0.2, 138.5 + i * 0.2, 139.0 + i * 0.2)
    for i in range(20)                                    # ATR warm-up drift
] + [
    (139.0, 139.4, 138.0, 138.2),        # red
    (138.2, 138.6, 132.0, 132.5),        # red
    (132.5, 136.5, 119.5, 136.0),        # green: deep touch of 120 (tier 2 / BOTTOM level)
    (136.0, 136.4, 135.5, 135.8),        # newest row: still forming, dropped by closed()
]


def test_failed_short_core_detection_and_flipped_levels():
    bars = _bars_of(FAIL_CORE)
    setup = _break_setup(bars, OclSellFailTop(ForexSymbolStub()).p)
    assert setup is not None
    assert setup["aggressive"] == 135.0          # broken C3 open
    assert setup["conservative"] == 120.0        # C2 open


class ForexSymbolStub:
    name = "EURUSD"
    pip = 0.0001


def test_stale_level_is_rejected():
    # A candle between the core and the break already traded at/above the
    # levels -> the retest would not be the first test; the setup must be None.
    bars = _bars_of(FAIL_CORE[:3])
    bars.append(_bar(132, 137, 131, 133))        # in-between candle: high 137 >= 135, close below
    bars.append(_bar(134, 140, 133, 139))        # the actual break candle
    assert _break_setup(bars, OclSellFailTop(ForexSymbolStub()).p) is None


def _strategy(cls):
    s = cls("EURUSD", htf="M15", ltf="M5", lookback=30, hm_enabled=False,
            ltf_pattern_enabled=False)
    s.candles_by_tf = {"M15": htf_window(extra=0), "M5": ltf_frame(LTF_TOUCH)}
    return s


def test_top_strategy_signals_buy_at_tier1():
    s = _strategy(OclSellFailTop)
    sig = s.signal_frames({"M15": s.candles_by_tf["M15"],
                           "M5": s.candles_by_tf["M5"]})
    assert sig is not None and sig["side"] == "BUY"
    assert sig["trigger"] == 135.0               # tier 1 = broken C3 open
    assert sig["aggressive"] == 135.0 and sig["conservative"] == 120.0


def test_top_escalates_to_tier2_after_tier1_stop():
    s = _strategy(OclSellFailTop)
    frames = {"M15": s.candles_by_tf["M15"], "M5": s.candles_by_tf["M5"]}
    sig = s.signal_frames(frames)
    s.on_entry()
    s.on_exit("SL", sig["entry_ref"] * 0.99)     # tier-1 LTF stop-out
    sig2 = s.signal_frames(frames)
    assert sig2 is not None and sig2["trigger"] == 120.0   # tier 2 = C2 open


def test_bottom_waits_for_the_top_handoff():
    # No TOP has retired this core yet -> BOTTOM must stay silent.
    s = _strategy(OclSellFailBottom)
    assert s.signal_frames({"M15": s.candles_by_tf["M15"],
                            "M5": s.candles_by_tf["M5"]}) is None


def _run_top_to_exhaustion(symbol="EURUSD"):
    """Drive a TOP strategy through tier 1 and tier 2 until it retires,
    marking the hand-off store. Returns (frames, handoff_file_contents)."""
    from algolab.ops.state import read_json
    top = _strategy(OclSellFailTop)
    frames = {"M15": top.candles_by_tf["M15"], "M5": top.candles_by_tf["M5"]}
    s1 = top.signal_frames(frames)
    top.on_entry()
    top.on_exit("SL", s1["entry_ref"] * 0.99)          # tier-1 stop-out
    s2 = top.signal_frames(frames)                      # tier 2 armed
    top.on_entry()
    for _ in range(top.p.max_ltf_retries):              # exhaust the ladder
        top.on_exit("SL", s2["entry_ref"] * 0.99)
        s2 = top.signal_frames(frames)
        if s2 is None:
            break
        top.on_entry()
    return frames, read_json("framework_handoff.json", {})


def test_bottom_runs_after_top_exhausts_the_core(tmp_path, monkeypatch):
    monkeypatch.setattr("algolab.ops.state.STATE_DIR", tmp_path)
    frames, recorded = _run_top_to_exhaustion()
    assert recorded, "TOP must record its hand-off when it exhausts"
    bottom = _strategy(OclSellFailBottom)
    sig = bottom.signal_frames(frames)
    assert sig is not None and sig["side"] == "BUY"
    # BOTTOM's single level = the lowest swing candle's OPEN (C2, open 120).
    assert sig["trigger"] == 120.0
    assert sig["htf_sl"] == 119.0


def test_bottom_has_a_single_level_and_no_tier2_escalation(tmp_path, monkeypatch):
    monkeypatch.setattr("algolab.ops.state.STATE_DIR", tmp_path)
    frames, _ = _run_top_to_exhaustion()
    s = _strategy(OclSellFailBottom)
    sig = s.signal_frames(frames)
    s.on_entry()
    s.on_exit("SL", sig["entry_ref"] * 0.99)     # the single level's stop
    # BOTTOM has no second tier: the ladder steps DOWN below the stop close,
    # like the normal OCL retry ladder.
    sig2 = s.signal_frames(frames)
    assert sig2 is not None and sig2["trigger"] <= sig["trigger"]


def test_ordinary_rrg_core_does_not_arm_the_failure_blueprints():
    # A normal red-red-green with a close above C2 open is the NORMAL OCL
    # setup, not a failed short: C2 must be green for the failure blueprints.
    bars = _bars_of([
        (125, 126, 124, 125),
        (122, 123, 117, 118),                    # C2 RED
        (120, 134, 119, 133),                    # C3 GREEN
        (132, 139, 131, 138),                    # closes above C2 open
    ])
    assert _break_setup(bars, OclSellFailTop(ForexSymbolStub()).p) is None
