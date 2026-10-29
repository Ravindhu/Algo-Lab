"""Engulfing with the OCL LTF entry model, the scanner's fib fields, and the in-memory sell-failure hand-off."""
import numpy as np
import pandas as pd
import pytest

from algolab.framework.ocl_backtest import run_sell_fail_pair
from algolab.framework.strategies import ocl_sell_fail as sf
from algolab.framework.strategies.engulfing_ocl import EngulfingOcl
from algolab.framework.strategies.ocl_inverse import EngulfingOclInv, mirror_frame
from algolab.framework.strategies.ocl_sell_fail import MemoryHandoff, MemoryLog, OclSellFailBottom, OclSellFailTop
from algolab.framework.symbol import ForexSymbol
from algolab.backtest.engine import Costs
from algolab.ops import state as ops_state
import importlib

eng_mod = importlib.import_module("algolab.signals.engulfing")   # the package re-exports a function of the same name
from algolab.signals.engulfing import fib_entry_level, scan_frame


def frame(rows, start, freq):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


# ---------- engulfing OCL: HTF setup and LTF touch -----------------------------

HTF_PREFIX = [(1.100, 1.104, 1.096, 1.102)] * 25        # green, no engulfing
HTF_RED = (1.100, 1.102, 1.090, 1.092)                  # body 0.008 of range 0.012
HTF_ENG = (1.091, 1.102, 1.090, 1.100)                  # bullish engulfing, body 0.009 of range 0.012
LEVEL = fib_entry_level(1.102, 1.090, +1)               # 1.09136
HTF_FORMING = (1.100, 1.101, 1.099, 1.100)

LTF_PREFIX = [(1.100, 1.104, 1.096, 1.100)] * 20
LTF_RED1 = (1.094, 1.095, 1.090, 1.0905)
LTF_RED2 = (1.0905, 1.0915, 1.0895, 1.090)
LTF_GREEN_TOUCH = (1.0900, 1.0950, 1.0880, 1.0940)     # low 1.088 reaches the level
LTF_FORMING = (1.094, 1.095, 1.093, 1.094)


def htf_frame(extra=()):
    return frame(HTF_PREFIX + [HTF_RED, HTF_ENG] + list(extra) + [HTF_FORMING], "2026-01-05", "4h")


def ltf_frame(last_closed=LTF_GREEN_TOUCH):
    return frame(LTF_PREFIX + [LTF_RED1, LTF_RED2, last_closed, LTF_FORMING], "2026-01-06 00:00", "15min")


def eng_strategy(**kw):
    params = {"hm_enabled": False, "ltf_pattern_enabled": True, **kw}
    return EngulfingOcl(ForexSymbol("EURUSD"), htf="H4", ltf="M15", **params)


def test_engulfing_ocl_arms_the_fib_level_and_fires_on_the_ltf_touch():
    s = eng_strategy()
    sig = s.signal_frames({"H4": htf_frame(), "M15": ltf_frame()})
    assert sig is not None and sig["side"] == "BUY"
    assert sig["trigger"] == pytest.approx(LEVEL)
    assert sig["htf_sl"] == pytest.approx(1.090)            # the engulfing candle's low is the floor
    assert sig["confirm_bar"] == "ENG"


def test_engulfing_ocl_no_entry_when_the_ltf_does_not_reach_the_level():
    s = eng_strategy()
    no_touch = (1.0930, 1.0950, 1.0920, 1.0940)              # low 1.092 is above the level 1.09136
    assert s.signal_frames({"H4": htf_frame(), "M15": ltf_frame(no_touch)}) is None


def test_engulfing_ocl_needs_the_ltf_red_red_green_pattern():
    s = eng_strategy()
    not_green = (1.0900, 1.0910, 1.0880, 1.0890)             # red last bar: the RRG pattern fails
    assert s.signal_frames({"H4": htf_frame(), "M15": ltf_frame(not_green)}) is None


def test_engulfing_ocl_rejects_a_small_body_engulfing():
    s = eng_strategy(min_body_pct=0.9)                       # the 0.75 body is not enough
    assert s.signal_frames({"H4": htf_frame(), "M15": ltf_frame()}) is None


def test_engulfing_ocl_inverse_is_the_mirror():
    # The SELL strategy must fire on a bearish market. It should match the BUY strategy run on the mirror of it.
    base = eng_strategy()
    inv = EngulfingOclInv(ForexSymbol("EURUSD"), htf="H4", ltf="M15", hm_enabled=False)
    htf, ltf = htf_frame(), ltf_frame()
    a = base.signal_frames({"H4": htf, "M15": ltf})                    # BUY on the bullish market
    b = inv.signal_frames({"H4": mirror_frame(htf), "M15": mirror_frame(ltf)})   # SELL on the bearish one
    assert a is not None and b is not None
    assert a["side"] == "BUY" and b["side"] == "SELL"
    assert b["trigger"] == pytest.approx(-a["trigger"])
    assert b["htf_sl"] == pytest.approx(-a["htf_sl"])


# ---------- scanner: fib fields ------------------------------------------------

def test_scanner_reports_the_fib_entry_level_and_floor(monkeypatch):
    monkeypatch.setattr(eng_mod, "hm_direction_series", lambda closes, r, e: np.ones(len(closes), dtype=int))
    rows = HTF_PREFIX + [HTF_RED, HTF_ENG, HTF_FORMING]
    df = frame(rows, "2026-01-05", "4h")[["open", "high", "low", "close"]]
    hit = scan_frame("EURUSD", df, {})
    assert hit is not None and hit["side"] == "BUY"
    assert hit["status"] == "ARMED"
    assert hit["entry_level"] == pytest.approx(LEVEL)
    assert hit["floor"] == pytest.approx(1.090)


def test_scanner_sell_side_fib_level_is_mirrored(monkeypatch):
    monkeypatch.setattr(eng_mod, "hm_direction_series", lambda closes, r, e: -np.ones(len(closes), dtype=int))
    mirrored = mirror_frame(frame(HTF_PREFIX + [HTF_RED, HTF_ENG, HTF_FORMING], "2026-01-05", "4h"))
    hit = scan_frame("EURUSD", mirrored[["open", "high", "low", "close"]], {})
    assert hit is not None and hit["side"] == "SELL"
    assert hit["entry_level"] == pytest.approx(-fib_entry_level(1.102, 1.090, +1))


# ---------- sell-failure: in-memory hand-off -----------------------------------

def test_memory_handoff_only_trusts_marks_from_the_past():
    log = MemoryLog()
    h = MemoryHandoff(log)
    t0, t1 = pd.Timestamp("2026-01-05 10:00"), pd.Timestamp("2026-01-05 11:00")
    log.clock = t0
    h.mark_exhausted("EURUSD", "core-A")
    assert h.exhausted("EURUSD", "core-A") is False          # same instant: not yet seen
    log.clock = t1
    assert h.exhausted("EURUSD", "core-A") is True           # later: the mark is in the past
    assert h.exhausted("EURUSD", "core-B") is False          # other keys unaffected


def test_sell_fail_pair_backtest_writes_no_handoff_file(tmp_path):
    rng = np.random.default_rng(2)
    n = 1500
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    opens = np.r_[close[0], close[:-1]]
    highs = np.maximum(opens, close) + np.abs(rng.normal(0, 0.3, n))
    lows = np.minimum(opens, close) - np.abs(rng.normal(0, 0.3, n))
    ltf = frame(np.column_stack([opens, highs, lows, close]), "2026-01-05", "5min")
    htf = ltf.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    costs = Costs()
    top = OclSellFailTop(ForexSymbol("EURUSD"), htf="M15", ltf="M5")
    bottom = OclSellFailBottom(ForexSymbol("EURUSD"), htf="M15", ltf="M5")
    run_sell_fail_pair(top, bottom, htf, ltf, costs)
    marks = sum(len(v) for v in top._memory_log.marks.values())
    assert marks > 0, "the sample should exhaust at least one core, so the hand-off is exercised"
    assert not (ops_state.STATE_DIR / sf.HandoffStore.FILE).exists()
