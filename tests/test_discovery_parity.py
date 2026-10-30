"""Parity tests: the discovery backtester's engulfing_ocl family must reproduce the LIVE engulfing rule.

Discovery (src/algolab/discovery_factory.py) is report-only, but its results are only meaningful
if the same bars give the same patterns, the same fib levels and the same entry bars as the live
Engulfing strategy. These tests compare the two on random price data.
"""
import numpy as np
import pandas as pd
import pytest

from algolab import discovery_factory as df_mod
from algolab.framework.strategies.engulfing_strategy import Engulfing, engulfing
from algolab.framework.strategies.indicators import hm_direction


def _random_ohlc(n, seed):
    """Random OHLC with plenty of engulfing candles (varied body sizes, wicks, gaps)."""
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    o = np.empty(n)
    o[0] = close[0]
    o[1:] = close[:-1] + rng.normal(0, 0.0003, n - 1)
    c = o + rng.normal(0, 0.0016, n) * rng.uniform(0.2, 1.8, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.0006, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.0006, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)


@pytest.fixture(scope="module")
def data():
    return _random_ohlc(5000, seed=7)


def test_random_data_has_patterns(data):
    """Guard: the parity tests below are only meaningful if the data really contains setups."""
    live = engulfing(data, 0.5).to_numpy()
    assert np.count_nonzero(live) > 30


def test_pattern_direction_matches_live_rule(data):
    """Body filter (both candles > 50% of range) + HM direction on the pattern bar."""
    I = df_mod.Ind(data)
    disc = df_mod._engulf_pattern(I, {"rsi_period": 9, "ema_period": 3, "inv": 0})

    live_eng = engulfing(data, 0.5).to_numpy()
    live_hm = hm_direction(data["close"].to_numpy(), 9, 3)
    live = np.where((live_eng != 0) & (live_hm == live_eng), live_eng, 0)

    # Compare after the indicator warm-up (first ~20 bars), where RSI/EMA are both defined.
    assert np.array_equal(disc[20:], live[20:])


def test_fib_entries_match_live_engulfing_signal(data):
    """For every closed bar k, the live Engulfing.signal (fed bars up to k) and the discovery
    fill model must agree on: whether a trade fires on bar k, its side, and its entry level."""
    I = df_mod.Ind(data)
    sig, fill = df_mod.f_engulfing_ocl(I, {"rsi_period": 9, "ema_period": 3, "inv": 0})
    live = Engulfing("EURUSD")

    checked, fired = 0, 0
    for k in range(20, len(data) - 1):
        # The live signal evaluates the last CLOSED bar, i.e. the second-to-last row of its input.
        frame = data.iloc[: k + 2]
        got = live.signal(frame)
        if got is None:
            assert sig[k] == 0, f"bar {k}: discovery fires but live does not"
        else:
            fired += 1
            side = 1 if got["side"] == "BUY" else -1
            assert sig[k] == side, f"bar {k}: side mismatch"
            assert abs(fill[k] - got["entry"]) < 1e-12, f"bar {k}: entry level mismatch"
        checked += 1
    assert checked > 1000
    assert fired > 5, "random data produced too few live setups to be a real test"


def test_inverse_family_negates_the_pattern(data):
    """inv=1 flips the pattern direction (the mirrored setup). The fib level is then measured from
    the mirrored extreme, so inverse trades land on different bars, and we check the pattern itself."""
    I = df_mod.Ind(data)
    d0 = df_mod._engulf_pattern(I, {"rsi_period": 9, "ema_period": 3, "inv": 0})
    d1 = df_mod._engulf_pattern(I, {"rsi_period": 9, "ema_period": 3, "inv": 1})
    assert np.array_equal(d1, -d0)
    sig1, _ = df_mod.f_engulfing_ocl(I, {"rsi_period": 9, "ema_period": 3, "inv": 1})
    assert np.count_nonzero(sig1) > 0



def test_run_bt_fill_price_and_exit_from_touch_bar():
    """A fill price is used as the entry, and the exits are checked from the touch bar."""
    n = 10
    o = np.full(n, 11.0)
    h = np.full(n, 11.0)
    l = np.full(n, 11.0)
    c = np.full(n, 11.0)
    atr = np.ones(n)
    sig = np.zeros(n, np.int8)
    fill = np.full(n, np.nan)

    # Long resting limit at 10.0 on bar 2 (the touch bar). Stop 1 ATR below, target 2 ATR above.
    sig[2] = 1
    fill[2] = 10.0
    l[2], h[2] = 9.5, 10.5           # touch bar: fills at 10.0, no stop, no target
    h[3] = 12.5                      # next bar reaches the target at 12.0
    r, _ = df_mod.run_bt(o, h, l, c, atr, sig, fill, 1.0, 2.0, 5, 0.0)
    assert len(r) == 1
    assert r[0] == pytest.approx(2.0)   # (12.0 - 10.0) / 1.0 ATR stop


def test_run_bt_without_fill_uses_next_open():
    """Without a fill price the entry is the next bar's open, as before."""
    n = 10
    o = np.full(n, 11.0)
    h = np.full(n, 11.5)
    l = np.full(n, 10.5)
    c = np.full(n, 11.0)
    atr = np.ones(n)
    sig = np.zeros(n, np.int8)
    fill = np.full(n, np.nan)
    sig[2] = 1
    o[3] = 11.0
    h[4] = 13.5                      # target 11 + 2 = 13 reached on bar 4
    r, _ = df_mod.run_bt(o, h, l, c, atr, sig, fill, 1.0, 2.0, 5, 0.0)
    assert len(r) == 1
    assert r[0] == pytest.approx(2.0)
