"""Parity: the vectorized trade simulator must reproduce the reference bar-by-bar loop exactly.

run_bt_loop is the reference (compiled under numba, plain Python without it). run_bt_vec must give the
same trades, in the same order, with the same exit bars and R multiples, on any input the factory can
produce. These tests run without numba, so the reference is the plain-Python loop.
"""
import numpy as np
import pytest

from algolab import discovery_factory as df_mod


def _market(n, seed):
    rng = np.random.default_rng(seed)
    c = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.0002, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.0004, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.0004, n))
    atr = np.abs(rng.normal(0.0006, 0.0002, n))
    atr[rng.random(n) < 0.03] = 0.0              # invalid ATR bars must be skipped
    atr[rng.random(n) < 0.02] = np.nan
    return o, h, l, c, atr


def _signals(n, rng, density):
    sig = np.zeros(n, np.int8)
    pick = rng.random(n) < density
    sig[pick] = rng.choice([-1, 1], size=pick.sum()).astype(np.int8)
    return sig


def _assert_same(args):
    r_ref, i_ref = df_mod.run_bt_loop(*args)
    r_vec, i_vec = df_mod.run_bt_vec(*args)
    assert len(r_vec) == len(r_ref)
    np.testing.assert_array_equal(i_vec, i_ref)
    np.testing.assert_allclose(r_vec, r_ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(25))
def test_random_markets_match_reference(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(50, 1200))
    o, h, l, c, atr = _market(n, seed)
    sig = _signals(n, rng, density=float(rng.uniform(0.01, 0.25)))
    fill = np.full(n, np.nan)
    use_fill = rng.random(n) < 0.4               # mix next-open entries and resting limits
    fill[use_fill] = c[use_fill] + rng.normal(0, 0.0003, use_fill.sum())
    sl_m = float(rng.uniform(0.5, 3.0))
    tp_m = float(rng.uniform(0.5, 4.0))
    hold = int(rng.integers(0, 60))
    cost = float(rng.uniform(0, 0.0002))
    _assert_same((o, h, l, c, atr, sig, fill, sl_m, tp_m, hold, cost))


def test_no_signals_returns_empty():
    n = 30
    o, h, l, c, atr = _market(n, 7)
    sig = np.zeros(n, np.int8)
    r, i = df_mod.run_bt_vec(o, h, l, c, atr, sig, np.full(n, np.nan), 1.5, 3.0, 10, 0.0)
    assert len(r) == 0 and len(i) == 0


def test_same_bar_sl_and_tp_takes_stop():
    # Long at open of bar 1 with a bar that touches both SL and TP: the loop assumes SL first.
    n = 6
    o = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    h = np.array([1.0, 1.0, 1.2, 1.0, 1.0, 1.0])
    l = np.array([1.0, 1.0, 0.8, 1.0, 1.0, 1.0])
    c = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    atr = np.full(n, 0.1)
    sig = np.array([0, 1, 0, 0, 0, 0], np.int8)
    fill = np.full(n, np.nan)
    args = (o, h, l, c, atr, sig, fill, 1.0, 1.0, 4, 0.0)
    _assert_same(args)
    r, i = df_mod.run_bt_vec(*args)
    assert r[0] == pytest.approx(-1.0)           # stopped at -1R, not +1R
    assert i[0] == 2


def test_fill_bar_counts_its_own_range():
    # A resting limit touched on the signal bar itself: exits are checked from that bar.
    n = 8
    o = np.full(n, 1.0)
    h = np.full(n, 1.05)
    l = np.full(n, 0.95)
    c = np.full(n, 1.0)
    atr = np.full(n, 0.1)
    sig = np.zeros(n, np.int8)
    sig[2] = -1
    fill = np.full(n, np.nan)
    fill[2] = 1.0
    _assert_same((o, h, l, c, atr, sig, fill, 2.0, 2.0, 3, 0.0))


def test_trades_do_not_overlap():
    rng = np.random.default_rng(3)
    n = 400
    o, h, l, c, atr = _market(n, 3)
    sig = _signals(n, rng, density=0.5)          # dense: most signals land inside an open trade
    _assert_same((o, h, l, c, atr, sig, np.full(n, np.nan), 1.0, 1.5, 20, 0.0))
    _, exits = df_mod.run_bt_vec(o, h, l, c, atr, sig, np.full(n, np.nan), 1.0, 1.5, 20, 0.0)
    starts = np.nonzero(sig)[0]
    assert len(exits) <= len(starts)
    assert np.all(np.diff(exits) >= 0)           # one trade at a time, in order
