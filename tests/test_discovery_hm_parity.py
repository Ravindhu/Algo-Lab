"""Parity: the discovery family `hm_cross` and the template strategy `plain_template` must both
fire on exactly the bars where the LIVE Hilega-Milega direction turns into +1 (BUY) or -1 (SELL).
If they drift apart, the discovery results stop describing the live rule."""
import sys

import numpy as np
import pandas as pd
import pytest

from algolab import discovery_factory as df_mod
from algolab.framework.strategies.indicators import hm_direction
from algolab.framework.strategies.plain_template import PlainTemplateStrategy


def _random_ohlc(n, seed):
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


def _expected(closes, rsi_n, ema_n):
    """Live rule: BUY on the bar HM direction becomes +1, SELL on the bar it becomes -1."""
    d = hm_direction(closes, rsi_n, ema_n)
    prev = np.concatenate([[0], d[:-1]])
    long_ = (d == 1) & (prev != 1)
    short_ = (d == -1) & (prev != -1)
    return long_, short_


@pytest.fixture(scope="module")
def data():
    return _random_ohlc(1500, seed=11)


@pytest.mark.parametrize("rsi_n,ema_n", [(9, 3), (5, 2), (14, 6)])
def test_discovery_hm_cross_matches_live_hm(data, rsi_n, ema_n):
    closes = data["close"].to_numpy(float)
    I = df_mod.Ind(data)
    sig = df_mod.f_hm_cross(I, {"rsi_n": rsi_n, "ema_n": ema_n, "inv": 0})
    long_, short_ = _expected(closes, rsi_n, ema_n)
    warm = 3 * (rsi_n + ema_n) + 5          # skip the warm-up bars (undefined RSI / EMA)
    assert np.count_nonzero(long_[warm:]) > 10, "guard: data must contain HM crosses"
    assert np.array_equal(sig[warm:] == 1, long_[warm:])
    assert np.array_equal(sig[warm:] == -1, short_[warm:])


def test_inverse_flag_swaps_sides(data):
    I = df_mod.Ind(data)
    a = df_mod.f_hm_cross(I, {"rsi_n": 9, "ema_n": 3, "inv": 0})
    b = df_mod.f_hm_cross(I, {"rsi_n": 9, "ema_n": 3, "inv": 1})
    assert np.array_equal(a == 1, b == -1) and np.array_equal(a == -1, b == 1)


def test_template_strategy_matches_live_hm(data):
    """The template's signal() on the last CLOSED bar must agree with the live HM crosses."""
    closes = data["close"].to_numpy(float)
    long_, short_ = _expected(closes, 9, 3)
    strat = PlainTemplateStrategy("EURUSD", "M15", rsi_period=9, ema_period=3)
    checked = 0
    for k in range(120, 400):                 # df.iloc[:k+2] -> last closed bar is index k
        res = strat.signal(data.iloc[: k + 2])
        want = "BUY" if long_[k] else ("SELL" if short_[k] else None)
        got = res["side"] if res else None
        assert got == want, f"bar {k}: template {got} vs live {want}"
        if want:
            checked += 1
    assert checked > 3, "guard: the window must contain HM crosses"
