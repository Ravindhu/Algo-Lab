#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
algolab.discovery.factory  -  strategy discovery for Algo-Lab (REPORT ONLY)
==========================================================================

Breeds, tests and ranks candidate strategies on the SAME history CSVs the rest of
Algo-Lab uses (data/{SYMBOL}_{TF}.csv). It follows the repo's own safety rules:

  * Reads CSVs only. It never imports MetaTrader5 (live/adapter.py stays the only MT5 module).
  * The final 10% of every history file is SEALED: never loaded into breeding, gates or
    ledger statistics, so `algolab sealed` stays a clean one-time test.
  * Every run that reaches the hold-out gate is counted in state/discovery_ledger.json.
    After max_holdout_uses on the same data window the factory refuses until new data arrives.
  * It never edits config/strategy.json or config/framework.json. It only writes
    reports/discovery.json (+ csv). A person reviews, then promotes a candidate.

Pipeline (one run): breeding (GA) -> optimizer (hill-climb) -> auto-tuner (adaptive mutation)
-> loop feedback (family weights + gene bank) -> built-in backtester -> gates
(validation, beat-random, Monte-Carlo, nudge, hold-out) -> top-N report.

Public API for later live wiring:
    from algolab.discovery.factory import signal_for_genome
    sig = signal_for_genome(df, genome)      # int8 array; sig[-1] = signal on last CLOSED bar

NOT financial advice. Backtests are estimates. Forward-test on demo first.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# numba is optional but makes everything ~10-20x faster.   pip install numba
# --------------------------------------------------------------------------------------
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

np.seterr(all="ignore")


# ======================================================================================
# 1. BUILT-IN BACKTESTER
# ======================================================================================
@njit(cache=True)
def run_bt(o, h, l, c, atr, sig, sl_m, tp_m, max_hold, cost):
    """
    Signal at close of bar i  ->  entry at OPEN of bar i+1 (no look-ahead).
    One trade at a time. SL/TP in ATR multiples. If SL and TP both touched in the same
    bar, SL is assumed first (pessimistic). Result per trade is in R (risk multiples):
    R = (price move - cost) / stop distance.
    """
    n = len(c)
    out_r = np.empty(n, np.float64)
    out_i = np.empty(n, np.int64)
    k = 0
    idx = np.nonzero(sig)[0]
    m = len(idx)
    p = 0
    while p < m:
        i = idx[p]
        if i >= n - 2:
            break
        a = atr[i]
        if not a > 0.0:
            p += 1
            continue
        s = sig[i]
        entry = o[i + 1]
        sl_d = a * sl_m
        tp_d = a * tp_m
        if s == 1:
            sl = entry - sl_d
            tp = entry + tp_d
        else:
            sl = entry + sl_d
            tp = entry - tp_d
        last = i + 1 + max_hold
        if last > n - 1:
            last = n - 1
        j_exit = last
        exit_p = c[last]
        for j in range(i + 1, last + 1):
            if s == 1:
                hit_sl = l[j] <= sl
                hit_tp = h[j] >= tp
            else:
                hit_sl = h[j] >= sl
                hit_tp = l[j] <= tp
            if hit_sl:
                exit_p = sl
                j_exit = j
                break
            if hit_tp:
                exit_p = tp
                j_exit = j
                break
        pnl = (exit_p - entry) * s - cost
        out_r[k] = pnl / sl_d
        out_i[k] = j_exit
        k += 1
        while p < m and idx[p] < j_exit:
            p += 1
    return out_r[:k], out_i[:k]


def metrics(r, risk=0.01):
    """Trade statistics from an array of R multiples. risk = fraction of equity per trade."""
    n = len(r)
    if n == 0:
        return dict(n=0, exp=0.0, pf=0.0, wr=0.0, tot=0.0, dd_r=0.0, dd_pct=0.0,
                    ret_pct=0.0, t=0.0, recovery=0.0)
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    pf = gp / gl if gl > 1e-12 else (99.0 if gp > 0 else 0.0)
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    dd_r = float((peak - cum).max())
    eq = np.cumprod(1.0 + risk * r)
    pk = np.maximum.accumulate(np.maximum(eq, 1.0))
    dd_pct = float((1.0 - eq / pk).max() * 100.0)
    sd = r.std(ddof=1) if n > 1 else 0.0
    t = r.mean() / sd * np.sqrt(n) if sd > 1e-12 else 0.0
    tot = float(cum[-1])
    rec = tot / dd_r if dd_r > 1e-9 else (tot if tot > 0 else 0.0)
    return dict(n=int(n), exp=float(r.mean()), pf=float(min(pf, 99.0)),
                wr=float((r > 0).mean() * 100.0), tot=tot, dd_r=dd_r, dd_pct=dd_pct,
                ret_pct=float((eq[-1] - 1.0) * 100.0), t=float(t), recovery=float(rec))


def fit_from_r(r, min_n, risk):
    """Fitness used by the breeding loop (TRAIN slice only)."""
    n = len(r)
    if n < min_n:
        return -5.0 + n / max(min_n, 1)          # strongly penalise too few trades
    m = metrics(r, risk)
    if m["t"] <= 0:
        return m["t"]
    return m["t"] * (1.0 - min(0.7, m["dd_pct"] / 60.0)) + 0.3 * min(m["recovery"], 5.0)


# ======================================================================================
# 2. INDICATORS (all causal - value at bar i uses data up to bar i only)
# ======================================================================================
def _sma(x, n):
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def _wilder(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def _std(x, n):
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def _hh(x, n):
    return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()


def _ll(x, n):
    return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()


def _sh(x, k=1):
    out = np.empty_like(x, dtype=np.float64)
    out[:k] = np.nan
    out[k:] = x[:-k]
    return out


def _shb(x):
    out = np.roll(x, 1)
    out[0] = False
    return out


def _xup(a, b):
    return (a > b) & (_sh(a) <= _sh(b))


def _xdn(a, b):
    return (a < b) & (_sh(a) >= _sh(b))


def _sig(long_m, short_m, inv=0):
    if inv:
        long_m, short_m = short_m, long_m
    s = np.zeros(len(long_m), np.int8)
    s[long_m] = 1
    s[short_m] = -1
    return s


@njit(cache=True)
def _supertrend(h, l, c, atr, mult):
    n = len(c)
    out = np.zeros(n, np.int8)
    direction = 0
    fub_p = np.nan
    flb_p = np.nan
    for i in range(n):
        a = atr[i]
        if np.isnan(a):
            continue
        hl2 = (h[i] + l[i]) * 0.5
        ub = hl2 + mult * a
        lb = hl2 - mult * a
        if np.isnan(fub_p):
            fub = ub
            flb = lb
            direction = 1 if c[i] >= hl2 else -1
        else:
            fub = ub if (ub < fub_p or c[i - 1] > fub_p) else fub_p
            flb = lb if (lb > flb_p or c[i - 1] < flb_p) else flb_p
            if direction == -1 and c[i] > fub_p:
                direction = 1
            elif direction == 1 and c[i] < flb_p:
                direction = -1
        fub_p = fub
        flb_p = flb
        out[i] = direction
    return out


class Ind:
    """Indicator cache over one symbol/timeframe."""

    def __init__(self, df):
        self.o = df["open"].to_numpy(np.float64).copy()
        self.h = df["high"].to_numpy(np.float64).copy()
        self.l = df["low"].to_numpy(np.float64).copy()
        self.c = df["close"].to_numpy(np.float64).copy()
        self.hour = df.index.hour.to_numpy()
        self._m = {}
        pc = _sh(self.c)
        tr = np.maximum(self.h - self.l, np.maximum(np.abs(self.h - pc), np.abs(self.l - pc)))
        tr[0] = self.h[0] - self.l[0]
        self.tr = tr

    def _memo(self, key, fn):
        v = self._m.get(key)
        if v is None:
            if len(self._m) > 1500:
                self._m.clear()
            v = fn()
            self._m[key] = v
        return v

    def sma(self, n): return self._memo(("sma", n), lambda: _sma(self.c, n))
    def ema(self, n): return self._memo(("ema", n), lambda: _ema(self.c, n))
    def std(self, n): return self._memo(("std", n), lambda: _std(self.c, n))
    def hh(self, n): return self._memo(("hh", n), lambda: _hh(self.h, n))
    def ll(self, n): return self._memo(("ll", n), lambda: _ll(self.l, n))
    def atr(self, n): return self._memo(("atr", n), lambda: _wilder(self.tr, n))

    def atr_ratio(self, n):
        return self._memo(("atrr", n), lambda: self.atr(n) / _sma(self.atr(n), 100))

    def rsi(self, n):
        def f():
            d = np.diff(self.c, prepend=self.c[0])
            au = _wilder(np.maximum(d, 0.0), n)
            ad = _wilder(np.maximum(-d, 0.0), n)
            return 100.0 - 100.0 / (1.0 + au / ad)
        return self._memo(("rsi", n), f)

    def stoch(self, n):
        def f():
            hh, ll = _hh(self.h, n), _ll(self.l, n)
            return _sma((self.c - ll) / (hh - ll) * 100.0, 3)
        return self._memo(("stoch", n), f)

    def cci(self, n):
        def f():  # std-based approximation of mean deviation (fast)
            tp = (self.h + self.l + self.c) / 3.0
            return (tp - _sma(tp, n)) / (0.015 * 0.7979 * _std(tp, n))
        return self._memo(("cci", n), f)

    def willr(self, n):
        def f():
            hh, ll = _hh(self.h, n), _ll(self.l, n)
            return (hh - self.c) / (hh - ll) * -100.0
        return self._memo(("willr", n), f)

    def adx(self, n):
        def f():
            up = self.h - _sh(self.h)
            dn = _sh(self.l) - self.l
            pdm = np.where((up > dn) & (up > 0), up, 0.0)
            mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
            a = _wilder(self.tr, n)
            pdi = 100.0 * _wilder(pdm, n) / a
            mdi = 100.0 * _wilder(mdm, n) / a
            dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
            return _wilder(dx, n), pdi, mdi
        return self._memo(("adx", n), f)

    def mom(self, n):  # momentum in ATR units (scale free across instruments)
        return self._memo(("mom", n), lambda: (self.c - _sh(self.c, n)) / self.atr(14))

    def supertrend(self, n, m):
        return self._memo(("st", n, m), lambda: _supertrend(self.h, self.l, self.c, self.atr(n), float(m)))

    # ---- Hilega-Milega (exact port of algolab/hilega_milega.py: SIMPLE-average RSI) ----
    def hm_rsi(self, n):
        """Simple-average RSI, same convention as hilega_milega.hilega_milega_rsi."""
        def f():
            c = self.c
            d = np.diff(c, prepend=c[0])
            g = pd.Series(np.maximum(d, 0.0)).rolling(n, min_periods=n).mean().to_numpy()
            lo = pd.Series(np.maximum(-d, 0.0)).rolling(n, min_periods=n).mean().to_numpy()
            rs = np.where(lo > 0, g / np.maximum(lo, 1e-12), np.inf)
            return 100.0 - 100.0 / (1.0 + rs)
        return self._memo(("hmrsi", n), f)

    def hm_line(self, rsi_n, e_n):
        def f(): return _ema(self.hm_rsi(rsi_n), e_n)
        return self._memo(("hmline", rsi_n, e_n), f)

    def engulf(self):
        def f():
            po, pc = _sh(self.o), _sh(self.c)
            bull = (pc < po) & (self.c > self.o) & (self.o <= pc) & (self.c >= po)
            bear = (pc > po) & (self.c < self.o) & (self.o >= pc) & (self.c <= po)
            return np.where(bull, 1, np.where(bear, -1, 0))
        return self._memo(("engulf",), f)

    def hm(self, rsi_n, e_n):
        """Hilega-Milega direction: +1 RSI above EMA(RSI), -1 below, 0 unknown."""
        def f():
            r, e = self.hm_rsi(rsi_n), self.hm_line(rsi_n, e_n)
            return np.where(r > e, 1, np.where(r < e, -1, 0))
        return self._memo(("hm", rsi_n, e_n), f)


# ======================================================================================
# 3. STRATEGY LIBRARY  (entry families)
# ======================================================================================
def f_ema_cross(I, p):
    a, b = I.ema(p["fast"]), I.ema(p["slow"])
    return _sig(_xup(a, b), _xdn(a, b), p["inv"])


# OCL entry, ported 1:1 from Algo-Lab (algolab/signals/engulfing.py + framework
# strategies/engulfing_strategy.py + hilega_milega.py). BUY = bullish engulfing AND
# RSI above EMA(RSI); SELL = bearish engulfing AND RSI below EMA(RSI). RSI is the
# simple-average RSI convention of hilega_milega.py (NOT the Wilder RSI of I.rsi).
def f_ocl_hm(I, p):
    eng = I.engulf()
    conf = I.hm(p["rn"], p["ren"])
    bull = (eng > 0) & (conf > 0)
    bear = (eng < 0) & (conf < 0)
    return _sig(bull, bear, p["inv"])


def f_macd(I, p):
    m = I.ema(p["fast"]) - I.ema(p["slow"])
    s = _ema(m, p["sig"])
    return _sig(_xup(m, s), _xdn(m, s), p["inv"])


def f_rsi_rev(I, p):
    r = I.rsi(p["n"])
    return _sig(r < p["lo"], r > p["hi"])


def f_boll_rev(I, p):
    m, s = I.sma(p["n"]), I.std(p["n"])
    return _sig(I.c < m - p["k"] * s, I.c > m + p["k"] * s)


def f_boll_break(I, p):
    m, s = I.sma(p["n"]), I.std(p["n"])
    up, lo = m + p["k"] * s, m - p["k"] * s
    return _sig(_xup(I.c, up), _xdn(I.c, lo), p["inv"])


def f_donchian(I, p):
    hi, lo = _sh(I.hh(p["n"])), _sh(I.ll(p["n"]))
    return _sig(I.c > hi, I.c < lo, p["inv"])


def f_stoch(I, p):
    k = I.stoch(p["n"])
    return _sig(k < p["lo"], k > 100 - p["lo"])


def f_cci(I, p):
    x = I.cci(p["n"])
    return _sig(x < -p["lvl"], x > p["lvl"])


def f_willr(I, p):
    w = I.willr(p["n"])
    return _sig(w < -100 + p["lvl"], w > -p["lvl"])


def f_mom(I, p):
    x = I.mom(p["n"])
    return _sig(_xup(x, np.full_like(x, p["thr"])), _xdn(x, np.full_like(x, -p["thr"])), p["inv"])


def f_keltner(I, p):
    mid, a = I.ema(p["n"]), I.atr(p["n"])
    return _sig(_xup(I.c, mid + p["m"] * a), _xdn(I.c, mid - p["m"] * a), p["inv"])


def f_supertrend(I, p):
    d = I.supertrend(p["n"], p["m"]).astype(np.int16)
    pd_ = np.roll(d, 1)
    pd_[0] = 0
    flip = (d != pd_) & (pd_ != 0) & (d != 0)
    return _sig(flip & (d == 1), flip & (d == -1), p["inv"])


def f_adx_di(I, p):
    a, pdi, mdi = I.adx(p["n"])
    ok = a > p["thr"]
    return _sig(_xup(pdi, mdi) & ok, _xup(mdi, pdi) & ok)


def f_ichimoku(I, p):
    t = (I.hh(p["t"]) + I.ll(p["t"])) / 2.0
    k = (I.hh(p["k"]) + I.ll(p["k"])) / 2.0
    return _sig(_xup(t, k), _xdn(t, k), p["inv"])


def f_zscore(I, p):
    z = (I.c - I.sma(p["n"])) / I.std(p["n"])
    return _sig(z < -p["z"], z > p["z"])


def f_engulf(I, p):
    o, c = I.o, I.c
    po, pc = _sh(o), _sh(c)
    big = np.abs(c - o) > p["minbody"] * I.atr(14)
    bull = (pc < po) & (c > o) & (c >= po) & (o <= pc) & big
    bear = (pc > po) & (c < o) & (c <= po) & (o >= pc) & big
    return _sig(bull, bear, p["inv"])


def f_consec(I, p):
    d = np.diff(I.c, prepend=I.c[0])
    dn = pd.Series((d < 0).astype(float)).rolling(p["n"]).sum().to_numpy() == p["n"]
    up = pd.Series((d > 0).astype(float)).rolling(p["n"]).sum().to_numpy() == p["n"]
    return _sig(dn, up, p["inv"])


def f_squeeze(I, p):
    m, s, a = I.sma(p["n"]), I.std(p["n"]), I.atr(p["n"])
    sq = (p["k"] * s) < (p["km"] * a)
    rel = _shb(sq) & ~sq
    return _sig(rel & (I.c > m), rel & (I.c < m))


def f_hour_break(I, p):
    at = I.hour == p["hour"]
    hi, lo = _sh(I.hh(p["n"])), _sh(I.ll(p["n"]))
    return _sig(at & (I.c > hi), at & (I.c < lo), p["inv"])


def f_trend_pullback(I, p):
    e, r = I.ema(p["trend"]), I.rsi(p["rn"])
    return _sig((I.c > e) & (r < p["lo"]), (I.c < e) & (r > 100 - p["lo"]))


INV = ("i", 0, 1)
FAMILIES = {
    "ocl_hm":         dict(fn=f_ocl_hm, space=dict(rn=("i", 5, 21), ren=("i", 2, 8), inv=INV)),
    "ema_cross":      dict(fn=f_ema_cross, space=dict(fast=("i", 3, 30), slow=("i", 20, 200), inv=INV)),
    "macd_cross":     dict(fn=f_macd, space=dict(fast=("i", 5, 20), slow=("i", 15, 50), sig=("i", 3, 15), inv=INV)),
    "rsi_reversion":  dict(fn=f_rsi_rev, space=dict(n=("i", 2, 21), lo=("i", 10, 35), hi=("i", 65, 90))),
    "boll_reversion": dict(fn=f_boll_rev, space=dict(n=("i", 10, 60), k=("f", 1.5, 3.0))),
    "boll_breakout":  dict(fn=f_boll_break, space=dict(n=("i", 10, 60), k=("f", 1.5, 3.0), inv=INV)),
    "donchian":       dict(fn=f_donchian, space=dict(n=("i", 10, 100), inv=INV)),
    "stochastic":     dict(fn=f_stoch, space=dict(n=("i", 5, 21), lo=("i", 10, 30))),
    "cci_reversion":  dict(fn=f_cci, space=dict(n=("i", 10, 50), lvl=("i", 80, 200))),
    "williams_r":     dict(fn=f_willr, space=dict(n=("i", 7, 30), lvl=("i", 10, 30))),
    "momentum":       dict(fn=f_mom, space=dict(n=("i", 5, 60), thr=("f", 0.5, 5.0), inv=INV)),
    "keltner":        dict(fn=f_keltner, space=dict(n=("i", 10, 50), m=("f", 1.0, 3.0), inv=INV)),
    "supertrend":     dict(fn=f_supertrend, space=dict(n=("i", 7, 21), m=("f", 1.5, 4.0), inv=INV)),
    "adx_di":         dict(fn=f_adx_di, space=dict(n=("i", 7, 30), thr=("i", 15, 35))),
    "ichimoku_tk":    dict(fn=f_ichimoku, space=dict(t=("i", 5, 15), k=("i", 20, 60), inv=INV)),
    "zscore_rev":     dict(fn=f_zscore, space=dict(n=("i", 10, 100), z=("f", 1.0, 3.0))),
    "engulfing":      dict(fn=f_engulf, space=dict(minbody=("f", 0.2, 1.5), inv=INV)),
    "consecutive":    dict(fn=f_consec, space=dict(n=("i", 2, 7), inv=INV)),
    "squeeze":        dict(fn=f_squeeze, space=dict(n=("i", 15, 40), k=("f", 1.5, 2.5), km=("f", 1.0, 2.0))),
    "hour_breakout":  dict(fn=f_hour_break, space=dict(hour=("i", 0, 23), n=("i", 4, 40), inv=INV)),
    "trend_pullback": dict(fn=f_trend_pullback, space=dict(trend=("i", 50, 200), rn=("i", 2, 5), lo=("i", 10, 40))),
}
FAM_NAMES = list(FAMILIES)


# --------------------------------------------------------------------------------------
# Filters: return (long_allowed, short_allowed) boolean arrays, or None
# --------------------------------------------------------------------------------------
def flt_none(I, p): return None
def flt_trend(I, p):
    e = I.ema(p["n"]); return I.c > e, I.c < e
def flt_adx_min(I, p):
    m = I.adx(p["n"])[0] > p["thr"]; return m, m
def flt_adx_max(I, p):
    m = I.adx(p["n"])[0] < p["thr"]; return m, m
def flt_vol_high(I, p):
    m = I.atr_ratio(p["n"]) > p["ratio"]; return m, m
def flt_vol_low(I, p):
    m = I.atr_ratio(p["n"]) < p["ratio"]; return m, m
def flt_session(I, p):
    m = ((I.hour - p["start"]) % 24) < p["len"]; return m, m


def flt_rsi_align(I, p):
    r = I.rsi(p["n"]); return r > p["mid"], r < p["mid"]


FILTERS = {
    "none":      dict(fn=flt_none, space={}),
    "trend_ema": dict(fn=flt_trend, space=dict(n=("i", 20, 200))),
    "adx_min":   dict(fn=flt_adx_min, space=dict(n=("i", 7, 30), thr=("i", 15, 35))),
    "adx_max":   dict(fn=flt_adx_max, space=dict(n=("i", 7, 30), thr=("i", 15, 30))),
    "vol_high":  dict(fn=flt_vol_high, space=dict(n=("i", 14, 30), ratio=("f", 0.8, 1.6))),
    "vol_low":   dict(fn=flt_vol_low, space=dict(n=("i", 14, 30), ratio=("f", 0.6, 1.2))),
    "session":   dict(fn=flt_session, space=dict(start=("i", 0, 23), len=("i", 4, 12))),
    "rsi_align": dict(fn=flt_rsi_align, space=dict(n=("i", 5, 21), mid=("i", 40, 60))),
}
FLT_NAMES = list(FILTERS)
FLT_P = np.array([0.28, 0.14, 0.11, 0.09, 0.09, 0.07, 0.12, 0.10])
FLT_P = FLT_P / FLT_P.sum()

EXIT_SPACE = dict(sl=("f", 0.8, 4.0), tp=("f", 0.8, 6.0), hold=("i", 4, 100))


# ======================================================================================
# 4. GENOME helpers: random / repair / mutate / crossover / nudge
# ======================================================================================
def gkey(g):
    return json.dumps(g, sort_keys=True)


def _rand_val(spec, rng, name=""):
    kind, lo, hi = spec
    if name == "inv":
        return int(rng.random() < 0.25)
    if kind == "i":
        return int(rng.integers(lo, hi + 1))
    return float(round(rng.uniform(lo, hi), 2))


def rand_params(space, rng):
    return {k: _rand_val(v, rng, k) for k, v in space.items()}


def _clip(v, spec):
    kind, lo, hi = spec
    v = min(hi, max(lo, v))
    return int(round(v)) if kind == "i" else float(round(v, 2))


def repair(g):
    p, f = g["p"], g["fam"]
    if f in ("ema_cross", "macd_cross"):
        p["slow"] = max(p["slow"], p["fast"] + 3)
    elif f == "ichimoku_tk":
        p["k"] = max(p["k"], p["t"] + 3)
    elif f == "ocl_hm":
        p["ren"] = max(p["ren"], 2)
    for k, spec in FAMILIES[f]["space"].items():
        p[k] = _clip(p[k], spec)
    # filters that were re-rolled (e.g. pinned-genome shortcut) get a fresh param dict
    if not isinstance(g.get("fp"), dict) or set(g["fp"]) != set(FILTERS[g["flt"]]["space"]):
        g["fp"] = {}
    for k, spec in FILTERS[g["flt"]]["space"].items():
        if not isinstance(g["fp"].get(k), (int, float)) or g["fp"][k] is None:
            g["fp"][k] = (spec[1] + spec[2]) / 2.0   # midpoint fallback
        g["fp"][k] = _clip(g["fp"][k], spec)
    for k, spec in EXIT_SPACE.items():
        g[k] = _clip(g[k], spec)
    return g


def random_genome(rng, fw):
    fam = str(rng.choice(FAM_NAMES, p=fw))
    flt = str(rng.choice(FLT_NAMES, p=FLT_P))
    g = dict(fam=fam, p=rand_params(FAMILIES[fam]["space"], rng),
             flt=flt, fp=rand_params(FILTERS[flt]["space"], rng),
             sl=_rand_val(EXIT_SPACE["sl"], rng), tp=_rand_val(EXIT_SPACE["tp"], rng),
             hold=_rand_val(EXIT_SPACE["hold"], rng),
             dir=int(rng.choice([0, 1, -1], p=[0.6, 0.2, 0.2])))
    return repair(g)


def _pinned_genome(pin, rng):
    g = dict(fam=pin, p=rand_params(FAMILIES[pin]["space"], rng),
             flt=str(rng.choice(FLT_NAMES, p=FLT_P)), fp=None,
             sl=_rand_val(EXIT_SPACE["sl"], rng), tp=_rand_val(EXIT_SPACE["tp"], rng),
             hold=_rand_val(EXIT_SPACE["hold"], rng),
             dir=int(rng.choice([0, 1, -1], p=[0.6, 0.2, 0.2])))
    return repair(g)


def valid_genome(g):
    try:
        return (g["fam"] in FAMILIES and g["flt"] in FILTERS and
                set(g["p"]) == set(FAMILIES[g["fam"]]["space"]) and
                set(g["fp"]) == set(FILTERS[g["flt"]]["space"]))
    except Exception:
        return False


def _perturb(v, spec, rng, scale):
    kind, lo, hi = spec
    span = hi - lo
    if kind == "i":
        if lo == 0 and hi == 1:
            return 1 - int(v)
        step = int(round(rng.normal(0, scale * span)))
        if step == 0:
            step = int(rng.choice([-1, 1]))
        return int(min(hi, max(lo, v + step)))
    return float(round(min(hi, max(lo, v + rng.normal(0, scale * span))), 2))


def numeric_keys(g):
    keys = []
    for k, spec in FAMILIES[g["fam"]]["space"].items():
        keys.append(("p", k, spec))
    for k, spec in FILTERS[g["flt"]]["space"].items():
        keys.append(("fp", k, spec))
    for k, spec in EXIT_SPACE.items():
        keys.append(("x", k, spec))
    return keys


def _get(g, sec, k):
    return g[k] if sec == "x" else g[sec][k]


def _set(g, sec, k, v):
    if sec == "x":
        g[k] = v
    else:
        g[sec][k] = v


def mutate(g, rng, rate, fw, pinned_fam=None):
    g = copy.deepcopy(g)
    if pinned_fam is None and rng.random() < rate * 0.15:
        fam = str(rng.choice(FAM_NAMES, p=fw))
        g["fam"], g["p"] = fam, rand_params(FAMILIES[fam]["space"], rng)
    if rng.random() < rate * 0.15:
        flt = str(rng.choice(FLT_NAMES, p=FLT_P))
        g["flt"], g["fp"] = flt, rand_params(FILTERS[flt]["space"], rng)
    for sec, k, spec in numeric_keys(g):
        if rng.random() < rate:
            _set(g, sec, k, _perturb(_get(g, sec, k), spec, rng, 0.15))
    if rng.random() < rate * 0.1:
        g["dir"] = int(rng.choice([0, 1, -1]))
    return repair(g)


def crossover(a, b, rng, pinned_fam=None):
    c = copy.deepcopy(a)
    if pinned_fam is not None:
        if a["flt"] == b["flt"]:
            for k in c["fp"]:
                if rng.random() < 0.5:
                    c["fp"][k] = b["fp"][k]
        elif rng.random() < 0.5:
            c["flt"], c["fp"] = b["flt"], copy.deepcopy(b["fp"])
        for k in ("sl", "tp", "hold", "dir"):
            if rng.random() < 0.5:
                c[k] = b[k]
        return repair(c)
    if a["fam"] == b["fam"]:
        for k in c["p"]:
            if rng.random() < 0.5:
                c["p"][k] = b["p"][k]
    elif rng.random() < 0.5:
        c["fam"], c["p"] = b["fam"], copy.deepcopy(b["p"])
    if a["flt"] == b["flt"]:
        for k in c["fp"]:
            if rng.random() < 0.5:
                c["fp"][k] = b["fp"][k]
    elif rng.random() < 0.5:
        c["flt"], c["fp"] = b["flt"], copy.deepcopy(b["fp"])
    for k in ("sl", "tp", "hold", "dir"):
        if rng.random() < 0.5:
            c[k] = b[k]
    return repair(c)


def nudge(g, rng, mag=0.15):
    """Small random change of every numeric gene - used for robustness testing."""
    g = copy.deepcopy(g)
    for sec, k, spec in numeric_keys(g):
        if spec == INV:
            continue
        v = _get(g, sec, k)
        nv = v * (1.0 + rng.uniform(-mag, mag))
        if spec[0] == "i":
            nv = int(round(nv))
            if nv == v and rng.random() < 0.5:
                nv += int(rng.choice([-1, 1]))
        _set(g, sec, k, nv)
    return repair(g)


def describe(g):
    p = ", ".join(f"{k}={v}" for k, v in g["p"].items())
    f = ", ".join(f"{k}={v}" for k, v in g["fp"].items())
    d = {0: "long+short", 1: "long only", -1: "short only"}[g["dir"]]
    return (f"{g['fam']}({p}) | filter {g['flt']}({f}) | SL {g['sl']}xATR TP {g['tp']}xATR "
            f"hold<={g['hold']} bars | {d}")


# ======================================================================================
# 5. ENGINE: signal -> backtest -> fitness
# ======================================================================================
@dataclass
class Data:
    symbol: str
    tf: str
    df: pd.DataFrame
    cost: float          # round-trip cost in PRICE units (spread*mult + commission)
    note: str = ""
    window: str = ""

    @property
    def key(self):
        return f"{self.symbol}|{self.tf}"


class Engine:
    def __init__(self, data: Data, args):
        self.data, self.args = data, args
        self.ind = Ind(data.df)
        I = self.ind
        self.atr = I.atr(14)
        n = len(data.df)
        a, b = int(n * 0.6), int(n * 0.8)
        self.seg = dict(train=(0, a), val=(a, b), test=(b, n), dev=(0, b), all=(0, n))
        self.cost = float(data.cost)
        self.risk = args.risk / 100.0
        self.cache, self.seen = {}, {}

    def signal(self, g):
        I = self.ind
        s = FAMILIES[g["fam"]]["fn"](I, g["p"]).copy()
        if g["dir"] == 1:
            s[s < 0] = 0
        elif g["dir"] == -1:
            s[s > 0] = 0
        fl = FILTERS[g["flt"]]["fn"](I, g["fp"])
        if fl is not None:
            lm, sm = fl
            s[(s > 0) & ~lm] = 0
            s[(s < 0) & ~sm] = 0
        return s

    def trades(self, g, seg, sig=None):
        if sig is None:
            sig = self.signal(g)
        a, b = self.seg[seg]
        I = self.ind
        r, _ = run_bt(I.o[a:b], I.h[a:b], I.l[a:b], I.c[a:b], self.atr[a:b], sig[a:b],
                      float(g["sl"]), float(g["tp"]), int(g["hold"]), self.cost)
        return r

    def fitness(self, g):
        k = gkey(g)
        v = self.cache.get(k)
        if v is None:
            v = fit_from_r(self.trades(g, "train"), self.args.min_train_trades, self.risk)
            self.cache[k], self.seen[k] = v, g
        return v


# ======================================================================================
# 6. BREEDING LOOP  (GA + optimizer + auto-tuner + loop feedback)
# ======================================================================================
def local_opt(eng, g, rounds=2):
    """OPTIMIZER: coordinate hill-climb around a genome (train fitness only)."""
    best, bf = g, eng.fitness(g)
    for _ in range(rounds):
        improved = False
        for sec, k, spec in numeric_keys(best):
            kind, lo, hi = spec
            if spec == INV:
                continue
            step = max(1, int(round(0.08 * (hi - lo)))) if kind == "i" else 0.08 * (hi - lo)
            for sgn in (1, -1):
                cand = copy.deepcopy(best)
                _set(cand, sec, k, _get(cand, sec, k) + sgn * step)
                cand = repair(cand)
                if gkey(cand) == gkey(best):
                    continue
                f = eng.fitness(cand)
                if f > bf + 1e-9:
                    best, bf, improved = cand, f, True
        if not improved:
            break
    return best


def load_bank(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {"weights": {}, "elite": {}}


def breed(eng: Engine, args, rng, bank):
    N, key = args.pop, eng.data.key
    pin = getattr(args, "pin", None)
    if pin and pin not in FAMILIES:
        sys.exit(f"--pin family '{pin}' is not in FAMILIES (use one of {FAM_NAMES})")
    w = np.array([bank.get("weights", {}).get(f, 1.0) for f in FAM_NAMES], float)
    fw = w / w.sum()

    pop, keys = [], set()

    def add(g):
        k = gkey(g)
        if k not in keys:
            keys.add(k)
            pop.append(g)
            return True
        return False

    seeds = [s for s in bank.get("elite", {}).get(key, []) if valid_genome(s)]
    if pin:
        seeds = [s for s in seeds if s["fam"] == pin]
    for s in seeds[: int(N * 0.3)]:
        add(repair(s))
    n_seed = len(pop)
    while len(pop) < N:
        add(_pinned_genome(pin, rng) if pin else random_genome(rng, fw))
    print(f"   population {N}  (seeded from gene bank: {n_seed}, random: {N - n_seed}" +
          (f", PINNED fam={pin})" if pin else ")"))

    mr, best_hist, stagn = 0.25, -1e9, 0
    for gen in range(1, args.gens + 1):
        t0 = time.time()
        fits = np.array([eng.fitness(g) for g in pop])
        order = np.argsort(-fits)
        pop = [pop[i] for i in order]
        fits = fits[order]

        # ---- LOOP FEEDBACK: reward families that produce winners -------------------
        top = [g for g, f in zip(pop[: max(10, N // 5)], fits[: max(10, N // 5)]) if f > 0]
        if top:
            share = np.array([sum(1 for g in top if g["fam"] == f) for f in FAM_NAMES], float)
            share /= share.sum()
            fw = 0.8 * fw + 0.2 * share
            fw = np.maximum(fw, 0.01)
            fw /= fw.sum()

        # ---- AUTO-TUNER: adapt mutation rate -------------------------------------------
        if fits[0] > best_hist + 1e-9:
            best_hist, stagn = fits[0], 0
            mr = max(0.12, mr * 0.92)
        else:
            stagn += 1
            if stagn >= 3:
                mr = min(0.60, mr * 1.3)

        promising = int((fits > 1.0).sum())
        print(f"   gen {gen:>2}/{args.gens}  best {fits[0]:6.2f}  top10-avg {fits[:10].mean():6.2f}  "
              f"promising {promising:>4}  evaluated {len(eng.cache):>6}  mut {mr:.2f}  {time.time() - t0:4.1f}s")
        if gen == args.gens:
            break

        # ---- build next generation -----------------------------------------------------
        n_el = max(2, int(0.10 * N))
        n_opt = max(3, int(0.03 * N))
        tuned = [local_opt(eng, g) for g in pop[:n_opt]]
        elites = tuned + pop[n_opt:n_el]
        nxt, keys = [], set()
        for g in elites:
            add_k = gkey(g)
            if add_k not in keys:
                keys.add(add_k)
                nxt.append(g)
        n_imm = int(0.10 * N)
        tries = 0
        while len(nxt) < N - n_imm and tries < N * 20:
            tries += 1
            a = pop[int(rng.integers(0, N, 4).min())]
            b = pop[int(rng.integers(0, N, 4).min())]
            child = crossover(a, b, rng, pinned_fam=pin) if rng.random() < 0.7 else a
            child = mutate(child, rng, mr, fw, pinned_fam=pin)
            k = gkey(child)
            if k in keys:
                continue
            keys.add(k)
            nxt.append(child)
        while len(nxt) < N:
            g = _pinned_genome(pin, rng) if pin else random_genome(rng, fw)
            k = gkey(g)
            if k not in keys:
                keys.add(k)
                nxt.append(g)
        pop = nxt

    bank.setdefault("weights", {})
    for f, x in zip(FAM_NAMES, fw):
        bank["weights"][f] = float(0.5 * bank["weights"].get(f, 1.0 / len(FAM_NAMES)) + 0.5 * x)

    # finalists = best distinct train genomes (cap per family to keep diversity)
    ranked = sorted(eng.cache.items(), key=lambda kv: -kv[1])
    per_fam, fin = {}, []
    for k, f in ranked:
        if f <= 0.5 or len(fin) >= args.finalists:
            break
        g = eng.seen[k]
        if per_fam.get(g["fam"], 0) >= max(8, args.finalists // 5):
            continue
        per_fam[g["fam"]] = per_fam.get(g["fam"], 0) + 1
        fin.append((f, g))
    return fin


# ======================================================================================
# 7. FILTER GATES  (validation, random baseline, Monte Carlo, nudge, hold-out)
# ======================================================================================
def run_gates(eng: Engine, finalists, args, rng):
    funnel = dict(candidates=len(finalists), val=0, rand=0, mc=0, nudge=0, holdout=0)
    out = []
    risk = eng.risk
    I = eng.ind
    for fit_train, g in finalists:
        sig = eng.signal(g)
        mt = metrics(eng.trades(g, "train", sig), risk)
        r_val = eng.trades(g, "val", sig)
        mv = metrics(r_val, risk)

        # GATE 1: validation slice
        if mv["n"] < args.min_val_trades or mv["exp"] <= 0 or mv["pf"] < args.min_pf:
            continue
        funnel["val"] += 1

        # GATE 2: must beat most of 100 random-entry strategies using the SAME exits
        a, b = eng.seg["val"]
        freq = float((sig[a:b] != 0).mean())
        rand_exp = []
        for _ in range(100):
            rs = np.zeros(len(sig), np.int8)
            hit = rng.random(len(sig)) < freq
            if g["dir"] == 0:
                rs[hit] = rng.choice(np.array([-1, 1], np.int8), hit.sum())
            else:
                rs[hit] = g["dir"]
            rr = eng.trades(g, "val", rs)
            rand_exp.append(rr.mean() if len(rr) else -1.0)
        beat = float((np.array(rand_exp) < mv["exp"]).mean())
        if beat < args.min_beat_random:
            continue
        funnel["rand"] += 1

        # GATE 3: Monte-Carlo shuffle of trade order -> 95th percentile drawdown
        r_dev = eng.trades(g, "dev", sig)
        idx = rng.random((1000, len(r_dev))).argsort(axis=1)
        eq = np.cumprod(1.0 + risk * r_dev[idx], axis=1)
        pk = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
        dd95 = float(np.percentile((1.0 - eq / pk).max(axis=1) * 100.0, 95))
        if dd95 > args.max_dd:
            continue
        funnel["mc"] += 1

        # GATE 4: nudge parameters - most neighbours must stay profitable
        ok = 0
        for _ in range(20):
            gg = nudge(g, rng)
            rr = eng.trades(gg, "dev")
            mm = metrics(rr, risk)
            if mm["n"] >= args.min_train_trades // 2 and mm["exp"] > 0 and mm["pf"] >= 1.0:
                ok += 1
        nudge_rate = ok / 20.0
        if nudge_rate < args.min_nudge:
            continue
        funnel["nudge"] += 1

        # GATE 5: final hold-out slice (never used in breeding or earlier gates)
        mte = metrics(eng.trades(g, "test", sig), risk)
        if mte["n"] < args.min_test_trades or mte["exp"] <= 0 or mte["pf"] < 1.0:
            continue
        funnel["holdout"] += 1

        dev_exp = (mt["n"] * mt["exp"] + mv["n"] * mv["exp"]) / max(1, mt["n"] + mv["n"])
        persist = min(1.0, max(0.0, mte["exp"] / dev_exp)) if dev_exp > 0 else 0.0
        n_all = mt["n"] + mv["n"] + mte["n"]
        score = (30 * persist + 20 * nudge_rate + 15 * beat
                 + 15 * max(0.0, 1 - dd95 / args.max_dd)
                 + 10 * min(1.0, max(0.0, mte["pf"] - 1.0))
                 + 10 * min(1.0, n_all / 200.0))
        out.append(dict(symbol=eng.data.symbol, tf=eng.data.tf, score=round(float(score), 1),
                        genome=g, desc=describe(g), train=mt, val=mv, test=mte,
                        mc_dd95=round(dd95, 1), nudge=nudge_rate, beat_random=beat, trades=n_all))
    return out, funnel


# ======================================================================================
# 8. REPO INTEGRATION: root, config, CSV loader, ledger, public API
# ======================================================================================
DEFAULT_CFG = dict(
    symbols=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
    timeframes=["M15", "H4"],
    data_dir="data",
    pop=1000, gens=25, finalists=150, top=10, max_per_family=3,
    risk=1.0, spread_mult=1.8,
    min_train_trades=80, min_val_trades=25, min_test_trades=15,
    min_pf=1.15, min_beat_random=0.90, min_nudge=0.50, max_dd=25.0,
    sealed_frac=0.10, max_holdout_uses=3, max_bars=60000, seed=42,
    # round-trip SPREAD in PRICE units. Replace with your broker's real numbers.
    costs={"EURUSD": 0.00012, "GBPUSD": 0.00015, "USDJPY": 0.012, "AUDUSD": 0.00014,
           "USDCAD": 0.00016, "USDCHF": 0.00016, "NZDUSD": 0.00018, "EURJPY": 0.020,
           "GBPJPY": 0.030, "XAUUSD": 0.30, "XAGUSD": 0.03},
)


def find_root(explicit=None):
    """Repo root = nearest parent holding pyproject.toml + src/algolab."""
    if explicit:
        return Path(explicit).resolve()
    for base in (Path.cwd(), Path(__file__).resolve()):
        for p in [base, *base.parents]:
            if (p / "pyproject.toml").exists() and (p / "src" / "algolab").is_dir():
                return p
    return Path.cwd()


def load_cfg(root):
    cfg = copy.deepcopy(DEFAULT_CFG)
    p = root / "config" / "discovery.json"
    if p.exists():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
            costs = user.pop("costs", {})
            cfg.update(user)
            cfg["costs"].update(costs)
        except Exception as e:  # fail loudly but do not crash the whole run
            print(f"!! could not read {p}: {e} - using defaults")
    return cfg


def read_csv_flexible(path):
    """Reads OHLC CSVs written by MT5 exports, pandas, or the Windows sync.
    Accepts , ; tab separators, <DATE>/<TIME> headers, epoch seconds/ms, any column case."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        col = next((c for c in ("time", "timestamp", "datetime", "date", "dt", "open_time", "open time")
                    if c in df.columns), df.columns[0])
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            ts = pd.to_datetime(s, unit="ms" if float(s.median()) > 1e11 else "s")
        else:
            ts = pd.to_datetime(s)
    for need in ("open", "high", "low", "close"):
        if need not in df.columns:
            raise ValueError(f"{path}: missing column '{need}' (found {list(df.columns)})")
    out = pd.DataFrame({k: pd.to_numeric(df[k], errors="coerce") for k in ("open", "high", "low", "close")})
    out.index = pd.DatetimeIndex(ts)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def cost_for(symbol, df, cfg):
    c = cfg["costs"].get(symbol)
    if isinstance(c, dict):
        spread = float(c.get("spread", 0.0)) * cfg["spread_mult"] + float(c.get("commission", 0.0))
        return spread, "config"
    if c is not None:
        return float(c) * cfg["spread_mult"], "config"
    h, l, cl = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    tr = np.maximum(h - l, np.abs(h - np.roll(cl, 1)))[1:]
    guess = 0.06 * float(np.median(tr)) * cfg["spread_mult"]
    return guess, "GUESS (set costs." + symbol + " in config/discovery.json)"


def load_csv_datasets(args, cfg, root):
    out = []
    for sym in args.symbols:
        for tf in args.timeframes:
            p = root / cfg["data_dir"] / f"{sym}_{tf}.csv"
            if not p.exists():
                print(f"-- {p.name} not found in {p.parent} - skipped")
                continue
            try:
                df = read_csv_flexible(p)
            except Exception as e:
                print(f"!! {p.name}: {e} - skipped")
                continue
            if len(df) > cfg["max_bars"]:
                df = df.iloc[-cfg["max_bars"]:]
            cut = int(len(df) * (1.0 - cfg["sealed_frac"]))
            work, sealed = df.iloc[:cut], df.iloc[cut:]
            if len(work) < 3000:
                print(f"!! {p.name}: only {len(work)} usable bars after sealing - skipped (need >= 3000)")
                continue
            cost, src = cost_for(sym, work, cfg)
            d = Data(sym, tf, work, cost,
                     note=f"{len(work)} bars {work.index[0]:%Y-%m-%d}..{work.index[-1]:%Y-%m-%d} | "
                          f"SEALED {len(sealed)} bars from {sealed.index[0]:%Y-%m-%d} untouched | "
                          f"cost/trade {cost:.5g} [{src}]")
            d.window = f"{work.index[0].isoformat()}..{work.index[-1].isoformat()}"
            out.append(d)
    return out


def synth(n=30000, seed=1):
    """Synthetic demo series with weak regime-switching autocorrelation (NOT real market)."""
    rng = np.random.default_rng(seed)
    vol = 0.0006 * (1 + 0.5 * np.sin(np.arange(n) / 800.0))
    r = np.zeros(n)
    phi = 0.0
    for i in range(1, n):
        if i % 2500 == 0:
            phi = float(rng.choice([-0.12, 0.0, 0.12]))
        r[i] = phi * r[i - 1] + rng.normal(0, vol[i])
    c = 1.10 * np.exp(np.cumsum(r))
    o = np.roll(c, 1)
    o[0] = c[0]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0002, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0002, n)))
    idx = pd.date_range("2021-01-01", periods=n, freq="h")
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c), index=idx)


def load_demo(args, cfg):
    d = Data("DEMO", "H1", synth(20000, args.seed), cost=0.00012 * cfg["spread_mult"],
             note="synthetic data - pipeline test only, results are meaningless")
    d.window = "demo"
    return [d]


def ledger_path(root):
    return root / "state" / "discovery_ledger.json"


def ledger_load(root):
    try:
        return json.loads(ledger_path(root).read_text(encoding="utf-8"))
    except Exception:
        return {}


def ledger_save(root, led):
    p = ledger_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(led, indent=1), encoding="utf-8")


def signal_for_genome(df, genome):
    """PUBLIC API for live use. df = recent CLOSED-bar OHLC (DatetimeIndex), >= ~300 bars.
    Returns int8 array (+1 buy / -1 sell / 0 none). sig[-1] is the signal on the last closed bar;
    the order would be placed at the next bar's open (same as the backtest)."""
    class _S:  # minimal stand-in so we reuse Engine.signal without a full Engine
        pass
    s = _S()
    s.ind = Ind(df)
    return Engine.signal(s, genome)


# ======================================================================================
# 9. MAIN
# ======================================================================================
def _py(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return str(o)


def parse(argv=None, cfg=None):
    cfg = cfg or DEFAULT_CFG
    ap = argparse.ArgumentParser(prog="algolab discover",
                                 description="Breed, test and rank candidate strategies (report only).")
    ap.add_argument("--root", help="repo root (auto-detected)")
    ap.add_argument("--symbols", default=",".join(cfg["symbols"]))
    ap.add_argument("--timeframes", default=",".join(cfg["timeframes"]), help="M5,M15,H4 ... must exist in data/")
    ap.add_argument("--pop", type=int, default=cfg["pop"])
    ap.add_argument("--gens", type=int, default=cfg["gens"])
    ap.add_argument("--finalists", type=int, default=cfg["finalists"])
    ap.add_argument("--top", type=int, default=cfg["top"])
    ap.add_argument("--max-per-family", type=int, default=cfg["max_per_family"])
    ap.add_argument("--risk", type=float, default=cfg["risk"], help="%% risk per trade for DD/return stats")
    ap.add_argument("--min-train-trades", type=int, default=cfg["min_train_trades"])
    ap.add_argument("--min-val-trades", type=int, default=cfg["min_val_trades"])
    ap.add_argument("--min-test-trades", type=int, default=cfg["min_test_trades"])
    ap.add_argument("--min-pf", type=float, default=cfg["min_pf"])
    ap.add_argument("--min-beat-random", type=float, default=cfg["min_beat_random"])
    ap.add_argument("--min-nudge", type=float, default=cfg["min_nudge"])
    ap.add_argument("--max-dd", type=float, default=cfg["max_dd"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--force", action="store_true", help="ignore the hold-out use limit (NOT recommended)")
    ap.add_argument("--pin", default=None,
                    help="pin the entry family (e.g. ocl_hm) - the GA then only evolves the filter, "
                         "exits and direction; filter weights stay fixed and every genome uses the pin")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no CSV needed")
    a = ap.parse_args(argv)
    a.symbols = [s.strip() for s in str(a.symbols).split(",") if s.strip()]
    a.timeframes = [t.strip().upper() for t in str(a.timeframes).split(",") if t.strip()]
    return a


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--root")
    known, _ = pre.parse_known_args(argv)
    root = find_root(known.root)
    cfg = load_cfg(root)
    args = parse(argv, cfg)
    rng = np.random.default_rng(args.seed)
    print("=" * 100)
    print(f" ALGO-LAB DISCOVERY | root {root} | numba: {'ON' if HAVE_NUMBA else 'OFF (pip install numba = 10-20x faster)'}"
          f" | families {len(FAMILIES)} | filters {len(FILTERS)}"
          + (f" | PINNED: {args.pin}" if getattr(args, 'pin', None) else ""))
    print(" REPORT ONLY - never changes config/strategy.json or framework.json. Sealed final "
          f"{int(cfg['sealed_frac'] * 100)}% is never read.")
    print("=" * 100)

    datasets = load_demo(args, cfg) if args.demo else load_csv_datasets(args, cfg, root)
    if not datasets:
        sys.exit(f"No data found. Put CSVs in {root / cfg['data_dir']} as SYMBOL_TF.csv "
                 f"(e.g. EURUSD_M15.csv) or use scripts/export_history.py, or try --demo.")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    bank_path = state_dir / "discovery_gene_bank.json"
    bank = load_bank(bank_path)
    led = ledger_load(root)
    results, skipped = [], []

    for d in datasets:
        lk = f"{d.key}|{d.window}"
        used = int(led.get(lk, {}).get("uses", 0))
        print(f"\n>>> {d.symbol} {d.tf}   {d.note}")
        if not args.demo and used >= cfg["max_holdout_uses"] and not args.force:
            print(f"   !! hold-out already used {used}x on this exact data window (limit "
                  f"{cfg['max_holdout_uses']}). Refusing - more looks = overfitting the hold-out.\n"
                  f"      Add fresh bars to the CSV, or run `algolab sealed` on your best candidate.")
            skipped.append(d.key)
            continue
        t0 = time.time()
        eng = Engine(d, args)
        fin = breed(eng, args, rng, bank)
        res, fn = run_gates(eng, fin, args, rng)
        print(f"   FUNNEL: finalists {fn['candidates']} -> validation {fn['val']} -> beat-random {fn['rand']} "
              f"-> monte-carlo {fn['mc']} -> nudge {fn['nudge']} -> hold-out {fn['holdout']}   "
              f"({time.time() - t0:.0f}s)")
        if fn["nudge"] > 0 and not args.demo:       # this run really looked at the hold-out
            led[lk] = dict(uses=used + 1, last=datetime.now(timezone.utc).isoformat())
            ledger_save(root, led)
        results += res
        elite = sorted(res, key=lambda x: -x["score"])[:30]
        bank.setdefault("elite", {})[d.key] = [x["genome"] for x in elite] or [g for _, g in fin[:20]]
        bank_path.write_text(json.dumps(bank, indent=1, default=_py), encoding="utf-8")

    results.sort(key=lambda x: -x["score"])
    final, per = [], {}
    for r in results:
        f = r["genome"]["fam"]
        if per.get(f, 0) >= args.max_per_family:
            continue
        per[f] = per.get(f, 0) + 1
        final.append(r)
        if len(final) >= args.top:
            break

    print("\n" + "=" * 100)
    print(f" FINAL TOP {len(final)}  (all gates passed, robustness score 0-100)")
    print("=" * 100)
    if not final:
        print(" No candidate survived all gates. That is a valid, honest result: no robust edge found in")
        print(" this data. Do not loosen the gates to force a result. Try more history, another timeframe,")
        print(" or more generations.")
    else:
        print(f"{'#':>2} {'Symbol':<8}{'TF':<4}{'Family':<15}{'Score':>6}{'Trades':>7}{'PF(hold)':>9}"
              f"{'ExpR':>6}{'Win%':>5}{'Ret%':>6}{'MCdd95':>8}{'Nudge':>6}")
        for i, r in enumerate(final, 1):
            t = r["test"]
            print(f"{i:>2} {r['symbol']:<8}{r['tf']:<4}{r['genome']['fam']:<15}{r['score']:>6.1f}{r['trades']:>7}"
                  f"{t['pf']:>9.2f}{t['exp']:>6.2f}{t['wr']:>5.0f}{t['ret_pct']:>6.1f}"
                  f"{r['mc_dd95']:>7.1f}%{r['nudge'] * 100:>5.0f}%")
        print()
        for i, r in enumerate(final, 1):
            print(f" {i:>2}. {r['symbol']} {r['tf']}  {r['desc']}")
            print(f"     train PF {r['train']['pf']:.2f} | val PF {r['val']['pf']:.2f} | hold-out PF "
                  f"{r['test']['pf']:.2f} ({r['test']['n']} trades) | beat-random {r['beat_random'] * 100:.0f}%")

    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report = dict(generated_at=datetime.now(timezone.utc).isoformat(), applied=False,
                  note="REPORT ONLY. Review, then forward-test on demo. Sealed final data untouched.",
                  settings={k: v for k, v in vars(args).items() if k != "root"},
                  skipped_datasets=skipped, candidates=final)
    jp = rep_dir / "discovery.json"
    jp.write_text(json.dumps(report, indent=1, default=_py), encoding="utf-8")
    if final:
        rows = [dict(rank=i, symbol=r["symbol"], tf=r["tf"], family=r["genome"]["fam"], score=r["score"],
                     trades=r["trades"], pf_holdout=round(r["test"]["pf"], 2), exp_r=round(r["test"]["exp"], 3),
                     win_pct=round(r["test"]["wr"], 1), mc_dd95=r["mc_dd95"], nudge=r["nudge"],
                     rule=r["desc"]) for i, r in enumerate(final, 1)]
        pd.DataFrame(rows).to_csv(rep_dir / "discovery_top.csv", index=False)
    print(f"\nSaved: {jp}\n       {bank_path}  (gene bank seeds the next run)\n       {ledger_path(root)}")
    print("\nNEXT: review the report -> forward-test top picks on DEMO 4-8 weeks -> only then promote one by hand.")
    print("Compare each candidate against your live rule on the same window before replacing anything.")
    return final


if __name__ == "__main__":
    main()
