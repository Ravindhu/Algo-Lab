"""Discovery strategy: engulfing_ocl (BUY and SELL). Self-contained.

This file holds everything for this one strategy: the HTF setup, the LTF trigger, the bar alignment,
and the trade loop. Deleting it does not affect the other discovery strategies.

Shared with live, not copied:
  - algolab.framework.protect.Trade and Protection: stop, breakeven, ATR trail, target, disaster stop.
  - algolab.discovery.indicators: pure indicator math (no strategy logic).

The state transitions and the bar order mirror framework/strategies/ocl.py (OclKeylevel) and
framework/ocl_backtest.py (run_ocl_backtest). tests/test_discovery_engulfing_parity.py checks this
against the live backtest on the same bars.

Setup (HTF, last closed bar): an engulfing with both bodies above min_body_pct, and RSI above EMA(RSI)
on that bar. It arms the level 88.6% into the engulfing candle. The floor is the candle low.
Trigger (LTF, last closed bar): low touches the level, red-red-green on the last 3 LTF bars, and the LTF
Hilega-Milega gate with the oversold cap. The entry is at the open of the next bar, plus half the spread.
Retries: a stop-out steps the level down to that bar's close, up to max_ltf_retries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algolab.discovery.indicators.atr import atr_series
from algolab.discovery.indicators.hilega_milega import bullish_last, direction
from algolab.framework.protect import Protection, Trade


@dataclass(frozen=True)
class EngulfingParams:
    rsi_period: int = 9
    ema_period: int = 3
    min_body_pct: float = 0.5
    fib_pct: float = 0.886
    oversold: float = 30.0
    hm_enabled: bool = True
    ltf_pattern_enabled: bool = True
    target_rr: float = 3.0
    max_ltf_retries: int = 3
    atr_period: int = 14
    atr_multiplier: float = 2.0
    atr_trail_enabled: bool = True
    breakeven_rr: float = 1.5
    round_trip_cost_pct: float = 0.20
    round_trip_cost_pips: float | None = None
    disaster_stop_atr: float = 1.0
    max_spread_atr: float = 0.25
    lookback: int = 120

    def protection(self) -> Protection:
        return Protection(atr_trail_enabled=self.atr_trail_enabled, atr_multiplier=self.atr_multiplier,
                          breakeven_rr=self.breakeven_rr, round_trip_cost_pct=self.round_trip_cost_pct,
                          round_trip_cost_pips=self.round_trip_cost_pips)


def _engulfing_last(o, h, l, c, min_body: float) -> int:
    """+1 bullish engulfing, -1 bearish, 0 none, on the LAST bar of the window."""
    po, pc = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]]
    rng = h - l
    ratio = np.where(rng > 0, np.abs(c - o) / np.where(rng > 0, rng, 1.0), 0.0)
    prev_ratio = np.r_[np.nan, ratio[:-1]]
    big = (ratio > min_body) & (prev_ratio > min_body)
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po) & big
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po) & big
    return int(bull[-1]) - int(bear[-1])


def _run_buy(htf: pd.DataFrame, ltf: pd.DataFrame, prm: EngulfingParams, spread: float, pip: float) -> list[dict]:
    p = prm.protection()
    times = ltf.index
    o, h, l, c = (ltf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ho, hh, hl, hc = (htf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    htf_idx = htf.index
    atr_full = atr_series(h, l, c, prm.atr_period)
    half = spread / 2.0

    st = {"armed": False, "key": None, "htf_sl": None, "level": None, "retries": 0, "used": set()}
    trade: Trade | None = None
    meta: dict = {}
    out: list[dict] = []

    def close(i: int, px: float, reason: str, stop_close: float | None = None) -> None:
        nonlocal trade, meta
        risk = trade.entry - meta["sl"]
        out.append({**meta, "exit_time": times[i], "exit": px, "R": (px - trade.entry) / risk,
                    "reason": reason})
        price = stop_close if stop_close is not None else px
        # Mirrors OclKeylevel.on_exit: a stop-out retries at a lower level, anything else ends the setup.
        if reason == "SL" and st["armed"] and st["retries"] < prm.max_ltf_retries:
            st["level"] = min(st["level"], price)
            st["retries"] += 1
        else:
            if st["key"] is not None:
                st["used"].add(st["key"])
            st["armed"] = False
        trade, meta = None, {}

    def signal(i: int, f: int) -> dict | None:
        lb = prm.lookback
        ls = max(0, i - lb)                     # closed LTF window is [ls, i)
        hs = max(0, f - lb)                     # closed HTF window is [hs, f)
        if i - ls < max(3, prm.atr_period + 2) or f - hs < 1:
            return None
        # Invalidation: the last closed HTF bar closed below the floor.
        if st["armed"] and st["htf_sl"] is not None and hc[f - 1] < st["htf_sl"]:
            st["used"].add(st["key"])
            st["armed"] = False
        if not st["armed"]:
            d = _engulfing_last(ho[hs:f], hh[hs:f], hl[hs:f], hc[hs:f], prm.min_body_pct)
            if d <= 0:
                return None
            if prm.hm_enabled and int(direction(hc[hs:f], prm.rsi_period, prm.ema_period)[-1]) != 1:
                return None
            key = str(htf_idx[f - 1])
            if key in st["used"]:
                return None
            high, low = hh[f - 1], hl[f - 1]
            st.update(armed=True, key=key, htf_sl=low, level=high - prm.fib_pct * (high - low), retries=0)

        bar = i - 1
        level, htf_sl = st["level"], st["htf_sl"]
        if l[bar] > level:                                                # touch of the level
            return None
        if prm.ltf_pattern_enabled and not (c[i - 3] < o[i - 3] and c[i - 2] < o[i - 2]
                                            and c[i - 1] > o[i - 1]):     # red, red, green
            return None
        if prm.hm_enabled and not bullish_last(c[ls:i], prm.rsi_period, prm.ema_period,
                                               oversold=prm.oversold):
            return None
        ref = c[bar]
        if ref <= htf_sl:
            return None
        atr_v = atr_full[bar]
        if np.isnan(atr_v) or atr_v <= 0:
            return None
        broker_sl = htf_sl - prm.disaster_stop_atr * atr_v
        if broker_sl >= ref:
            return None
        tol = 0.005 * abs(level)                  # AUTO exit mode, as in OclKeylevel._resolve_exit
        if ref >= level - tol:
            mode = "FIXED"
        elif ref <= level + tol:
            mode = "DYNAMIC"
        else:
            mode = "FIXED"
        return {"sl_price": broker_sl, "htf_sl": htf_sl, "trigger": level, "exit_mode": mode,
                "tp_dist": prm.target_rr * (ref - htf_sl) if mode == "FIXED" else None,
                "retries": st["retries"], "atr": atr_v}

    for i in range(1, len(ltf)):
        bar = i - 1
        # 1) broker exits during bar i-1
        if trade is not None:
            ev = trade.intrabar_exit(h[bar], l[bar])
            if ev is not None:
                close(bar, trade.broker_sl if ev == "DISASTER" else trade.tp, ev)
        # 2) close-based rules for bar i-1, executed at the open of bar i
        if trade is not None:
            a = None if np.isnan(atr_full[bar]) else float(atr_full[bar])
            ev = trade.on_closed_bar(high=h[bar], close=c[bar], atr=a, p=p)
            if ev is not None:
                close(i, o[i], ev, stop_close=c[bar] if ev == "SL" else None)
        # 3) HTF floor: the last closed HTF bar closed below it
        f = htf_idx.searchsorted(times[i], side="right") - 1
        if trade is not None and f - 1 >= 0 and hc[f - 1] < trade.htf_sl:
            close(i, o[i], "HTF")
        # 4) entry
        if trade is not None or f < 1:
            continue
        sig = signal(i, f)
        if sig is None or spread > prm.max_spread_atr * sig["atr"]:
            continue
        fill = o[i] + half
        broker_sl = sig["sl_price"]
        if fill - broker_sl <= 0:
            continue
        tp = fill + prm.target_rr * (fill - sig["htf_sl"]) if sig["tp_dist"] is not None else None
        trade = Trade.open(entry=fill, trigger=sig["trigger"], htf_sl=sig["htf_sl"], broker_sl=broker_sl,
                           tp=tp, p=p, pip=pip)
        meta = {"entry_time": times[i], "entry": fill, "sl": broker_sl, "tp": tp,
                "htf_sl": sig["htf_sl"], "exit_mode": sig["exit_mode"], "retries": sig["retries"],
                "side": "BUY"}

    if trade is not None:
        close(len(ltf) - 1, float(c[-1]), "EOD")
    return out


def _mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Price mirror for the SELL side: BUY logic on -price. Same mirror the live inverse wrapper uses."""
    return pd.DataFrame({"open": -df["open"], "high": -df["low"], "low": -df["high"],
                         "close": -df["close"]}, index=df.index)


def run(htf: pd.DataFrame, ltf: pd.DataFrame, prm: EngulfingParams | None = None, *,
        spread: float = 0.0, pip: float = 0.0001, side: str = "BOTH") -> pd.DataFrame:
    """Trades for engulfing_ocl. htf and ltf are OHLC frames with DatetimeIndex (open times).
    side: "BUY", "SELL" or "BOTH". R is the trade result in units of the initial risk."""
    prm = prm or EngulfingParams()
    rows: list[dict] = []
    if side in ("BUY", "BOTH"):
        rows += _run_buy(htf, ltf, prm, spread, pip)
    if side in ("SELL", "BOTH"):
        for r in _run_buy(_mirror(htf), _mirror(ltf), prm, spread, pip):
            for key in ("entry", "sl", "tp", "htf_sl", "exit"):
                if r.get(key) is not None:
                    r[key] = -r[key]
            r["side"] = "SELL"
            rows.append(r)
    return pd.DataFrame(rows)
