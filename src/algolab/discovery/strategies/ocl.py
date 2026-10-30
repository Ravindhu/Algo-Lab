"""Discovery strategy: ocl_keylevel (BUY and SELL). Self-contained.

This file holds everything for this one strategy: the HTF setup, the LTF trigger, the bar alignment, and
the trade loop. Deleting it does not affect the other discovery strategies.

Shared with live, not copied:
  - algolab.framework.protect.Trade and Protection: stop, breakeven, ATR trail, target, disaster stop.
  - algolab.discovery.indicators: pure indicator math (no strategy logic).

The state transitions and the bar order mirror framework/strategies/ocl.py (OclKeylevel) and
framework/ocl_backtest.py (run_ocl_backtest). tests/test_discovery_ocl_parity.py checks this against the
live backtest on the same bars.

Setup (HTF, closed bars, newest last): C1 red, C2 red, C3 green. C3 must close above C2's open (C3 break),
or C4 must close above C2's open (C4 confirmation, when allowed). The aggressive level is C2's open (C3
break) or C3's close (C4 confirmation). The conservative level is C3's open. The floor is the lowest low of
C2, C3 and the confirmation candle. The HTF Hilega-Milega gate (RSI above EMA(RSI), no oversold cap) must
hold. The level armed is the conservative one (trigger_level_mode CONSERVATIVE, the live default).
Trigger (LTF): the same as engulfing (touch, red-red-green, oversold cap). Exit mode AUTO: a fill at or above
the aggressive level is a fixed target, at or below the conservative level is dynamic (ATR trail only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algolab.discovery.indicators.atr import atr_series
from algolab.discovery.indicators.hilega_milega import bullish_last
from algolab.framework.protect import Protection, Trade


@dataclass(frozen=True)
class OclParams:
    rsi_period: int = 9
    ema_period: int = 3
    oversold: float = 30.0
    hm_enabled: bool = True
    ltf_pattern_enabled: bool = True
    allow_c4_confirmation: bool = True
    touch_buffer: float = 0.0
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


def _htf_setup_last(o, h, l, c, prm: OclParams) -> dict | None:
    """OCL setup on closed HTF bars (newest last). Returns levels and floor, or None."""
    need = 4 if prm.allow_c4_confirmation else 3
    if len(c) < need:
        return None
    if prm.allow_c4_confirmation:
        c1, c2, c3, c4 = (o[-4], h[-4], l[-4], c[-4]), (o[-3], h[-3], l[-3], c[-3]), \
                         (o[-2], h[-2], l[-2], c[-2]), (o[-1], h[-1], l[-1], c[-1])
    else:
        c1 = (o[-3], h[-3], l[-3], c[-3])
        c2 = (o[-2], h[-2], l[-2], c[-2])
        c3 = (o[-1], h[-1], l[-1], c[-1])
        c4 = None

    def red(b):
        return b[3] < b[0]

    def green(b):
        return b[3] > b[0]

    if not (red(c1) and red(c2) and green(c3)):
        return None
    if c3[3] > c2[0]:                           # C3 break: C3 closes above C2's open
        confirm, name = c3, "C3"
        aggressive = c2[0]
    elif c4 is not None and c4[3] > c2[0]:      # C4 confirmation
        confirm, name = c4, "C4"
        aggressive = c3[3]
    else:
        return None
    return {"htf_sl": min(c2[2], c3[2], confirm[2]), "aggressive": aggressive,
            "conservative": c3[0], "confirm_bar": name}


def _run_buy(htf: pd.DataFrame, ltf: pd.DataFrame, prm: OclParams, spread: float, pip: float) -> list[dict]:
    p = prm.protection()
    times = ltf.index
    o, h, l, c = (ltf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ho, hh, hl, hc = (htf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    htf_idx = htf.index
    atr_full = atr_series(h, l, c, prm.atr_period)
    half = spread / 2.0

    st = {"armed": False, "key": None, "htf_sl": None, "level": None, "aggressive": None,
          "conservative": None, "retries": 0, "used": set()}
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
            setup = _htf_setup_last(ho[hs:f], hh[hs:f], hl[hs:f], hc[hs:f], prm)
            if setup is None:
                return None
            # HTF Hilega-Milega gate, no oversold cap. The live gate returns False on short windows.
            htf_closes = hc[hs:f]
            if prm.hm_enabled and (len(htf_closes) < prm.rsi_period + prm.ema_period + 2 or
                                   not bullish_last(htf_closes, prm.rsi_period, prm.ema_period)):
                return None
            need = 4 if prm.allow_c4_confirmation else 3
            key = str(htf_idx[f - need])
            if key in st["used"]:
                return None
            st.update(armed=True, key=key, htf_sl=setup["htf_sl"], aggressive=setup["aggressive"],
                      conservative=setup["conservative"], level=setup["conservative"], retries=0)

        bar = i - 1
        level, htf_sl = st["level"], st["htf_sl"]
        if l[bar] > level + prm.touch_buffer:                             # touch of the level
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
        # AUTO exit mode, as in OclKeylevel._resolve_exit
        ag, cons = st["aggressive"], st["conservative"]
        if ref >= ag - 0.005 * abs(ag):
            mode = "FIXED"
        elif ref <= cons + 0.005 * abs(cons):
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


def run(htf: pd.DataFrame, ltf: pd.DataFrame, prm: OclParams | None = None, *,
        spread: float = 0.0, pip: float = 0.0001, side: str = "BOTH") -> pd.DataFrame:
    """Trades for ocl_keylevel. htf and ltf are OHLC frames with DatetimeIndex (open times).
    side: "BUY", "SELL" or "BOTH". R is the trade result in units of the initial risk."""
    prm = prm or OclParams()
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
