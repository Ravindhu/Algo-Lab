"""Discovery strategy: ocl_sell_fail_bottom (BUY and SELL). Self-contained.

Everything for this one strategy is in this file: the HTF failed-short setup, the LTF trigger, the bar
alignment, and the trade loop. It does not read, write or depend on any other discovery strategy. In
particular it has NO hand-off from the TOP blueprint: BOTTOM arms any valid core on its own, in its own file.

Shared with live, not copied:
  - algolab.framework.protect.Trade and Protection: stop, breakeven, ATR trail, target, disaster stop.
  - algolab.discovery.indicators: pure indicator math (no strategy logic).

HTF setup (closed bars, newest last, scanned within the last max_window bars):
  core:   C2 green and C3 red, both bodies above min_body_pct_c23 (C1 any colour), newest core first.
  break:  the first later bar whose CLOSE is above C3's open, and no bar in between has reached the
          top of the two sell levels (freshness).
  BOTTOM: one BUY level = the OPEN of the lowest-low candle in the core..fail window (C1 through the break).
          The aggressive reference for exit mode is C3's open.
  floor:  lowest low of C2, C3 and the break bar.
LTF trigger: the same as the other OCL-family strategies (touch, red-red-green on the last three bars,
  oversold Hilega-Milega cap). Entry at the open of the next bar plus half the spread.

Mirrors framework/strategies/ocl_sell_fail.py (OclSellFailBottom) and framework/ocl_backtest.py. The live setup
key uses the core's index inside the window but looks it up in the full closed frame; that quirk is reproduced
here so the two stay identical (see tests/test_discovery_sell_fail_bottom_parity.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algolab.discovery.indicators.atr import atr_series
from algolab.discovery.indicators.hilega_milega import bullish_last
from algolab.framework.protect import Protection, Trade

BLUEPRINT = "BOTTOM"


@dataclass(frozen=True)
class SellFailParams:
    rsi_period: int = 9
    ema_period: int = 3
    oversold: float = 30.0
    hm_enabled: bool = True
    ltf_pattern_enabled: bool = True
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
    min_body_pct_c23: float = 0.5
    max_window: int = 30

    def protection(self) -> Protection:
        return Protection(atr_trail_enabled=self.atr_trail_enabled, atr_multiplier=self.atr_multiplier,
                          breakeven_rr=self.breakeven_rr, round_trip_cost_pct=self.round_trip_cost_pct,
                          round_trip_cost_pips=self.round_trip_cost_pips)


def _core_index(o, h, l, c, prm: SellFailParams) -> int | None:
    """Newest failed-short core: C2 green at i-1, C3 red at i, both bodies above the threshold."""
    n = len(c)
    lo = max(2, n - prm.max_window)
    for i in range(n - 1, lo - 1, -1):
        if not (c[i - 1] > o[i - 1] and c[i] < o[i]):
            continue
        ok = True
        for j in (i - 1, i):
            rng = h[j] - l[j]
            if rng <= 0 or abs(c[j] - o[j]) / rng <= prm.min_body_pct_c23:
                ok = False
                break
        if ok:
            return i
    return None


def _break_index(o, h, l, c, i: int) -> int | None:
    """First later bar whose close is above C3's open. Stale if an earlier bar already traded at the levels."""
    brk = o[i]
    top = max(o[i], o[i - 1])
    for k in range(i + 1, len(c)):
        if c[k] > brk:
            if all(h[m] < top for m in range(i + 1, k)):
                return k
            return None
    return None


def _setup(o, h, l, c, prm: SellFailParams) -> dict | None:
    """Failed-short setup on a window of closed bars (newest last). Indices are window-relative."""
    i = _core_index(o, h, l, c, prm)
    if i is None:
        return None
    k = _break_index(o, h, l, c, i)
    if k is None:
        return None
    start = i - 2                                 # C1
    # Lowest-low candle in core..fail (first one wins on ties, as Python's min does live).
    swing = start + int(np.argmin(l[start:k + 1]))
    return {"core_start": start, "aggressive": o[i], "swing_open": o[swing],
            "fail_low": min(l[i - 1], l[i], l[k])}


def _run_buy(htf: pd.DataFrame, ltf: pd.DataFrame, prm: SellFailParams, spread: float, pip: float) -> list[dict]:
    p = prm.protection()
    times = ltf.index
    o, h, l, c = (ltf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ho, hh, hl, hc = (htf[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    htf_idx = htf.index
    atr_full = atr_series(h, l, c, prm.atr_period)
    half = spread / 2.0

    st = {"armed": False, "key": None, "htf_sl": None, "level": None,
          "aggressive": None, "conservative": None, "retries": 0, "used": set()}
    trade: Trade | None = None
    meta: dict = {}
    out: list[dict] = []

    def retire() -> None:
        if st["key"] is not None:
            st["used"].add(st["key"])
        st["armed"] = False

    def close(i: int, px: float, reason: str, stop_close: float | None = None) -> None:
        nonlocal trade, meta
        risk = trade.entry - meta["sl"]
        out.append({**meta, "exit_time": times[i], "exit": px, "R": (px - trade.entry) / risk,
                    "reason": reason})
        # BOTTOM has one level: a stop-out steps the level down to the stop bar's close, up to the retry cap.
        if reason != "SL" or not st["armed"]:
            retire()
        elif st["retries"] < prm.max_ltf_retries:
            price = stop_close if stop_close is not None else px
            st["level"] = min(st["level"], price)
            st["retries"] += 1
        else:
            retire()
        trade, meta = None, {}

    def signal(i: int, f: int) -> dict | None:
        lb = prm.lookback
        ls = max(0, i - lb)                       # closed LTF window is [ls, i)
        hs = max(0, f - lb)                       # closed HTF window is [hs, f)
        if i - ls < max(3, prm.atr_period + 2) or f - hs < 3:
            return None
        if st["armed"] and st["htf_sl"] is not None and hc[f - 1] < st["htf_sl"]:
            retire()                               # HTF close below the floor invalidates the setup
        if not st["armed"]:
            w = min(f - hs, prm.max_window)
            ws = f - w                             # window of the last max_window closed bars
            setup = _setup(ho[ws:f], hh[ws:f], hl[ws:f], hc[ws:f], prm)
            if setup is None:
                return None
            # Live quirk, reproduced: the key takes the core index inside the window and looks it up in the
            # full closed frame, which starts at hs.
            key = str(htf_idx[hs + setup["core_start"]])
            if key in st["used"]:
                return None
            if prm.hm_enabled:
                closes = hc[hs:f]
                if len(closes) < prm.rsi_period + prm.ema_period + 2 or \
                        not bullish_last(closes, prm.rsi_period, prm.ema_period):
                    return None
            # BOTTOM level: the swing candle's open. Exit-mode reference levels: aggressive = C3 open,
            # conservative = swing open (as the live _levels returns for BOTTOM).
            st.update(armed=True, key=key, htf_sl=setup["fail_low"], level=setup["swing_open"],
                      aggressive=setup["aggressive"], conservative=setup["swing_open"], retries=0)

        bar = i - 1
        level, htf_sl = st["level"], st["htf_sl"]
        if l[bar] > level + prm.touch_buffer:
            return None
        if prm.ltf_pattern_enabled and not (c[i - 3] < o[i - 3] and c[i - 2] < o[i - 2]
                                            and c[i - 1] > o[i - 1]):
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
        if trade is not None:
            ev = trade.intrabar_exit(h[bar], l[bar])
            if ev is not None:
                close(bar, trade.broker_sl if ev == "DISASTER" else trade.tp, ev)
        if trade is not None:
            a = None if np.isnan(atr_full[bar]) else float(atr_full[bar])
            ev = trade.on_closed_bar(high=h[bar], close=c[bar], atr=a, p=p)
            if ev is not None:
                close(i, o[i], ev, stop_close=c[bar] if ev == "SL" else None)
        f = htf_idx.searchsorted(times[i], side="right") - 1
        if trade is not None and f - 1 >= 0 and hc[f - 1] < trade.htf_sl:
            close(i, o[i], "HTF")
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
                "side": "BUY", "blueprint": BLUEPRINT}

    if trade is not None:
        close(len(ltf) - 1, float(c[-1]), "EOD")
    return out


def _mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Price mirror for the SELL side: BUY logic on -price. Same mirror the live inverse wrapper uses."""
    return pd.DataFrame({"open": -df["open"], "high": -df["low"], "low": -df["high"],
                         "close": -df["close"]}, index=df.index)


def run(htf: pd.DataFrame, ltf: pd.DataFrame, prm: SellFailParams | None = None, *,
        spread: float = 0.0, pip: float = 0.0001, side: str = "BOTH") -> pd.DataFrame:
    """Trades for ocl_sell_fail_bottom. side: "BUY", "SELL" or "BOTH". R is in units of the initial risk."""
    prm = prm or SellFailParams()
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
