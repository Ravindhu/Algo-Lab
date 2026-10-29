# OCL SELL-FAILURE blueprints (ported from ocl_keylevel.py setup_mode
# SELL_FAILURE_TOP / SELL_FAILURE_BOTTOM). Two separate plug-and-play strategies;
# the normal OCL BUY blueprint stays in ocl.py untouched.
#
# ALL OCL blueprints trade BUY. The "sell" in the name is the SETUP that fails:
#
#   HTF core (the failed short): C1 any colour, C2 GREEN (its open is the
#   conservative SELL level), C3 RED (its open is the aggressive SELL level),
#   >50% bodies on C2 and C3. A later closed candle that CLOSES ABOVE the
#   aggressive sell level (C3 open) breaks the pattern -> levels FLIP to BUY
#   levels and the entry waits for the RETEST from above.
#
#   TOP   (2 key levels): aggressive BUY = broken C3 open (tier 1),
#         conservative BUY = C2 open (tier 2). Tier 2 arms when tier 1's
#         LTF stop is hit (up to max_ltf_retries entries overall).
#   BOTTOM (1 key level): single BUY level = OPEN of the LOWEST SWING candle
#         in the core..failure window; the stop is that candle's LOW.
#
# Freshness: no candle between C3 and the failure candle may have already
# traded at/above the sell levels (high >= max(C2 open, C3 open)) - the break
# must be the FIRST test of the level.
#
# Everything downstream of the signal is shared with the normal OCL:
# OclParams (HM gates, touch buffer, exit_mode AUTO, target_rr, disaster stop,
# protection) and protect.Trade via the live position manager. All three
# blueprints are BUY setups, so the existing BUY-only trade management applies
# unchanged.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from algolab.framework.protect import Protection
from algolab.framework.strategies.multi import MultiTimeframeStrategy, closed
from algolab.framework.strategies.ocl import (
    EXIT_FIXED,
    EXIT_AUTO,
    Bar,
    OclParams,
    _bars,
    hm_bullish,
    map_timeframe,
)
from algolab.hilega_milega import atr as atr_series
from algolab.hilega_milega import hilega_milega_rsi

log = logging.getLogger("algolab.ocl_sell_fail")


@dataclass(frozen=True)
class SellFailParams(OclParams):
    # OclParams fields plus the failure-blueprint specifics.
    min_body_pct_c23: float = 0.5   # C2 and C3 bodies must exceed this body/range ratio
    max_window: int = 30            # HTF bars back to scan for the failed-short core

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.min_body_pct_c23 <= 1.0:
            raise ValueError("min_body_pct_c23 must be within [0, 1]")
        if self.max_window < 5:
            raise ValueError("max_window must be >= 5 HTF bars")


def _sell_core(bars: list[Bar], p: SellFailParams) -> tuple[int, int] | None:
    """Newest failed-short core: index i where bars[i-2] any, bars[i-1] GREEN,
    bars[i] RED, both >body. Returns (i, k) with k > i the BREAK candle, or None.
    Scanned newest-first so the freshest valid setup wins."""
    lo = max(2, len(bars) - p.max_window)
    for i in range(len(bars) - 1, lo - 1, -1):
        c2, c3 = bars[i - 1], bars[i]
        if not (c2.green and c3.red):
            continue
        for c in (c2, c3):                      # >body filter on C2 and C3
            rng = c.high - c.low
            if rng <= 0 or abs(c.close - c.open) / rng <= p.min_body_pct_c23:
                break
        else:
            return i
    return None


def _break_index(bars: list[Bar], i: int, p: SellFailParams) -> int | None:
    """First candle after the core whose CLOSE breaks the aggressive sell level."""
    break_level = bars[i].open                  # C3 open = aggressive SELL level
    cons = bars[i - 1].open                     # C2 open = conservative SELL level
    top = max(break_level, cons)
    for k in range(i + 1, len(bars)):
        if bars[k].close > break_level:
            # Freshness: anything BETWEEN the core and the break that already
            # traded at/above the levels would have consumed them. The break
            # candle itself is excluded - it is the break by definition.
            if all(bars[m].high < top for m in range(i + 1, k)):
                return k
            return None                          # stale: level already tested
    return None


def _break_setup(bars: list[Bar], p: SellFailParams) -> dict[str, Any] | None:
    """Full failed-short setup: core indices, break candle and flipped levels."""
    i = _sell_core(bars, p)
    if i is None:
        return None
    k = _break_index(bars, i, p)
    if k is None:
        return None
    c2, c3, fail = bars[i - 1], bars[i], bars[k]
    aggressive = c3.open                        # broken aggressive SELL level
    conservative = c2.open                      # conservative SELL level
    return {
        "core_start": i - 2,                    # index of C1 in bars
        "aggressive": aggressive,               # TOP tier 1
        "conservative": conservative,           # TOP tier 2
        "fail_high": max(c2.high, c3.high, fail.high),
        "fail_low": min(c2.low, c3.low, fail.low),
        "fail_index": k,
        "key": str(bars[i - 2].open),           # coarse identity of the core window
        "bars": bars,                           # needed by BOTTOM's swing search
    }


class HandoffStore:
    """Small shared file recording which failed-short cores TOP has exhausted.
    TOP writes "exhausted"; BOTTOM refuses to arm a core it has not seen there.
    Plain JSON at state/framework_handoff.json so it survives restarts."""

    FILE = "framework_handoff.json"

    def __init__(self, read=None, write=None, scope: str = ""):
        from algolab.ops.state import read_json, write_json
        self._read = read or read_json
        self._write = write or write_json
        self._scope = scope          # "" for the BUY blueprints, "SELL:" for the inverted ones

    def exhausted(self, symbol: str, key: str) -> bool:
        rec = self._read(self.FILE, {})
        entry = rec.get(self._scope + symbol, {})
        return entry.get("key") == key and entry.get("state") == "exhausted"

    def mark_exhausted(self, symbol: str, key: str) -> None:
        if not key:
            return
        rec = self._read(self.FILE, {})
        rec[self._scope + symbol] = {"key": key, "state": "exhausted"}
        self._write(self.FILE, rec)


def _lowest_swing(bars: list[Bar], start: int, end: int) -> Bar:
    """The candle with the LOWEST low in bars[start..end] (core..failure window)."""
    window = bars[start:end + 1]
    return min(window, key=lambda b: b.low)


class OclSellFailBase(MultiTimeframeStrategy):
    """Shared engine for the two sell-failure blueprints. Subclasses set
    `blueprint` ("TOP" or "BOTTOM"); the level bookkeeping differs only there."""

    blueprint = "TOP"

    def __init__(self, symbol, htf="15minute", ltf="5minute", count=None, **params):
        self.htf_tf = map_timeframe(htf)
        self.ltf_tf = map_timeframe(ltf)
        self.p = SellFailParams.from_dict(params)
        super().__init__(symbol, self.ltf_tf, count or self.p.lookback,
                         htf=self.htf_tf, ltf=self.ltf_tf, **params)
        self.timeframes = (self.htf_tf, self.ltf_tf)
        # Same state shape as the normal OCL: armed pattern, trigger level,
        # floor, retry ladder. level2 is TOP's conservative tier.
        self.st = self._new_state()
        if not self.p.atr_trail_enabled:
            log.warning("[%s] ATR trail disabled; DYNAMIC exits then rely on the LTF stop only",
                        self.symbol.name)

    # ---- subclass hooks --------------------------------------------------
    def _new_state(self) -> dict[str, Any]:
        return {"armed": False, "key": None, "htf_sl": None, "level": None,
                "level2": None, "aggressive": None, "conservative": None,
                "retries": 0, "in_trade": False, "used": []}

    def _levels(self, setup: dict[str, Any], bars: list[Bar]) -> tuple[float, float, float, float]:
        """(trigger, level2, aggressive, conservative) for this blueprint."""
        raise NotImplementedError

    # ---- state persistence (caller saves/restores; same contract as OCL) --
    def state(self) -> dict[str, Any]:
        return dict(self.st, used=list(self.st["used"]))

    def restore(self, d: dict[str, Any]) -> None:
        known = set(self._new_state())
        self.st = self._new_state() | {k: v for k, v in d.items() if k in known}

    # ---- signal ----------------------------------------------------------
    def signal_frames(self, frames: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
        p = self.p
        htf = closed(frames[self.htf_tf])
        ltf = closed(frames[self.ltf_tf])
        if len(ltf) < max(3, p.atr_period + 2) or len(htf) < 3:
            return None
        st = self.st

        htf_last = _bars(htf, 1)[0]
        if st["armed"] and st["htf_sl"] is not None and htf_last.close < st["htf_sl"]:
            log.info("[%s] HTF close %.5f below floor %.5f; setup invalidated",
                     self.symbol.name, htf_last.close, st["htf_sl"])
            self._retire()
        if st["in_trade"]:
            return None

        if not st["armed"]:
            bars = _bars(htf, min(len(htf), p.max_window))
            setup = _break_setup(bars, p)
            if setup is None:
                return None
            key = str(htf.index[setup["core_start"]])
            if key in st["used"]:
                return None
            if p.hm_enabled and not hm_bullish([float(x) for x in htf["close"]], p,
                                               oversold_cap=False):
                return None
            level, level2, aggr, cons = self._levels(setup, bars)
            st.update(armed=True, key=key, level=level, level2=level2,
                      aggressive=aggr, conservative=cons,
                      htf_sl=setup["fail_low"], retries=0)
            log.info("[%s %s] armed %s: level %.5f, tier2 %.5f, floor %.5f",
                     self.symbol.name, self.blueprint, key, level, level2,
                     st["htf_sl"])

        last = _bars(ltf, 1)[0]
        buffer = p.touch_buffer + p.touch_buffer_pips * self.symbol.pip
        if last.low > st["level"] + buffer:
            return None
        b1, b2, b3 = _bars(ltf, 3)                  # RRG on the last 3 closed LTF bars
        if not (b1.red and b2.red and b3.green):
            return None
        if p.hm_enabled and not hm_bullish([float(x) for x in ltf["close"]], p,
                                           oversold_cap=True):
            return None

        ref = last.close
        if ref <= st["htf_sl"]:
            return None
        atr_v = self._ltf_atr(ltf)
        if atr_v is None or atr_v <= 0:
            return None
        broker_sl = st["htf_sl"] - p.disaster_stop_atr * atr_v
        if broker_sl >= ref:
            return None
        exit_mode = self._resolve_exit(ref)
        tp_dist = p.target_rr * (ref - st["htf_sl"]) if exit_mode == EXIT_FIXED else None
        return {
            "side": "BUY",
            "sl_price": broker_sl,                  # broker stop level; the sizer uses this
            "sl_dist": ref - broker_sl,
            "tp_dist": tp_dist,                     # None for DYNAMIC
            "htf_sl": st["htf_sl"],
            "target_rr": p.target_rr,
            "trigger": st["level"],
            "aggressive": st["aggressive"],
            "conservative": st["conservative"],
            "exit_mode": exit_mode,
            "entry_ref": ref,
            "retries": st["retries"],
            "atr": atr_v,
            "atr_period": p.atr_period,
            "protect": p.protection.to_dict(),
            "max_spread_atr": p.max_spread_atr,
            "time": str(ltf.index[-1]),
        }

    # ---- lifecycle (same ladder rules as the normal OCL) -----------------
    def on_entry(self) -> None:
        self.st["in_trade"] = True

    def on_exit(self, reason: str, price: float) -> None:
        st = self.st
        st["in_trade"] = False
        if reason != "SL" or not st["armed"]:
            self._retire()
            return
        if self.blueprint == "TOP" and st["level2"] is not None \
                and st["level"] == st["aggressive"]:
            # Tier 1 failed -> arm tier 2 (the conservative BUY level).
            st["level"] = st["level2"]
            st["retries"] += 1
            log.info("[%s] tier-1 stop-out; tier 2 armed at %.5f (retry %d)",
                     self.symbol.name, st["level"], st["retries"])
            return
        if st["retries"] < self.p.max_ltf_retries:
            st["level"] = min(st["level"], price)   # cheaper re-entry, same ladder as OCL
            st["retries"] += 1
            log.info("[%s] stop-out; retry %d/%d at %.5f", self.symbol.name,
                     st["retries"], self.p.max_ltf_retries, st["level"])
            return
        self._retire()

    def _retire(self) -> None:
        st = self.st
        if st["key"] is not None and st["key"] not in st["used"]:
            st["used"].append(st["key"])
        st["armed"] = False

    # ---- helpers (identical rules to the normal OCL) ---------------------
    def _resolve_exit(self, ref: float) -> str:
        if self.p.exit_mode != EXIT_AUTO:
            return self.p.exit_mode
        st = self.st
        if st["aggressive"] is not None and ref >= st["aggressive"] - 0.005 * abs(st["aggressive"]):
            return EXIT_FIXED
        if st["conservative"] is not None and ref <= st["conservative"] + 0.005 * abs(st["conservative"]):
            return "DYNAMIC"
        return EXIT_FIXED

    def _ltf_atr(self, ltf: pd.DataFrame) -> float | None:
        highs = [float(x) for x in ltf["high"]]
        lows = [float(x) for x in ltf["low"]]
        closes = [float(x) for x in ltf["close"]]
        return atr_series(highs, lows, closes, period=self.p.atr_period)[-1]


class OclSellFailTop(OclSellFailBase):
    """SELL_FAILURE_TOP: two key levels. Tier 1 = broken C3 open, tier 2 = C2 open."""

    name = "ocl_sell_fail_top"
    blueprint = "TOP"
    handoff_scope = ""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._handoff = HandoffStore(scope=self.handoff_scope)

    def _levels(self, setup: dict[str, Any], bars: list[Bar]) -> tuple[float, float, float, float]:
        # Tier 1 (aggressive BUY) = the broken C3 open; tier 2 (conservative) = C2 open.
        return setup["aggressive"], setup["conservative"], setup["aggressive"], setup["conservative"]

    def _retire(self) -> None:
        super()._retire()
        # Hand-off: once TOP has nothing left for this core, BOTTOM may arm it.
        self._handoff.mark_exhausted(self.symbol.name, self.st.get("key") or "")


class OclSellFailBottom(OclSellFailBase):
    """SELL_FAILURE_BOTTOM: one key level - the open of the lowest swing candle
    in the core..failure window; the stop is that candle's LOW.
    Runs only after TOP has exhausted the same failed-short core (hand-off)."""

    name = "ocl_sell_fail_bottom"
    blueprint = "BOTTOM"
    handoff_scope = ""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._handoff = HandoffStore(scope=self.handoff_scope)

    def _levels(self, setup: dict[str, Any], bars: list[Bar]) -> tuple[float, float, float, float]:
        swing = _lowest_swing(bars, setup["core_start"], setup["fail_index"])
        # Single BUY level = the swing candle's OPEN; the floor (st["htf_sl"])
        # is set from fail_low, which cannot be above the swing candle's low.
        return swing.open, swing.open, setup["aggressive"], swing.open

    def _new_state(self) -> dict[str, Any]:
        st = super()._new_state()
        st["htf_sl_override"] = None
        return st

    def signal_frames(self, frames: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
        # Hand-off gate: BOTTOM may only arm a core that TOP has marked exhausted.
        p = self.p
        htf = closed(frames[self.htf_tf])
        if len(htf) >= 3 and not self.st["armed"] and not self.st["in_trade"]:
            bars = _bars(htf, min(len(htf), p.max_window))
            setup = _break_setup(bars, p)
            if setup is None:
                return None
            key = str(htf.index[setup["core_start"]])
            if not self._handoff.exhausted(self.symbol.name, key):
                return None
        return super().signal_frames(frames)


# Plugin registry exports (framework_runner loads strategies by name through this).
STRATEGY = [OclSellFailTop, OclSellFailBottom]
