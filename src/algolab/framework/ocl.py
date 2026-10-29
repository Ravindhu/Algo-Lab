"""OCL keylevel strategy, ported from ocl_keylevel.py (NORMAL OCL setup, BUY only, AUTO exit mode).

Naming: RRG = Red, Red, Green. A red candle closes below its open; a green candle closes above it.

HTF (higher timeframe), last closed candles C1..C4 (C4 newest):
  - C1 red, C2 red, C3 green.
  - C3 closes above C2's open (C3 break), or C4 closes above C2's open (C4 confirmation, if enabled).
  - Aggressive level  = C2 open (C3 break) or C3 close (C4 confirmation).
  - Conservative level = C3 open.
  - HTF floor = min(low C2, low C3, low confirmation candle).
  - HTF Hilega-Milega gate: RSI above EMA(RSI). No oversold cap on the HTF (the operator's default).
  - An HTF close below the floor invalidates the pattern.
LTF (lower timeframe), last closed bar and the 3 before it:
  - Touch: last closed low <= trigger level + touch buffer (level = aggressive or conservative per mode).
  - RRG on the last 3 closed LTF bars.
  - LTF Hilega-Milega gate: RSI above EMA(RSI) AND RSI <= oversold (default 30).
Exit mode AUTO: entry at the aggressive level -> FIXED (fixed-RR target); at the conservative level ->
  DYNAMIC (no target; the ATR trail runs the trade). Exit rules live in protect.py.
Retry ladder: an LTF stop-out (close below the LTF stop) steps the trigger down to that bar's close and
  retries, up to max_ltf_retries. Then the pattern is abandoned. TP, trail, HTF and disaster exits end it.

Deviations from the original, all explicit:
  - touch_buffer is price units in the original (0.05, a stock-price setting). Forex config uses
    touch_buffer_pips, converted with the symbol's pip size.
  - Disaster stop: a broker-side stop at HTF floor minus disaster_stop_atr * ATR. Not in the original.
  - Not ported: DV-ATR trail (on by default in the original), confirmation layer (off by default),
    band halt, SELL_FAILURE blueprints (off by default), execution-mode variants other than one-shot per pattern.

Pure rules: signal_frames() reads frames and this object's own state. The caller persists state
(state()/restore()) and reports fills and exits (on_entry/on_exit). The strategy never touches the broker.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import pandas as pd

from algolab.framework.multi import MultiTimeframeStrategy, closed
from algolab.framework.protect import Protection
from algolab.framework.timeframe import TimeFrame
from algolab.hilega_milega import atr as atr_series
from algolab.hilega_milega import ema, hilega_milega_rsi

log = logging.getLogger("algolab.ocl")

AGGRESSIVE = "AGGRESSIVE"
CONSERVATIVE = "CONSERVATIVE"
EXIT_AUTO, EXIT_FIXED, EXIT_DYNAMIC = "AUTO", "FIXED", "DYNAMIC"

# Timeframe names used in ocl_keylevel.py -> algolab names. algolab names also work.
SOURCE_TF = {"1minute": "M1", "5minute": "M5", "15minute": "M15", "30minute": "M30",
             "60minute": "H1", "day": "D1"}
_META_KEYS = frozenset({"enabled", "reason", "version", "updated"})


def map_timeframe(name: str) -> str:
    key = str(name).strip()
    if key in SOURCE_TF:
        return SOURCE_TF[key]
    try:
        return TimeFrame(key.upper()).value
    except ValueError as e:
        raise ValueError(f"unknown timeframe {name!r}; use {sorted(SOURCE_TF)} or M1..D1") from e


class Bar(NamedTuple):
    open: float
    high: float
    low: float
    close: float

    @property
    def red(self) -> bool:
        return self.close < self.open

    @property
    def green(self) -> bool:
        return self.close > self.open


def _bars(df: pd.DataFrame, n: int) -> list[Bar]:
    rows = df[["open", "high", "low", "close"]].iloc[-n:].to_numpy(dtype=float)
    return [Bar(*map(float, r)) for r in rows]


@dataclass(frozen=True)
class OclParams:
    rsi_period: int = 9
    ema_period: int = 3
    oversold: float = 30.0               # LTF cap only; the HTF gate has none
    hm_enabled: bool = True
    ltf_pattern_enabled: bool = True     # RRG on the last 3 closed LTF bars
    allow_c4_confirmation: bool = True
    trigger_level_mode: str = CONSERVATIVE
    touch_buffer: float = 0.0            # price units (original default 0.05 is a stock setting)
    touch_buffer_pips: float = 0.0       # converted with the symbol's pip size
    target_rr: float = 3.0
    exit_mode: str = EXIT_AUTO
    max_ltf_retries: int = 3
    atr_period: int = 14
    lookback: int = 120                  # closed bars per frame; enough for RSI/EMA/ATR warm-up
    disaster_stop_atr: float = 1.0       # broker stop = HTF floor - this * ATR (safety, not in the original)
    max_spread_atr: float = 0.25         # skip entries whose spread exceeds this fraction of ATR
    # Protection (see protect.py)
    atr_trail_enabled: bool = True
    atr_multiplier: float = 2.0
    breakeven_rr: float = 1.5
    round_trip_cost_pct: float = 0.20

    def __post_init__(self) -> None:
        if self.trigger_level_mode not in (CONSERVATIVE, AGGRESSIVE):
            raise ValueError("trigger_level_mode must be CONSERVATIVE or AGGRESSIVE")
        if self.exit_mode not in (EXIT_AUTO, EXIT_FIXED, EXIT_DYNAMIC):
            raise ValueError("exit_mode must be AUTO, FIXED or DYNAMIC")
        if self.rsi_period < 2 or self.ema_period < 1 or self.atr_period < 2:
            raise ValueError("rsi_period >= 2, ema_period >= 1, atr_period >= 2")
        if not 0.0 <= self.oversold <= 100.0:
            raise ValueError("oversold must be within [0, 100]")
        if min(self.touch_buffer, self.touch_buffer_pips, self.max_spread_atr, self.disaster_stop_atr) < 0:
            raise ValueError("buffers, max_spread_atr and disaster_stop_atr must be >= 0")
        if self.target_rr <= 0:
            raise ValueError("target_rr must be > 0")
        if self.max_ltf_retries < 0:
            raise ValueError("max_ltf_retries must be >= 0")
        if self.lookback < 30:
            raise ValueError("lookback must be >= 30 bars")
        Protection(self.atr_trail_enabled, self.atr_multiplier, self.breakeven_rr, self.round_trip_cost_pct)

    @property
    def protection(self) -> Protection:
        return Protection(self.atr_trail_enabled, self.atr_multiplier, self.breakeven_rr,
                          self.round_trip_cost_pct)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "OclParams":
        known = set(cls.__dataclass_fields__)
        unknown = set(cfg) - known - _META_KEYS - {"htf", "ltf"}
        if unknown:
            raise ValueError(f"unknown OCL parameter(s): {sorted(unknown)}")
        return cls(**{k: v for k, v in cfg.items() if k in known})


def hm_bullish(closes: list[float], p: OclParams, *, oversold_cap: bool) -> bool:
    """RSI above EMA(RSI). With oversold_cap, RSI must also be at or below p.oversold."""
    if len(closes) < p.rsi_period + p.ema_period + 2:
        return False
    rsi = hilega_milega_rsi(closes, rsi_period=p.rsi_period)
    line = ema(rsi, p.ema_period)
    i = min(len(rsi), len(line)) - 1
    r, e = rsi[i], line[i]
    if r is None or e is None or r <= e:
        return False
    return (r <= p.oversold) if oversold_cap else True


def htf_setup(htf_closed: pd.DataFrame, p: OclParams) -> dict[str, Any] | None:
    """OCL blueprint on CLOSED HTF bars (newest last)."""
    need = 4 if p.allow_c4_confirmation else 3
    if len(htf_closed) < need:
        return None
    bars = _bars(htf_closed, need)
    if p.allow_c4_confirmation:
        c1, c2, c3, c4 = bars
    else:
        (c1, c2, c3), c4 = bars, None
    if not (c1.red and c2.red and c3.green):
        return None
    if c3.close > c2.open:
        confirm, name = c3, "C3"
    elif c4 is not None and c4.close > c2.open:
        confirm, name = c4, "C4"
    else:
        return None
    return {
        "aggressive": c2.open if name == "C3" else c3.close,
        "conservative": c3.open,
        "htf_sl": min(c2.low, c3.low, confirm.low),
        "confirm_bar": name,
    }


@dataclass
class SetupState:
    """Everything the strategy remembers between cycles. Plain types, so it serialises to JSON."""
    armed: bool = False
    key: str | None = None               # identity of the armed pattern: time of its first candle
    htf_sl: float | None = None
    level: float | None = None           # current trigger; steps DOWN after each stop-out
    aggressive: float | None = None
    conservative: float | None = None
    confirm_bar: str | None = None
    retries: int = 0
    in_trade: bool = False
    used: list[str] = field(default_factory=list)


class OclKeylevel(MultiTimeframeStrategy):
    name = "ocl_keylevel"

    def __init__(self, symbol, htf="15minute", ltf="5minute", count=None, **params):
        self.htf_tf = map_timeframe(htf)
        self.ltf_tf = map_timeframe(ltf)
        self.p = OclParams.from_dict(params)
        super().__init__(symbol, self.ltf_tf, count or self.p.lookback,
                         htf=self.htf_tf, ltf=self.ltf_tf, **params)
        self.timeframes = (self.htf_tf, self.ltf_tf)
        self.st = SetupState()
        if not self.p.atr_trail_enabled:
            log.warning("[%s] ATR trail disabled; DYNAMIC exits then rely on the LTF stop only",
                        self.symbol.name)

    # ---- state persistence (the caller saves and restores; the strategy never touches disk) ----
    def state(self) -> dict[str, Any]:
        return dict(self.st.__dict__, used=list(self.st.used))

    def restore(self, d: dict[str, Any]) -> None:
        known = set(SetupState.__dataclass_fields__)
        self.st = SetupState(**{k: v for k, v in d.items() if k in known})

    # ---- signal ----
    def signal_frames(self, frames: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
        p = self.p
        htf = closed(frames[self.htf_tf])
        ltf = closed(frames[self.ltf_tf])
        need_ltf = max(3 if p.ltf_pattern_enabled else 1, p.atr_period + 2)
        if len(ltf) < need_ltf or len(htf) < 1:
            return None

        st = self.st
        htf_last = _bars(htf, 1)[0]
        if st.armed and st.htf_sl is not None and htf_last.close < st.htf_sl:
            log.info("[%s] HTF close %.5f below floor %.5f; pattern invalidated",
                     self.symbol.name, htf_last.close, st.htf_sl)
            if st.key is not None and st.key not in st.used:
                st.used.append(st.key)              # dead until a NEW HTF pattern forms
            st.armed = False
        if st.in_trade:
            return None

        if not st.armed:
            setup = htf_setup(htf, p)
            if setup is None:
                return None
            if p.hm_enabled and not hm_bullish([float(x) for x in htf["close"]], p, oversold_cap=False):
                return None
            need = 4 if p.allow_c4_confirmation else 3
            key = str(htf.index[-need])
            if key in st.used:
                return None
            st.armed, st.key = True, key
            st.htf_sl = setup["htf_sl"]
            st.aggressive, st.conservative = setup["aggressive"], setup["conservative"]
            st.confirm_bar = setup["confirm_bar"]
            st.level = st.aggressive if p.trigger_level_mode == AGGRESSIVE else st.conservative
            st.retries = 0
            log.info("[%s] armed %s: level %.5f, floor %.5f", self.symbol.name, key, st.level, st.htf_sl)

        last = _bars(ltf, 1)[0]
        buffer = p.touch_buffer + p.touch_buffer_pips * self.symbol.pip
        if last.low > st.level + buffer:
            return None
        if p.ltf_pattern_enabled:
            b1, b2, b3 = _bars(ltf, 3)
            if not (b1.red and b2.red and b3.green):
                return None
        if p.hm_enabled and not hm_bullish([float(x) for x in ltf["close"]], p, oversold_cap=True):
            return None

        ref = last.close
        if ref <= st.htf_sl:
            return None
        atr_v, rsi_v = self._ltf_atr_rsi(ltf)
        if atr_v is None or atr_v <= 0:
            return None
        broker_sl = st.htf_sl - p.disaster_stop_atr * atr_v
        if broker_sl >= ref:
            return None
        exit_mode = self._resolve_exit(ref)
        tp_dist = p.target_rr * (ref - st.htf_sl) if exit_mode == EXIT_FIXED else None
        return {
            "side": "BUY",
            "sl_price": broker_sl,               # broker stop level; the sizer uses this
            "sl_dist": ref - broker_sl,
            "tp_dist": tp_dist,                  # None for DYNAMIC
            "htf_sl": st.htf_sl,
            "target_rr": p.target_rr,
            "trigger": st.level,
            "aggressive": st.aggressive,
            "conservative": st.conservative,
            "exit_mode": exit_mode,
            "entry_ref": ref,
            "confirm_bar": st.confirm_bar,
            "retries": st.retries,
            "atr": atr_v,
            "atr_period": p.atr_period,
            "rsi": rsi_v,
            "protect": p.protection.to_dict(),
            "max_spread_atr": p.max_spread_atr,
            "time": str(ltf.index[-1]),
        }

    def exit_check(self, frames: dict[str, pd.DataFrame]) -> bool:
        """True when the last closed HTF bar closed below the floor of an open trade."""
        if not self.st.in_trade or self.st.htf_sl is None:
            return False
        htf = closed(frames[self.htf_tf])
        return len(htf) > 0 and _bars(htf, 1)[0].close < self.st.htf_sl

    def on_entry(self) -> None:
        self.st.in_trade = True

    def on_exit(self, reason: str, price: float) -> None:
        """reason: SL (LTF stop, retry), TRAIL, TP, HTF, DISASTER, BROKER_CLOSE, PROP, WEEKEND, EOD.
        price for SL is the close of the bar that closed below the stop (the ladder uses it)."""
        st = self.st
        st.in_trade = False
        if reason == "SL" and st.armed and st.retries < self.p.max_ltf_retries:
            st.level = min(st.level, price)
            st.retries += 1
            log.info("[%s] LTF stop-out; retry %d/%d at %.5f", self.symbol.name,
                     st.retries, self.p.max_ltf_retries, st.level)
            return
        if st.key is not None and st.key not in st.used:
            st.used.append(st.key)
        st.armed = False

    # ---- helpers ----
    def _resolve_exit(self, ref: float) -> str:
        if self.p.exit_mode != EXIT_AUTO:
            return self.p.exit_mode
        st = self.st
        if ref >= st.aggressive * 0.995:
            return EXIT_FIXED
        if ref <= st.conservative * 1.005:
            return EXIT_DYNAMIC
        return EXIT_FIXED

    def _ltf_atr_rsi(self, ltf: pd.DataFrame) -> tuple[float | None, float | None]:
        highs = [float(x) for x in ltf["high"]]
        lows = [float(x) for x in ltf["low"]]
        closes = [float(x) for x in ltf["close"]]
        a = atr_series(highs, lows, closes, period=self.p.atr_period)[-1]
        r = hilega_milega_rsi(closes, rsi_period=self.p.rsi_period)[-1]
        return a, r
