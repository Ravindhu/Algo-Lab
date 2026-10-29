"""Stop and exit rules for an open OCL trade. Pure: the backtest and the live position manager call the same code.

Rules follow ocl_keylevel.py (its defaults are in Protection and OclParams):
  - LTF stop : a CLOSED LTF bar closes below the stop          -> "SL"    (the entry ladder retries)
  - ATR trail: a CLOSED LTF bar closes below the trail         -> "TRAIL" (pattern finished)
  - HTF stop : a CLOSED HTF bar closes below the HTF floor     -> "HTF"   (checked by the caller)
  - Target   : a bar's high reaches the fixed target           -> "TP"    (intrabar, broker take-profit)
  - Disaster : a bar's low reaches the broker stop             -> "DISASTER" (intrabar, broker stop)
Breakeven: when price reaches trigger + risk * breakeven_rr, the LTF stop moves up to
  trigger * (1 + round_trip_cost_pct / 100). It only moves up. risk = trigger - HTF floor.
Trail: highest close since entry - atr_multiplier * ATR, never below the current stop, only moves up.

Deliberate deviation: the original has no broker-side stop (it polls). Here a disaster stop sits beyond
the structure so a fast wick cannot run unbounded between polls. It is an extra safety stop, not a rule.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class Protection:
    atr_trail_enabled: bool = True
    atr_multiplier: float = 2.0
    breakeven_rr: float = 1.5
    round_trip_cost_pct: float = 0.20

    def __post_init__(self) -> None:
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if self.breakeven_rr < 0:
            raise ValueError("breakeven_rr must be >= 0")
        if self.round_trip_cost_pct < 0:
            raise ValueError("round_trip_cost_pct must be >= 0")

    @classmethod
    def from_dict(cls, d: dict) -> "Protection":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trade:
    side: int                    # +1 BUY (the only side the OCL strategy trades)
    entry: float
    trigger: float               # touch level the entry fired on
    htf_sl: float                # HTF structural floor
    stop: float                  # current LTF stop (starts at the HTF floor)
    broker_sl: float             # server-side disaster stop
    tp: Optional[float]          # fixed target; None = DYNAMIC (the trail runs the trade)
    risk: float                  # trigger - HTF floor, the breakeven basis
    be_trigger: float
    be_floor: float
    be_armed: bool = False
    best_close: float = 0.0
    trail: Optional[float] = None

    @classmethod
    def open(cls, *, entry: float, trigger: float, htf_sl: float, broker_sl: float,
             tp: Optional[float], p: Protection) -> "Trade":
        risk = trigger - htf_sl
        be_trigger = trigger + risk * p.breakeven_rr if risk > 0 else math.inf
        return cls(side=1, entry=entry, trigger=trigger, htf_sl=htf_sl, stop=htf_sl,
                   broker_sl=broker_sl, tp=tp, risk=risk, be_trigger=be_trigger,
                   be_floor=trigger * (1.0 + p.round_trip_cost_pct / 100.0),
                   best_close=entry)

    def intrabar_exit(self, high: float, low: float) -> Optional[str]:
        """Broker-side exits inside one bar. The disaster stop is checked first (conservative)."""
        if low <= self.broker_sl:
            return "DISASTER"
        if self.tp is not None and high >= self.tp:
            return "TP"
        return None

    def on_closed_bar(self, *, high: float, close: float, atr: Optional[float], p: Protection) -> Optional[str]:
        """Close-based rules for one CLOSED LTF bar. Returns "SL" or "TRAIL" when the trade must close.
        Otherwise it advances breakeven and the trail for the next bar and returns None."""
        if close < self.stop:
            return "SL"
        if self.trail is not None and close < self.trail:
            return "TRAIL"
        if not self.be_armed and high >= self.be_trigger:
            self.be_armed = True
            self.stop = max(self.stop, self.be_floor)
        self.best_close = max(self.best_close, close)
        if p.atr_trail_enabled and atr is not None and atr > 0:
            candidate = max(self.best_close - p.atr_multiplier * atr, 0.0)
            new_trail = max(candidate, self.stop, self.htf_sl)
            if self.trail is None or new_trail > self.trail:
                self.trail = new_trail
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**d)
