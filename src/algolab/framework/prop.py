"""Prop-firm rules management.

A prop-firm evaluation is judged against rules: a profit target, a daily loss limit, a total loss limit
(static or trailing), a minimum number of trading days, and often position and weekend limits. The numbers
differ by firm and by challenge phase. This module keeps them as named profiles in config/prop.json,
enforces them on every framework cycle and in the backtest, and records the account's status.

Limits are checked against EQUITY, which includes floating P/L. That is the stricter reading. Check your
firm's terms for the exact definition (balance or equity, static or trailing, the daily reset time).

Status: ACTIVE -> FAILED (a limit was breached) or ACTIVE -> PASSED (target met and minimum trading days).
FAILED is sticky. Only a person can clear it: python -m algolab.framework.prop reset --confirm
Changing the active profile is refused while a saved state belongs to another profile, for the same reason.

Known limits:
  - Equity is observed once per framework cycle (every 5 minutes live, once per bar in the backtest).
    A breach between observations is detected at the next observation. Keep the safety buffer above zero.
  - "Trading day" counts days with at least one entry. Firms may count differently.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from algolab.profile import ROOT

log = logging.getLogger("algolab.prop")
ACTIVE, PASSED, FAILED = "ACTIVE", "PASSED", "FAILED"
PROP_CONFIG = ROOT / "config" / "prop.json"
PROP_STATE = "prop_state.json"


class PropError(RuntimeError):
    pass


def utc_naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


@dataclass(frozen=True)
class PropRules:
    starting_balance: float
    profit_target_pct: float = 10.0           # 0 = no target (funded or unlimited account)
    max_daily_loss_pct: float = 5.0           # % of starting balance, from start-of-day equity
    max_total_loss_pct: float = 10.0          # % of starting balance
    trailing_drawdown: bool = False           # True: floor follows peak equity; False: fixed at start
    min_trading_days: int = 0
    max_risk_per_trade_pct: float = 1.0       # % of current equity risked at the broker stop
    max_open_positions: int = 1
    max_lots: float = 0.0                     # 0 = no extra cap
    close_before_weekend: bool = True
    weekend_close_hour_utc: int = 20          # Friday: no entries from this hour; flatten here
    day_reset_hour_utc: int = 0               # the trading day starts at this UTC hour
    safety_buffer_pct: float = 0.5            # % of starting balance kept clear of every limit

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise PropError("starting_balance must be > 0")
        for name in ("max_daily_loss_pct", "max_total_loss_pct", "max_risk_per_trade_pct"):
            if not 0 < getattr(self, name) <= 100:
                raise PropError(f"{name} must be in (0, 100]")
        if not 0 <= self.profit_target_pct <= 100:
            raise PropError("profit_target_pct must be in [0, 100]")
        if self.min_trading_days < 0 or self.max_open_positions < 1 or self.max_lots < 0:
            raise PropError("min_trading_days >= 0, max_open_positions >= 1, max_lots >= 0")
        for name in ("weekend_close_hour_utc", "day_reset_hour_utc"):
            if not 0 <= getattr(self, name) <= 23:
                raise PropError(f"{name} must be 0..23")
        if not 0 <= self.safety_buffer_pct < min(self.max_daily_loss_pct, self.max_total_loss_pct):
            raise PropError("safety_buffer_pct must be >= 0 and smaller than both loss limits")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PropRules":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise PropError(f"unknown prop rule(s): {sorted(unknown)}")
        return cls(**d)


@dataclass
class PropState:
    profile: str = ""
    status: str = ACTIVE
    fail_reason: str | None = None
    failed_at: str | None = None
    day: str | None = None
    day_start_equity: float | None = None
    peak_equity: float | None = None
    last_equity: float | None = None
    last_ts: str | None = None
    trading_days: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PropState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PropGuard:
    """Stateful enforcement. Pure apart from its own state object: it never talks to the broker."""

    def __init__(self, rules: PropRules, state: PropState | None = None, profile: str = ""):
        self.r = rules
        self.s = state or PropState(profile=profile)

    # ---- limits ----
    def daily_limit(self) -> float:
        return self.r.starting_balance * self.r.max_daily_loss_pct / 100

    def total_limit(self) -> float:
        return self.r.starting_balance * self.r.max_total_loss_pct / 100

    def buffer(self) -> float:
        return self.r.starting_balance * self.r.safety_buffer_pct / 100

    def floor(self) -> float:
        """Equity below this breaches the total loss limit."""
        if self.r.trailing_drawdown:
            peak = self.s.peak_equity if self.s.peak_equity is not None else self.r.starting_balance
            return peak - self.total_limit()
        return self.r.starting_balance - self.total_limit()

    def target_amount(self) -> float:
        return self.r.starting_balance * self.r.profit_target_pct / 100

    def room(self, equity: float) -> tuple[float, float]:
        """(daily_room, total_room): how far equity can still fall before each limit is breached."""
        day_start = self.s.day_start_equity if self.s.day_start_equity is not None else equity
        return self.daily_limit() - (day_start - equity), equity - self.floor()

    # ---- day handling ----
    def _day(self, ts: pd.Timestamp) -> str:
        return (ts - pd.Timedelta(hours=self.r.day_reset_hour_utc)).strftime("%Y-%m-%d")

    # ---- feeding equity ----
    def update(self, ts, equity: float) -> list[str]:
        """Feed mark-to-market equity at time ts. Returns status changes, for alerts."""
        t = utc_naive(ts)
        s = self.s
        day = self._day(t)
        if s.day != day:
            s.day = day
            s.day_start_equity = s.last_equity if s.last_equity is not None else equity
        s.peak_equity = equity if s.peak_equity is None else max(s.peak_equity, equity)
        s.last_equity, s.last_ts = equity, t.isoformat()
        if s.status != ACTIVE:
            return []
        if equity <= self.floor():
            return self._fail(t, f"total loss limit: equity {equity:.2f} <= floor {self.floor():.2f}")
        if s.day_start_equity - equity >= self.daily_limit():
            return self._fail(t, f"daily loss limit: fell {s.day_start_equity - equity:.2f} "
                                 f"from day start {s.day_start_equity:.2f}")
        gained = equity - self.r.starting_balance
        if self.r.profit_target_pct > 0 and gained >= self.target_amount() \
                and len(s.trading_days) >= self.r.min_trading_days:
            s.status = PASSED
            log.info("[prop] profit target reached at equity %.2f", equity)
            return [PASSED]
        return []

    def _fail(self, t: pd.Timestamp, reason: str) -> list[str]:
        self.s.status, self.s.fail_reason, self.s.failed_at = FAILED, reason, t.isoformat()
        log.error("[prop] FAILED: %s", reason)
        return [FAILED]

    # ---- decisions ----
    def weekend_blocked(self, ts) -> bool:
        if not self.r.close_before_weekend:
            return False
        t = utc_naive(ts)
        return t.weekday() >= 5 or (t.weekday() == 4 and t.hour >= self.r.weekend_close_hour_utc)

    def must_flatten(self, ts) -> str | None:
        """Why every open position must close now, or None."""
        if self.s.status == FAILED:
            return f"prop account FAILED ({self.s.fail_reason})"
        if self.r.close_before_weekend and self.weekend_blocked(ts):
            return "weekend close"
        return None

    def check_entry(self, ts, equity: float, open_positions: int, risk_amount: float) -> tuple[bool, str]:
        s = self.s
        if s.status != ACTIVE:
            return False, f"prop account {s.status}" + (f": {s.fail_reason}" if s.fail_reason else "")
        if self.weekend_blocked(ts):
            return False, "prop: no new entries in the weekend window"
        if open_positions >= self.r.max_open_positions:
            return False, "prop: max open positions reached"
        daily, total = self.room(equity)
        buf = self.buffer()
        if daily <= buf:
            return False, "prop: inside the daily-loss safety buffer"
        if total <= buf:
            return False, "prop: inside the total-loss safety buffer"
        if risk_amount > min(daily, total) - buf:
            return False, f"prop: trade risk {risk_amount:.2f} exceeds the remaining room"
        return True, "ok"

    def max_lots_for(self, equity: float, risk_per_lot: float) -> float:
        """Largest lot size within max_risk_per_trade_pct, the remaining room and max_lots."""
        if risk_per_lot <= 0:
            return 0.0
        daily, total = self.room(equity)
        caps = [equity * self.r.max_risk_per_trade_pct / 100 / risk_per_lot,
                (min(daily, total) - self.buffer()) / risk_per_lot]
        if self.r.max_lots > 0:
            caps.append(self.r.max_lots)
        return max(0.0, min(caps))

    def record_entry(self, ts) -> None:
        day = self._day(utc_naive(ts))
        if day not in self.s.trading_days:
            self.s.trading_days.append(day)

    def reset(self, ts) -> None:
        """Operator action only. Clears FAILED or PASSED and keeps the equity history."""
        self.s.status, self.s.fail_reason, self.s.failed_at = ACTIVE, None, None
        log.warning("[prop] status reset by operator at %s", utc_naive(ts).isoformat())

    def summary(self) -> dict[str, Any]:
        s = self.s
        out = {"profile": s.profile, "status": s.status, "fail_reason": s.fail_reason,
               "equity": s.last_equity, "floor": self.floor(), "peak_equity": s.peak_equity,
               "trading_days": len(s.trading_days), "min_trading_days": self.r.min_trading_days}
        if s.last_equity is not None:
            daily, total = self.room(s.last_equity)
            out.update(daily_room=daily, total_room=total)
        return out

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.s)


def load_prop_config(path: Path = PROP_CONFIG) -> dict[str, Any]:
    if not path.exists():
        return {"enabled": False}
    return json.loads(path.read_text())


def select_profile(cfg: dict[str, Any]) -> tuple[str, PropRules]:
    name = cfg.get("active_profile")
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        raise PropError(f"active_profile {name!r} is not in profiles {sorted(profiles)}")
    return name, PropRules.from_dict(profiles[name]["rules"])


def guard_from_config(cfg: dict[str, Any], saved: dict[str, Any] | None) -> PropGuard:
    """Build the guard for the active profile and restore the saved state. Refuses a profile change
    while the saved state belongs to a different profile, so the operator has to decide."""
    name, rules = select_profile(cfg)
    if saved and saved.get("profile") not in (None, "", name):
        raise PropError(f"saved prop state belongs to profile {saved.get('profile')!r}, "
                        f"but active_profile is {name!r}. Run: python -m algolab.framework.prop reset --confirm "
                        f"after reviewing the old account")
    state = PropState.from_dict(saved) if saved else PropState(profile=name)
    state.profile = name
    return PropGuard(rules, state, profile=name)


# ---------- operator commands ----------

def main(argv: list[str] | None = None) -> int:
    from algolab.ops.state import now_utc, read_json, write_json

    ap = argparse.ArgumentParser(prog="python -m algolab.framework.prop",
                                 description="Prop-firm rules: show profiles, status, reset.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="list profiles and validate each one")
    sub.add_parser("status", help="show the saved account status")
    r = sub.add_parser("reset", help="clear FAILED/PASSED (a person must run this)")
    r.add_argument("--confirm", action="store_true", help="required to reset")
    args = ap.parse_args(argv)

    cfg = load_prop_config()
    if args.cmd == "show":
        active = cfg.get("active_profile")
        for name, prof in cfg.get("profiles", {}).items():
            try:
                PropRules.from_dict(prof["rules"])
                ok = "valid"
            except (PropError, TypeError, KeyError) as e:
                ok = f"INVALID: {e}"
            mark = "*" if name == active else " "
            print(f"{mark} {name}: {ok}  {prof.get('description', '')}")
        print(f"enabled={cfg.get('enabled', False)} active={active}")
        return 0
    saved = read_json(PROP_STATE, None)
    if args.cmd == "status":
        print(json.dumps(saved or {"status": "no saved state"}, indent=2, default=str))
        return 0
    if not args.confirm:
        print("refusing to reset without --confirm")
        return 2
    if not saved:
        print("nothing to reset: no saved prop state")
        return 0
    saved.update(status=ACTIVE, fail_reason=None, failed_at=None)
    saved["reset_at"] = now_utc().isoformat()
    write_json(PROP_STATE, saved)
    print(f"reset: profile {saved.get('profile')} is ACTIVE again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
