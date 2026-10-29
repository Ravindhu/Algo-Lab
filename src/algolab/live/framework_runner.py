"""Connects the framework Bot to the live pipeline (run every 5 minutes with --framework).

One run, in order:
  1. Prop rules (if enabled): update the account state from equity, and flatten if FAILED or weekend.
  2. Manage open framework positions: close-based LTF stop, ATR trail, HTF floor (positions.py).
  3. Build the Bot. Restore each multi-timeframe strategy's saved state.
  4. Apply closure events to the strategies (retry ladder or pattern finished).
  5. Run the Bot: new entries. Prop rules gate each entry and cap its size.
  6. Save strategy state, tracked positions and prop state.

It runs only after the pipeline's guard() has passed. It uses DRY_RUN from the profile, so nothing reaches
the broker unless DRY_RUN=false. The framework is OFF by default (config/framework.json "enabled": false).
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime

from algolab.framework import Bot, ForexSymbol, TimeFrame
from algolab.framework.prop import ACTIVE, FAILED, PROP_CONFIG, PROP_STATE, PropGuard, guard_from_config, utc_naive
from algolab.framework.strategies import MultiTimeframeStrategy
from algolab.framework.strategies import registry
from algolab.live import positions
from algolab.live.adapter import AdapterError
from algolab.live.sizing import quote_to_account_rate
from algolab.ops import alerts, journal
from algolab.ops.state import now_utc, read_json, write_json
from algolab.profile import ROOT

log = logging.getLogger("algolab.framework")
STRATEGY_TYPES = {"scalp": "scalp_trend_pullback", "engulfing": "engulfing_hm",
                  "ocl": "ocl_keylevel", "ocl_sell_fail_top": "ocl_sell_fail_top",
                  "ocl_sell_fail_bottom": "ocl_sell_fail_bottom"}
STRATEGY_STATE = "framework_strategies.json"
POSITIONS_STATE = "framework_positions.json"


def load_config() -> dict:
    return json.loads((ROOT / "config" / "framework.json").read_text())


def load_prop_config() -> dict:
    if not PROP_CONFIG.exists():
        return {"enabled": False}
    return json.loads(PROP_CONFIG.read_text())


def make_sizer(ad, equity: float, cfg: dict, account_ccy: str = "USD",
               guard: PropGuard | None = None, now: datetime | None = None, open_count=None):
    """Lots from the risk budget, risk converted to account currency, and prop limits applied.

    Structural stop (OCL, sl_price): the broker stop LEVEL is kept, distance measured from the live price,
    and the target is recomputed as target_rr x (live price - HTF floor).
    Distance-based stops (scalp, engulfing): the distance is kept and the level is placed from the live price.
    A returned {"skip": reason} tells the Bot why no order was sent."""
    def sizer(rec: dict):
        sym = rec["symbol"]
        info = ad.symbol_info(sym)
        t = ad.tick(sym)
        sign = 1 if rec["side"] == "BUY" else -1
        entry = t["ask"] if sign > 0 else t["bid"]

        atr = rec.get("atr")
        max_sp = rec.get("max_spread_atr", cfg.get("max_spread_atr"))
        if atr and max_sp is not None and (t["ask"] - t["bid"]) > float(max_sp) * float(atr):
            log.info("[%s] spread %.5f too wide for ATR %.5f; skipped", sym, t["ask"] - t["bid"], atr)
            return {"skip": "spread too wide for ATR"}

        if rec.get("sl_price") is not None:
            sl = float(rec["sl_price"])
            sl_dist = sign * (entry - sl)
            if rec.get("tp_dist") is not None and rec.get("htf_sl") is not None:
                tp_dist = float(rec["target_rr"]) * sign * (entry - float(rec["htf_sl"]))
            else:
                tp_dist = None
        else:
            sl_dist = float(rec["sl_dist"])
            tp_dist = float(rec["tp_dist"]) if rec.get("tp_dist") is not None else None
            sl = entry - sign * sl_dist
        if not sl_dist > 0:
            return None

        rate = quote_to_account_rate(ad, sym, account_ccy)
        if rate is None:
            log.warning("[%s] no %s conversion pair; trade skipped (fail closed)", sym, account_ccy)
            return {"skip": f"no {account_ccy} conversion pair"}
        risk_per_lot = sl_dist * info["contract_size"] * rate
        lots = min(math.floor(equity * cfg["risk_per_trade"] / risk_per_lot * 100) / 100, cfg["max_lots"])
        if guard is not None:
            lots = min(lots, math.floor(guard.max_lots_for(equity, risk_per_lot) * 100) / 100)
            ok, why = guard.check_entry(now, equity, open_count() if open_count else 0, lots * risk_per_lot)
            if not ok:
                return {"skip": why}
        if lots < info["volume_min"]:
            return None                                  # Bot reports "size below broker minimum"
        digits = info["digits"]
        return {"lots": round(lots, 2),
                "sl": round(sl, digits),
                "tp": round(entry + sign * tp_dist, digits) if tp_dist is not None else 0.0}
    return sizer


def _key(strategy_name: str, symbol: str) -> str:
    return f"{strategy_name}:{symbol}"


def build_bot(ad, p: dict, symbols: list[str], equity: float, cfg: dict, account_ccy: str,
              tracked: dict, guard: PropGuard | None = None, now: datetime | None = None
              ) -> tuple[Bot, dict]:
    tf = TimeFrame(cfg["timeframe"])
    open_count = lambda: len(tracked)  # noqa: E731  (framework positions only)
    bot = Bot(adapter=ad, dry_run=p["dry_run"], max_open=cfg["max_open"],
              magic=cfg["magic"], deviation=cfg["deviation_points"],
              sizer=make_sizer(ad, equity, cfg, account_ccy, guard, now, open_count))
    prev_gate = bot.gate
    gate_ok, gate_why = journal.entry_gate(cfg.get("entry_gate"))

    def combined_gate(rec: dict) -> tuple[bool, str]:
        if not gate_ok:
            return False, gate_why
        if guard is not None:
            return guard.check_entry(now, equity, open_count(), 0.0)
        return prev_gate(rec) if prev_gate else (True, "ok")

    bot.gate = combined_gate
    saved = read_json(STRATEGY_STATE, {})
    objs: dict[tuple[str, str], object] = {}

    async def notify(rec: dict) -> None:
        if "error" in rec:
            alerts.send(f"framework {rec['strategy']}: {rec['error']}", "WARNING",
                        key=f"fw-{rec['strategy']}", cooldown_min=60)
            return
        mode = "DRY RUN" if p["dry_run"] else "LIVE"
        obj = objs.get((rec["strategy"], rec["symbol"]))
        if rec.get("executed") and "protect" in rec and obj is not None:
            obj.on_entry()
            tracked[str(rec["ticket"])] = positions.new_record(
                rec, strategy=rec["strategy"], ltf=obj.ltf_tf, htf=obj.htf_tf)
            text = (f"{rec['symbol']} BUY {rec['lots']} lots | SL {rec['sl']} "
                    f"TP {rec['tp'] or 'none (trail)'} | {mode}")
            alerts.send(f"[{rec['strategy']}] {text}", "WARNING")
        elif rec.get("executed"):
            alerts.send(f"[{rec['strategy']}] {rec['symbol']} {rec['side']} {rec['lots']} lots | "
                        f"SL {rec['sl']} TP {rec['tp']} | {mode}", "WARNING")
        else:
            why = rec.get("why", "not placed")
            if why.startswith("prop"):
                alerts.send(f"[{rec['strategy']}] {rec['symbol']} {rec['side']} blocked by prop rules: "
                            f"{why} | {mode}", "WARNING", key=f"prop-block-{rec['symbol']}", cooldown_min=60)
            else:
                alerts.send(f"[{rec['strategy']}] {rec['symbol']} {rec['side']} signal ({why}) | {mode}",
                            "WARNING", key=f"fw-skip-{rec['strategy']}-{rec['symbol']}", cooldown_min=60)

    bot.on_signal = notify
    for entry in cfg["strategies"]:
        if not registry.enabled(entry):
            log.info("strategy %s disabled in config; skipped", entry.get("type"))
            continue
        try:
            cls = registry.load(STRATEGY_TYPES.get(entry["type"], entry["type"]))
        except KeyError as e:
            log.error("%s; entry skipped", e)
            alerts.send(f"framework: unknown strategy type {entry.get('type')!r} (skipped)",
                        "WARNING", key=f"fw-unknown-{entry.get('type')}", cooldown_min=1440)
            continue
        params = dict(entry.get("params", {}))
        strategy_tf = params.pop("timeframe", tf)
        for s in symbols:
            if issubclass(cls, MultiTimeframeStrategy):
                obj = cls(ForexSymbol(s), **params)
            else:
                obj = cls(ForexSymbol(s), strategy_tf, **params)
            objs[(obj.name, s)] = obj
            if isinstance(obj, MultiTimeframeStrategy) and _key(obj.name, s) in saved:
                obj.restore(saved[_key(obj.name, s)])
            bot.add(obj)
    return bot, objs


def _apply_events(objs: dict, events: list[dict]) -> None:
    for ev in events:
        obj = objs.get((ev["strategy"], ev["symbol"]))
        if obj is not None:
            obj.on_exit(ev["reason"], ev["price"])
        alerts.send(f"[{ev['strategy']}] {ev['symbol']} closed: {ev['reason']} at {ev['price']:.5f} "
                    f"(ticket {ev['ticket']})", "WARNING")


async def run_framework(ad, p: dict, symbols: list[str], equity: float, now: datetime | None = None,
                        cfg: dict | None = None, prop_cfg: dict | None = None) -> dict | None:
    """Returns {"results", "events", "prop"}, or None when the framework is disabled.
    cfg and prop_cfg default to the config files; tests and the e2e replay pass their own."""
    cfg = load_config() if cfg is None else cfg
    if not cfg.get("enabled"):
        return None
    now = utc_naive(now if now is not None else now_utc())
    prop_cfg = load_prop_config() if prop_cfg is None else prop_cfg
    account_ccy = ad.account_currency()
    tracked = read_json(POSITIONS_STATE, {})

    guard, changes = None, []
    if prop_cfg.get("enabled"):
        guard = guard_from_config(prop_cfg, read_json(PROP_STATE, None))  # PropError: fail closed
        changes = guard.update(now, equity)

    events = await positions.manage(ad, tracked, magic=cfg["magic"], deviation=cfg["deviation_points"])

    if guard is not None:
        for ch in changes:
            alerts.send(f"prop account {ch}: {guard.s.fail_reason or 'profile target reached'} "
                        f"(profile {guard.s.profile}, equity {equity:.2f})",
                        "CRITICAL" if ch == FAILED else "WARNING", key=f"prop-{ch}", cooldown_min=1440)
        why = guard.must_flatten(now)
        if why and tracked:
            reason = "PROP" if FAILED in why else "WEEKEND"
            events += positions.flatten(ad, tracked, magic=cfg["magic"],
                                        deviation=cfg["deviation_points"], reason=reason)

    bot, objs = build_bot(ad, p, symbols, equity, cfg, account_ccy, tracked, guard, now)
    _apply_events(objs, events)
    results = await bot.run_once()

    write_json(STRATEGY_STATE, {_key(name, sym): o.state()
                                for (name, sym), o in objs.items()
                                if isinstance(o, MultiTimeframeStrategy)})
    write_json(POSITIONS_STATE, tracked)
    if guard is not None:
        write_json(PROP_STATE, {**guard.state_dict(), "profile": guard.s.profile})
    return {"results": results, "events": events,
            "prop": guard.summary() if guard is not None else None}
