"""Connects the framework Bot to the live pipeline.

Runs only after the pipeline's guard() has passed (kill switch, validation gate, drawdown, daily loss,
open-position limits). It uses DRY_RUN from the profile, so nothing is sent to the broker unless
DRY_RUN=false. The framework is OFF by default (config/framework.json "enabled": false).
"""
from __future__ import annotations

import json
import logging
import math

from algolab.framework import Bot, EngulfingRsiStrategy, ForexSymbol, ScalpTrader, TimeFrame
from algolab.ops import alerts
from algolab.profile import ROOT

log = logging.getLogger("algolab.framework")
STRATEGY_TYPES = {"scalp": ScalpTrader, "engulfing": EngulfingRsiStrategy}


def load_config() -> dict:
    return json.loads((ROOT / "config" / "framework.json").read_text())


def make_sizer(ad, equity: float, cfg: dict):
    """Lots from risk_per_trade and the stop distance; SL/TP from the live price."""
    def sizer(rec: dict) -> dict | None:
        sl_dist, tp_dist = float(rec["sl_dist"]), float(rec["tp_dist"])
        if not sl_dist > 0:
            return None
        info = ad.symbol_info(rec["symbol"])
        t = ad.tick(rec["symbol"])
        sign = 1 if rec["side"] == "BUY" else -1
        entry = t["ask"] if sign > 0 else t["bid"]
        raw = equity * cfg["risk_per_trade"] / (sl_dist * info["contract_size"])
        lots = min(math.floor(raw * 100) / 100, cfg["max_lots"])
        if lots < info["volume_min"]:
            return None
        digits = info["digits"]
        return {"lots": round(lots, 2),
                "sl": round(entry - sign * sl_dist, digits),
                "tp": round(entry + sign * tp_dist, digits)}
    return sizer


def build_bot(ad, p: dict, symbols: list[str], equity: float, cfg: dict) -> Bot:
    tf = TimeFrame(cfg["timeframe"])
    bot = Bot(adapter=ad, dry_run=p["dry_run"], max_open=cfg["max_open"],
              magic=cfg["magic"], deviation=cfg["deviation_points"],
              sizer=make_sizer(ad, equity, cfg))

    async def notify(rec: dict) -> None:
        if "error" in rec:
            alerts.send(f"framework {rec['strategy']}: {rec['error']}", "WARNING",
                        key=f"fw-{rec['strategy']}", cooldown_min=60)
            return
        mode = "DRY RUN" if rec["dry_run"] else "LIVE"
        if rec.get("executed"):
            text = f"{rec['symbol']} {rec['side']} {rec['lots']} lots | SL {rec['sl']} TP {rec['tp']} | {mode}"
        else:
            text = f"{rec['symbol']} {rec['side']} signal ({rec.get('why', 'not placed')}) | {mode}"
        alerts.send(f"[{rec['strategy']}] {text}", "WARNING")

    bot.on_signal = notify
    for entry in cfg["strategies"]:
        cls = STRATEGY_TYPES[entry["type"]]
        for s in symbols:
            bot.add(cls(ForexSymbol(s), tf, **entry.get("params", {})))
    return bot


async def run_framework(ad, p: dict, symbols: list[str], equity: float) -> list[dict] | None:
    """Returns the Bot's results, or None when the framework is disabled."""
    cfg = load_config()
    if not cfg.get("enabled"):
        return None
    bot = build_bot(ad, p, symbols, equity, cfg)
    return await bot.run_once()
