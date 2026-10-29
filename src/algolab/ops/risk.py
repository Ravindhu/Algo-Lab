"""Drawdown and daily-loss limits. Shared by the live pipeline and the ops (alert-only) pipeline.

The limits measure TRADING results. Deposits and withdrawals are removed first (adjusted_equity), so moving
money into or out of the account cannot trigger the kill switch."""
from __future__ import annotations

from algolab.ops.state import now_utc, read_json, write_json


def adjusted_equity(equity: float, net_deposits: float) -> float:
    """Equity with money moved in or out removed: equity - (deposits - withdrawals)."""
    return float(equity) - float(net_deposits or 0.0)


def check_limits(equity: float, r: dict) -> tuple[bool, str, bool]:
    """Returns (ok, reason, drawdown_breach). The drawdown breach is the caller's cue to engage the
    kill switch and alert. Updates risk.json (peak and the day's starting equity) in both cases.
    `equity` must already be adjusted with adjusted_equity()."""
    today = now_utc().date().isoformat()
    risk = read_json("risk.json", {})
    peak = max(float(risk.get("peak", equity)), equity)
    day_start = float(risk["day_start"]) if risk.get("day") == today else equity

    if equity <= peak * (1 - r["max_drawdown_kill"]):
        return False, f"max drawdown {r['max_drawdown_kill']:.0%} breached", True

    write_json("risk.json", {"peak": peak, "day": today, "day_start": day_start})
    if equity <= day_start * (1 - r["max_daily_loss"]):
        return False, "daily loss limit reached (paused until next UTC day)", False
    return True, "ok", False
