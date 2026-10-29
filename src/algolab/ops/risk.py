"""Drawdown and daily-loss limits. Shared by the live pipeline and the ops (alert-only) pipeline,
which used to carry two copies of this logic."""
from __future__ import annotations

from algolab.ops.state import now_utc, read_json, write_json


def check_limits(equity: float, r: dict) -> tuple[bool, str, bool]:
    """Returns (ok, reason, drawdown_breach). The drawdown breach is the caller's cue to engage the
    kill switch and alert. Updates risk.json (peak and the day's starting equity) in both cases."""
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
