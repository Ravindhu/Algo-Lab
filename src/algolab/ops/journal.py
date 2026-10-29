"""Trade journal: one append-only JSON-lines file in the state folder. Every executed order writes an
"open" line. Every framework closure writes a "close" line with the price P&L (in price units, signed by
side). The dashboard reads it, and the entry gate uses the closed trades to decide whether new entries
are allowed.

Legacy engulfing trades write "open" lines only. Their exits are on the broker side and are not yet
reported back, so they do not feed the entry gate."""
from __future__ import annotations

import json
from typing import Any

from algolab.framework.montecarlo import entry_allowed
from algolab.ops.state import STATE_DIR, now_utc

FILE = "journal.jsonl"


def _path():
    return STATE_DIR / FILE


def _append(rec: dict[str, Any]) -> dict[str, Any]:
    STATE_DIR.mkdir(exist_ok=True)
    rec = {"ts": now_utc().isoformat(timespec="seconds"), **rec}
    with _path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return rec


def read(limit: int | None = None) -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a torn line never blocks the reader
    return rows[-limit:] if limit else rows


def record_open(*, source: str, strategy: str, symbol: str, side: str, ticket: int | None,
                lots: float, entry: float, sl: float, tp: float | None) -> dict:
    return _append({"kind": "open", "source": source, "strategy": strategy, "symbol": symbol,
                    "side": side, "ticket": ticket, "lots": lots, "entry": entry, "sl": sl, "tp": tp})


def record_close(event: dict) -> dict | None:
    """event: {strategy, symbol, reason, price, ticket} from positions.manage. Joins the matching open
    line by ticket to compute the P&L. Returns None when no open line exists for that ticket."""
    opened = None
    for row in read():
        if row.get("kind") == "open" and row.get("ticket") == event.get("ticket"):
            opened = row
    if opened is None:
        return None
    sign = 1.0 if opened["side"] == "BUY" else -1.0
    pnl = sign * (float(event["price"]) - float(opened["entry"]))
    risk = abs(float(opened["entry"]) - float(opened["sl"])) or 1e-12
    return _append({"kind": "close", "strategy": event["strategy"], "symbol": event["symbol"],
                    "ticket": event["ticket"], "reason": event["reason"],
                    "price": float(event["price"]), "pnl": pnl, "r": pnl / risk})


def closed_pnls(limit: int = 60) -> list[float]:
    return [float(r["pnl"]) for r in read() if r.get("kind") == "close"][-limit:]


def entry_gate(cfg: dict | None) -> tuple[bool, str]:
    """Gate for new framework entries. Disabled -> allowed.
    Warm-up: until min_trades closes exist, up to warmup_entries opens are allowed. Without this the
    gate could never open, because closed trades only come from trades. After warm-up, entries need the
    lower bound of the recent win rate to clear min_lower_bound (see montecarlo.entry_allowed)."""
    if not cfg or not cfg.get("enabled"):
        return True, "entry gate off"
    rows = read()
    closes = [r for r in rows if r.get("kind") == "close"]
    opens = [r for r in rows if r.get("kind") == "open" and r.get("source") == "framework"]
    min_trades = int(cfg["min_trades"])
    if len(closes) < min_trades:
        if len(opens) < int(cfg["warmup_entries"]):
            return True, f"warm-up: {len(opens)}/{cfg['warmup_entries']} entries, {len(closes)} closed"
        return False, f"entry gate: {len(closes)} closed trades, need {min_trades} before more entries"
    recent = [float(r["pnl"]) for r in closes][-int(cfg["window"]):]
    return entry_allowed(recent, min_trades=min_trades,
                         min_lower_bound=float(cfg["min_lower_bound"]))
