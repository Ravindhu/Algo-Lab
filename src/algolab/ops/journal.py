"""Trade journal: one append-only JSON-lines file in the state folder. Every executed order writes an
"open" line. Every framework closure writes a "close" line with the P&L and where it came from:

  pnl_source = "broker"   realised profit from the broker's deal history (commission and swap included)
  pnl_source = "unknown"  the broker history could not be read; pnl is None and the trade is NOT
                          counted in the entry gate or in strategy selection

Estimated P&L (from a stop level or a bar close) is never written: it would make the statistics look better
or worse than the account really did.

The dashboard reads it, and the entry gate and strategy selection use the closed trades with broker P&L.
Legacy engulfing trades write "open" lines only."""
from __future__ import annotations

import json
from typing import Any

from algolab.framework.montecarlo import entry_allowed
from algolab.ops import state as _state
from algolab.ops.state import now_utc

FILE = "journal.jsonl"
STATE_DIR = None      # optional override (tests); None means the shared state folder, read at call time


def _dir():
    return STATE_DIR if STATE_DIR is not None else _state.STATE_DIR


def _path():
    return _dir() / FILE


def _append(rec: dict[str, Any]) -> dict[str, Any]:
    _dir().mkdir(parents=True, exist_ok=True)
    rec = {"ts": now_utc().isoformat(timespec="seconds"), **rec}
    with _path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        fh.flush()
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


def _open_for(ticket) -> dict | None:
    for row in reversed(read()):
        if row.get("kind") == "open" and row.get("ticket") == ticket:
            return row
    return None


def record_close(event: dict) -> dict:
    """event: {strategy, symbol, reason, price, ticket, pnl (broker realised P&L or None)}.
    Always writes a close line, so a closure can never disappear from the record."""
    opened = _open_for(event.get("ticket"))
    pnl = event.get("pnl")
    pnl = None if pnl is None else float(pnl)
    r = None
    if pnl is not None and opened is not None:
        risk = abs(float(opened["entry"]) - float(opened["sl"])) or 1e-12
        r = pnl / risk
    return _append({"kind": "close", "strategy": event["strategy"], "symbol": event["symbol"],
                    "ticket": event.get("ticket"), "reason": event["reason"],
                    "price": float(event["price"]), "pnl": pnl, "r": r,
                    "pnl_source": "broker" if pnl is not None else "unknown",
                    "orphan": opened is None})


def usable_closes(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Closed trades whose P&L is known from the broker."""
    rows = read() if rows is None else rows
    return [r for r in rows if r.get("kind") == "close" and r.get("pnl") is not None]


def closed_pnls(limit: int = 60) -> list[float]:
    return [float(r["pnl"]) for r in usable_closes()][-limit:]


def entry_gate(cfg: dict | None) -> tuple[bool, str]:
    """Gate for new framework entries. Disabled -> allowed.
    Warm-up: until min_trades closes exist, up to warmup_entries opens are allowed. Without this the
    gate could never open, because closed trades only come from trades. After warm-up, entries need the
    lower bound of the recent win rate to clear min_lower_bound (see montecarlo.entry_allowed)."""
    if not cfg or not cfg.get("enabled"):
        return True, "entry gate off"
    rows = read()
    closes = usable_closes(rows)
    opens = [r for r in rows if r.get("kind") == "open" and r.get("source") == "framework"]
    min_trades = int(cfg["min_trades"])
    if len(closes) < min_trades:
        if len(opens) < int(cfg["warmup_entries"]):
            return True, f"warm-up: {len(opens)}/{cfg['warmup_entries']} entries, {len(closes)} closed"
        return False, f"entry gate: {len(closes)} closed trades, need {min_trades} before more entries"
    recent = [float(r["pnl"]) for r in closes][-int(cfg["window"]):]
    return entry_allowed(recent, min_trades=min_trades,
                         min_lower_bound=float(cfg["min_lower_bound"]))
