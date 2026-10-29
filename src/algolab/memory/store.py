"""Agent memory: three append-only stores, one JSON-lines file each.

  working      what the agents are doing now (current hypothesis, open questions, last sync)
  calibration  predicted vs realised outcomes (used to check whether confidence means anything)
  mistake      what went wrong, why, and the rule adopted to avoid it

Records are plain dicts. Each gets a UTC timestamp. Files live under the ops STATE_DIR so the
bot's runtime state stays in one place. Nothing here places orders or changes configuration.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algolab.ops import state as ops_state

KINDS = ("working", "calibration", "mistake")


class MemoryStoreError(ValueError):
    pass


def _path(kind: str) -> Path:
    if kind not in KINDS:
        raise MemoryStoreError(f"unknown memory kind {kind!r}; expected one of {KINDS}")
    return ops_state.STATE_DIR / "memory" / f"{kind}.jsonl"


def remember(kind: str, record: dict[str, Any]) -> dict[str, Any]:
    """Append one record. Returns the stored record (with its timestamp)."""
    if not isinstance(record, dict) or not record:
        raise MemoryStoreError("record must be a non-empty dict")
    stored = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    path = _path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stored, sort_keys=True, default=str) + "\n")
    return stored


def recall(kind: str, limit: int = 20) -> list[dict[str, Any]]:
    """Most recent `limit` records, oldest first. Missing file means empty memory."""
    path = _path(kind)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-limit:] if limit > 0 else lines:
        if line.strip():
            out.append(json.loads(line))
    return out


def calibration_gap(kind_records: list[dict[str, Any]]) -> dict[str, float] | None:
    """Mean predicted probability vs realised win rate over calibration records that carry
    `predicted` (0-1) and `won` (bool). A large gap means the confidence numbers are not trustworthy."""
    rows = [r for r in kind_records if "predicted" in r and "won" in r]
    if not rows:
        return None
    pred = sum(float(r["predicted"]) for r in rows) / len(rows)
    real = sum(1.0 for r in rows if r["won"]) / len(rows)
    return {"n": len(rows), "mean_predicted": pred, "realised_win_rate": real, "gap": pred - real}
