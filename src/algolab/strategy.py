"""Single source of truth for strategy parameters: config/strategy.json.

Written by the autotuner (only after the holdout gate and the formal validation pass) or by hand.
Every write keeps a timestamped backup in state/param_history/."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from algolab.backtest.engine import Params
from algolab.ops.state import STATE_DIR, now_utc
from algolab.profile import ROOT

FILE = ROOT / "config" / "strategy.json"
KEYS = ("rsi_n", "oversold", "overbought", "atr_n", "sl_mult", "tp_mult")


def load() -> dict:
    return json.loads(FILE.read_text())


def params(d: dict | None = None) -> Params:
    d = d or load()
    return Params(**{k: d[k] for k in KEYS})


def live_overrides(d: dict | None = None) -> dict:
    d = d or load()
    return {k: d[k] for k in ("sl_mult", "tp_mult", "atr_n")}


def write(new: dict, reason: str) -> Path:
    """Back up the current file, then replace it atomically. Returns the backup path."""
    hist = STATE_DIR / "param_history"
    hist.mkdir(parents=True, exist_ok=True)
    backup = hist / f"strategy_{now_utc().strftime('%Y%m%dT%H%M%SZ')}.json"
    shutil.copy(FILE, backup)
    cur = load()
    out = {**{k: new[k] for k in KEYS}, "ml_threshold": float(new["ml_threshold"]),
           "version": cur.get("version", 0) + 1, "updated": now_utc().isoformat(),
           "reason": reason}
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, FILE)
    return backup


def restore(backup: Path) -> None:
    shutil.copy(backup, FILE)
