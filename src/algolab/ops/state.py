"""Atomic state files, kill switch and ops config. Everything the unattended loop remembers lives in state/."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from algolab.profile import ROOT

STATE_DIR = ROOT / "state"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def minutes_since(iso_ts: str) -> float:
    return (now_utc() - datetime.fromisoformat(iso_ts)).total_seconds() / 60


def read_json(name: str, default=None):
    path = STATE_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json(name: str, obj) -> None:
    """Write to a temp file, then rename. A crash never leaves a half-written state file."""
    STATE_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, STATE_DIR / name)


def kill_switch_on() -> bool:
    return (STATE_DIR / "KILL").exists()


def engage_kill(reason: str) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / "KILL").write_text(f"{now_utc().isoformat()} {reason}\n")


def ops_config() -> dict:
    return json.loads((ROOT / "config" / "ops.json").read_text())
