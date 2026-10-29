"""Atomic state files, kill switch, cycle lock and ops config. Everything the unattended loop remembers lives in state/."""
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from algolab.profile import ROOT

STATE_DIR = ROOT / "state"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def minutes_since(iso_ts: str) -> float:
    return (now_utc() - datetime.fromisoformat(iso_ts)).total_seconds() / 60


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temp file in the same folder, fsync, then rename over the target.
    A crash leaves either the old file or the new one, never a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(name: str, default=None):
    path = STATE_DIR / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json(name: str, obj) -> None:
    atomic_write_text(STATE_DIR / name, json.dumps(obj, indent=2, default=str))


class LockBusy(RuntimeError):
    pass


@contextmanager
def cycle_lock(name: str, stale_minutes: float = 10.0):
    """Exclusive lock for one trading cycle. Two overlapping runs (timer overlap, a manual run during a
    scheduled one) cannot both place orders. A lock older than stale_minutes is treated as left behind by
    a crashed process and removed."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{name}.lock"
    if path.exists():
        age_min = (now_utc().timestamp() - path.stat().st_mtime) / 60
        if age_min < stale_minutes:
            raise LockBusy(f"{name} cycle already running (lock age {age_min:.1f} min)")
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise LockBusy(f"{name} cycle already running") from e
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def kill_switch_on() -> bool:
    return (STATE_DIR / "KILL").exists()


def engage_kill(reason: str) -> None:
    atomic_write_text(STATE_DIR / "KILL", f"{now_utc().isoformat()} {reason}\n")


def ops_config() -> dict:
    return json.loads((ROOT / "config" / "ops.json").read_text())
