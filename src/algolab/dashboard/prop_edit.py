"""Edit the active prop-firm profile from the dashboard.

Every save goes through PropRules, so an invalid value is refused and config/prop.json is left as it was.
A save writes a backup (config/prop.json.bak), replaces the file atomically, and appends a line to
state/prop_changes.jsonl with the before and after values. The live cycle reads config/prop.json on every
run, so a saved change applies from the next framework cycle.

Not editable here: the enabled switch (set it in config/prop.json, a deliberate step), and starting_balance
while an evaluation state is saved for the same profile (changing it would contradict the saved state).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algolab.framework.prop import PROP_STATE, PropError, PropRules
from algolab.ops import state as st

TYPES: dict[str, type] = {
    "starting_balance": float, "profit_target_pct": float, "max_daily_loss_pct": float,
    "max_total_loss_pct": float, "trailing_drawdown": bool, "min_trading_days": int,
    "max_risk_per_trade_pct": float, "max_open_positions": int, "max_lots": float,
    "close_before_weekend": bool, "weekend_close_hour_utc": int, "day_reset_hour_utc": int,
    "safety_buffer_pct": float,
}
assert set(TYPES) == set(PropRules.__dataclass_fields__), "every prop rule needs an editor type"


def _cfg_path(root: Path) -> Path:
    return Path(root) / "config" / "prop.json"


def _load(root: Path) -> dict:
    try:
        return json.loads(_cfg_path(root).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise PropError(f"missing {_cfg_path(root)}") from e
    except json.JSONDecodeError as e:
        raise PropError(f"invalid JSON in prop.json: {e}") from e


def _coerce(name: str, value: Any) -> Any:
    t = TYPES[name]
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise PropError(f"{name} must be true or false")
    try:
        x = float(value)
    except (TypeError, ValueError) as e:
        raise PropError(f"{name} must be a number") from e
    if not math.isfinite(x):
        raise PropError(f"{name} must be a finite number")
    if t is int:
        if x != int(x):
            raise PropError(f"{name} must be a whole number")
        return int(x)
    return x


def get_active(root: Path) -> dict:
    cfg = _load(root)
    name = cfg["active_profile"]
    prof = cfg["profiles"][name]
    saved = st.read_json(PROP_STATE, None) or {}
    return {
        "profile": name,
        "description": prof.get("description", ""),
        "enabled": bool(cfg.get("enabled", False)),
        "rules": prof["rules"],
        "fields": [{"name": k, "type": TYPES[k].__name__} for k in TYPES],
        "saved_state": {"profile": saved.get("profile"), "status": saved.get("status")} if saved else None,
    }


def save_rules(root: Path, updates: dict) -> dict:
    """Validate and save changes to the ACTIVE profile's rules. Returns the changes made."""
    cfg = _load(root)
    name = cfg["active_profile"]
    current = dict(cfg["profiles"][name]["rules"])
    unknown = set(updates) - set(TYPES)
    if unknown:
        raise PropError(f"not editable here: {sorted(unknown)}")
    new = dict(current)
    for k, v in updates.items():
        new[k] = _coerce(k, v)
    PropRules.from_dict(new)                              # raises PropError with the reason if invalid

    changes = {k: {"from": current.get(k), "to": new[k]} for k in TYPES
               if k in new and new[k] != current.get(k)}
    if not changes:
        return {"profile": name, "changes": {}}

    saved = st.read_json(PROP_STATE, None) or {}
    if "starting_balance" in changes and saved and saved.get("profile") == name:
        raise PropError("starting_balance cannot change while an evaluation state is saved for this profile")

    cfg["profiles"][name]["rules"] = new
    path = _cfg_path(root)
    shutil.copy2(path, path.with_suffix(".json.bak"))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

    st.STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = {"ts": datetime.now(timezone.utc).isoformat(), "profile": name, "changes": changes}
    with (st.STATE_DIR / "prop_changes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"profile": name, "changes": changes}
