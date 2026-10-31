"""Trade setup for the dashboard header: HTF, LTF, risk per trade and symbols.

Pure functions over config files, standard library only, so they can be tested without the dashboard stack.

Where each value lives (the keys the live code already reads):
  htf      config/framework.json  engulfing_ocl params.htf   (+ active profile scanner.htf for the alert scan)
  ltf      config/framework.json  engulfing_ocl params.ltf
  risk     config/framework.json  risk_per_trade             (+ config/live.json risk_per_trade)
  symbols  config/ops.json        symbols

Changing htf, ltf or risk changes the config fingerprint in backtest/validate.py, so the last validation goes
stale and trading stops until `algolab validate` runs again. save_setup reports this as needs_validate.

Writes need DASH_ALLOW_WRITES=true (the same gate as Promote). Every value is checked before any file is touched.
Each file is backed up to <name>.bak and replaced atomically. Nothing here places orders or touches DRY_RUN."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path

from algolab.profile import ROOT

HTF_CHOICES = ("H1", "H4", "D1")
LTF_CHOICES = ("M5", "M15", "M30")
TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
RISK_MIN_PCT = 0.1
RISK_ABS_MAX_PCT = 3.0          # never above 3% of equity per trade, whatever the config says
MAX_SYMBOLS = 20
SYM_RE = re.compile(r"^[A-Z0-9]{3,6}$")
STRATEGY_TYPE = "engulfing_ocl"  # the framework strategy whose htf/ltf the header controls


class SetupError(ValueError):
    """A setup that cannot be saved. The message is safe to show in the dashboard."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_name() -> str:
    return os.getenv("ALGOLAB_PROFILE", "local")   # same default as algolab.profile.load_profile


def _engulfing_entry(fw: dict) -> dict | None:
    return next((e for e in fw.get("strategies", []) if e.get("type") == STRATEGY_TYPE), None)


def risk_cap_pct(ops: dict) -> float:
    """Per-trade risk may not exceed the daily loss limit in ops.json, or RISK_ABS_MAX_PCT."""
    dl = ((ops.get("risk") or {}).get("max_daily_loss"))
    cap = RISK_ABS_MAX_PCT if dl is None else min(RISK_ABS_MAX_PCT, float(dl) * 100)
    return round(cap, 4)


def normalize_symbol(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(raw or "").strip().upper())


def add_symbol(symbols: list[str], raw: str) -> tuple[list[str], str | None]:
    """Returns (new_list, error). The list is unchanged when there is an error."""
    sym = normalize_symbol(raw)
    if not sym:
        return list(symbols), "Enter a symbol, e.g. EURUSD"
    if not SYM_RE.match(sym):
        return list(symbols), "Use 3 to 6 letters or digits, e.g. EURUSD"
    if sym in symbols:
        return list(symbols), f"{sym} is already added"
    if len(symbols) >= MAX_SYMBOLS:
        return list(symbols), f"Maximum {MAX_SYMBOLS} symbols"
    return [*symbols, sym], None


def read_setup(root: Path = ROOT) -> dict:
    """The current setup as the dashboard shows it. Missing optional files give None, not an error."""
    cfg = Path(root) / "config"
    fw = _load(cfg / "framework.json")
    ops = _load(cfg / "ops.json")
    eng = _engulfing_entry(fw) or {}
    params = eng.get("params") or {}
    profile_path = cfg / "profiles" / f"{_profile_name()}.json"
    profile_htf = _load(profile_path).get("scanner", {}).get("htf") if profile_path.exists() else None
    return {
        "htf": params.get("htf") or profile_htf,
        "ltf": params.get("ltf"),
        "risk_pct": round(float(fw["risk_per_trade"]) * 100, 4),
        "symbols": list(ops.get("symbols", [])),
        "risk_cap_pct": risk_cap_pct(ops),
    }


def validate_setup(s: dict, root: Path = ROOT) -> list[str]:
    """Every problem with the setup, as a list of messages. Empty means it can be saved."""
    errs: list[str] = []
    htf, ltf = s.get("htf"), s.get("ltf")
    if htf not in HTF_CHOICES:
        errs.append("Pick an HTF: H1, H4 or D1")
    if ltf not in LTF_CHOICES:
        errs.append("Pick an LTF: M5, M15 or M30")
    if htf in TF_MINUTES and ltf in TF_MINUTES and TF_MINUTES[htf] <= TF_MINUTES[ltf]:
        errs.append("HTF must be higher than LTF")

    risk = s.get("risk_pct")
    cap = risk_cap_pct(_load(Path(root) / "config" / "ops.json"))
    if isinstance(risk, bool) or not isinstance(risk, (int, float)) or not math.isfinite(risk):
        errs.append("Risk must be a number")
    elif not (RISK_MIN_PCT <= risk <= cap):
        errs.append(f"Risk must be between {RISK_MIN_PCT}% and {cap}%")

    syms = s.get("symbols") or []
    if not syms:
        errs.append("Add at least one symbol")
    if len(syms) > MAX_SYMBOLS:
        errs.append(f"Maximum {MAX_SYMBOLS} symbols")
    bad = [x for x in syms if not SYM_RE.match(str(x))]
    if bad:
        errs.append("Invalid symbol: " + ", ".join(map(str, bad)))
    if len(set(syms)) != len(syms):
        errs.append("Duplicate symbols")
    return errs


def _write_json(path: Path, obj: dict) -> None:
    """Back up the current file, then replace it atomically so a crash never leaves half a file."""
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_setup(s: dict, root: Path = ROOT, *, allow_writes: bool) -> dict:
    """Validate, then write the four config files. Returns {"written": [...], "needs_validate": bool}."""
    if not allow_writes:
        raise SetupError("Saving is disabled. Set DASH_ALLOW_WRITES=true to allow it.")
    clean = {
        "htf": s.get("htf"),
        "ltf": s.get("ltf"),
        "risk_pct": s.get("risk_pct"),
        "symbols": [normalize_symbol(x) for x in (s.get("symbols") or [])],
    }
    errs = validate_setup(clean, root)
    if errs:
        raise SetupError("; ".join(errs))

    root = Path(root)
    cfg = root / "config"
    fw_path, live_path, ops_path = cfg / "framework.json", cfg / "live.json", cfg / "ops.json"
    profile_path = cfg / "profiles" / f"{_profile_name()}.json"

    fw = _load(fw_path)
    eng = _engulfing_entry(fw)
    if eng is None:
        raise SetupError(f"framework.json has no {STRATEGY_TYPE} entry to set HTF and LTF on")
    before = read_setup(root)

    risk_frac = round(clean["risk_pct"] / 100, 6)
    fw["risk_per_trade"] = risk_frac
    eng.setdefault("params", {})
    eng["params"]["htf"] = clean["htf"]
    eng["params"]["ltf"] = clean["ltf"]

    live = _load(live_path)
    live["risk_per_trade"] = risk_frac

    ops = _load(ops_path)
    ops["symbols"] = clean["symbols"]

    written = [fw_path, live_path, ops_path]
    _write_json(fw_path, fw)
    _write_json(live_path, live)
    _write_json(ops_path, ops)
    if profile_path.exists():
        prof = _load(profile_path)
        prof.setdefault("scanner", {})["htf"] = clean["htf"]
        _write_json(profile_path, prof)
        written.append(profile_path)

    after = read_setup(root)
    changed = any(before[k] != after[k] for k in ("htf", "ltf", "risk_pct"))
    return {"written": [p.relative_to(root).as_posix() for p in written], "needs_validate": changed}
