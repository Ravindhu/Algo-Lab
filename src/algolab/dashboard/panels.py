"""Panels for the FastAPI dashboard: status, trades, open positions, risk, signals, equity, alerts, autotune,
tokens, and the validate job.

Standard library only (journal and validate are imported inside the functions that need them), so the logic is
testable without the web stack. Every reader returns an empty value when its file is missing.

Nothing here changes trading settings or places orders. The one write is the validate job: it runs the repo's own
`algolab validate` command in the background and records its outcome in state/validate_job.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any

from algolab.ops import state as st
from algolab.profile import ROOT

WARN_AT = 0.7                 # a limit shows "warn" once 70% of it is used
JOB_STALE_MIN = 90            # a validate job "running" longer than this is treated as left behind
JOB_FILE = "validate_job.json"
_job_lock = threading.Lock()


class JobBusy(RuntimeError):
    pass


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _round(x: Any, nd: int = 2) -> Any:
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def dry_run(root: Path = ROOT) -> bool:
    """DRY_RUN from the environment when set, otherwise from the active profile. Defaults to True."""
    env = os.getenv("DRY_RUN")
    if env is not None:
        return env.strip().lower() != "false"
    name = os.getenv("ALGOLAB_PROFILE", "local")
    return bool(_read(Path(root) / "config" / "profiles" / f"{name}.json", {}).get("dry_run", True))


# ---------- validation and status ----------

def validation(root: Path = ROOT) -> dict:
    """passed / failed / stale / not_run. Stale means the settings changed after the last validation, so the
    report no longer describes what would trade."""
    rep = _read(Path(root) / "reports" / "validation.json")
    if not rep:
        return {"state": "not_run", "passed": False,
                "reasons": ["no validation report yet: run validation"]}
    try:
        from algolab.backtest import validate as v
        current = v.config_fingerprint()
    except Exception:                       # cannot compute the fingerprint: do not claim it is current
        current = None
    if current is None or rep.get("fingerprint") != current:
        return {"state": "stale", "passed": False,
                "reasons": ["settings changed since the last validation: run validation again"]}
    return {"state": "passed" if rep.get("passed") else "failed", "passed": bool(rep.get("passed")),
            "reasons": rep.get("reasons", [])}


def _kill_reason() -> str:
    """The KILL file is "<UTC timestamp> <reason>". Returns only the reason."""
    f = st.STATE_DIR / "KILL"
    try:
        text = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    except OSError:
        return ""
    parts = text.split(" ", 1)
    return parts[1] if len(parts) == 2 else text


def status(root: Path = ROOT) -> dict:
    """Trading state and everything that is stopping it, in one place."""
    kill = st.kill_switch_on()
    val = validation(root)
    fw = st.read_json("framework_last_cycle.json", {}) or {}
    halts: list[str] = []
    if kill:
        halts.append("kill switch on: " + (_kill_reason() or "no reason recorded"))
    if not val["passed"]:
        halts.append(f"validation {val['state']}")
    if fw.get("error"):
        halts.append("last framework cycle failed: " + str(fw["error"]))
    elif fw and fw.get("may_trade") is False:
        halts.append(str(fw.get("reason", "new entries blocked")))
    return {"trading_enabled": not kill, "can_trade": not halts, "kill": kill,
            "kill_reason": _kill_reason() if kill else "",
            "dry_run": dry_run(root), "validation": val, "halts": halts,
            "last_cycle": fw.get("ts"), "cycle_error": fw.get("error")}


# ---------- journal: trades and open positions ----------

def trades(limit: int = 100) -> list[dict]:
    from algolab.ops import journal
    rows = journal.read(limit)[::-1]
    out = []
    for r in rows:
        out.append({"time": r.get("ts", ""), "kind": r.get("kind", ""), "strategy": r.get("strategy", ""),
                    "symbol": r.get("symbol", ""), "side": r.get("side", ""),
                    "entry": r.get("entry", r.get("price", "")), "sl": r.get("sl", ""),
                    "pnl": _round(r.get("pnl")) if r.get("pnl") is not None else
                    ("unknown" if r.get("kind") == "close" else ""),
                    "R": _round(r.get("r")) if r.get("r") is not None else ""})
    return out


def open_positions() -> list[dict]:
    """Framework trades that were opened and have no close line yet. Only rows with a broker ticket are
    considered: a row without one cannot be matched to its close, so it is not shown as open."""
    from algolab.ops import journal
    rows = journal.read()
    closed = {r.get("ticket") for r in rows if r.get("kind") == "close"}
    out = []
    for r in rows:
        if r.get("kind") != "open" or r.get("ticket") is None or r.get("ticket") in closed:
            continue
        out.append({"time": r.get("ts", ""), "ticket": r.get("ticket"), "symbol": r.get("symbol", ""),
                    "side": r.get("side", ""), "strategy": r.get("strategy", ""), "lots": r.get("lots"),
                    "entry": r.get("entry"), "sl": r.get("sl"), "tp": r.get("tp")})
    return out


# ---------- risk ----------

def _ratio_status(ratio: float) -> str:
    return "breach" if ratio >= 1 else ("warn" if ratio >= WARN_AT else "ok")


def risk(root: Path = ROOT) -> dict:
    """Daily loss and drawdown used against the limits in ops.json, and the prop-firm limits when a profile is
    active. Figures come from the last framework cycle's equity, so they lag by one cycle."""
    ops_risk = (_read(Path(root) / "config" / "ops.json", {}) or {}).get("risk", {})
    rk = st.read_json("risk.json", {}) or {}
    fw = st.read_json("framework_last_cycle.json", {}) or {}
    live = st.read_json("live_last_cycle.json", {}) or {}
    equity = fw.get("equity", live.get("equity"))
    if equity is None:
        return {"equity": None, "items": [], "note": "no cycle has run yet"}
    eq = float(equity)
    peak = max(float(rk.get("peak", eq)), eq)
    day_start = float(rk["day_start"]) if rk.get("day") == _today() else eq
    items = []

    def add(name: str, used: float, limit: float | None, note: str = "") -> None:
        if not limit or limit <= 0:
            return
        ratio = max(used, 0.0) / limit
        items.append({"name": name, "used_pct": round(max(used, 0.0) * 100, 2),
                      "limit_pct": round(limit * 100, 2), "status": _ratio_status(ratio), "note": note})

    add("Daily loss (ops)", (day_start - eq) / day_start if day_start else 0,
        ops_risk.get("max_daily_loss"), "since today's first cycle")
    add("Drawdown (ops, kill switch)", (peak - eq) / peak if peak else 0,
        ops_risk.get("max_drawdown_kill"), "from the highest equity seen")

    prop = _read(Path(root) / "config" / "prop.json", {}) or {}
    active = prop.get("active_profile")
    rules = ((prop.get("profiles") or {}).get(active) or {}).get("rules", {}) if active else {}
    if rules:
        add("Daily loss (prop)", (day_start - eq) / day_start if day_start else 0,
            (rules.get("max_daily_loss_pct") or 0) / 100, f"profile {active}"
            + ("" if prop.get("enabled") else ", enforcement off"))
        start_bal = float(rules.get("starting_balance") or 0)
        if start_bal:
            add("Total loss (prop)", (start_bal - eq) / start_bal,
                (rules.get("max_total_loss_pct") or 0) / 100, "vs starting balance")
    return {"equity": round(eq, 2), "items": items, "note": ""}


# ---------- signals, equity, alerts ----------

def signals(limit: int = 50, root: Path = ROOT) -> list[dict]:
    out: list[dict] = []
    for f in sorted(glob(str(Path(root) / "reports" / "cycle_*.json")))[-20:]:
        for hit in (_read(Path(f), {}) or {}).get("take", []):
            out.append({"time": hit.get("time") or "", "symbol": hit.get("symbol"), "side": hit.get("side"),
                        "rsi": hit.get("rsi", ""), "confidence": hit.get("confidence", ""),
                        "reason": hit.get("reason", "")})
    fw = st.read_json("framework_last_cycle.json", {}) or {}
    for r in (fw.get("framework") or {}).get("results", []):
        verdict = "EXECUTED" if r.get("executed") else (r.get("why") or "not placed")
        out.append({"time": fw.get("ts") or "", "symbol": r.get("symbol"), "side": r.get("side"),
                    "rsi": "", "confidence": "", "reason": f"{verdict} ({r.get('strategy', '')})"})
    out.sort(key=lambda x: str(x.get("time") or ""), reverse=True)
    return out[:limit]


def equity(limit: int = 500) -> list[dict]:
    try:
        lines = (st.STATE_DIR / "equity_history.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-limit:]:
        try:
            rec = json.loads(line)
            rows.append({"ts": rec["ts"], "equity": float(rec["equity"])})
        except (ValueError, KeyError, TypeError):
            continue                       # a torn line is skipped, never fatal
    return rows


def alerts() -> list[dict]:
    """Last time each alert type was delivered (alerts.json keeps one timestamp per type)."""
    sent = st.read_json("alerts.json", {}) or {}
    return sorted(({"key": k, "last_sent": v} for k, v in sent.items()),
                  key=lambda r: str(r["last_sent"]), reverse=True)


# ---------- autotune and tokens ----------

BEST_KEYS = ("score", "trades", "rsi_n", "ema_n", "atr_n", "sl_mult", "tp_mult", "ml_threshold")


def autotune(root: Path = ROOT) -> dict:
    """Read-only summary of the last optimizer run. Applying a candidate stays a manual CLI step."""
    rep = _read(Path(root) / "reports" / "autotune_latest.json", {}) or {}
    trials = rep.get("trials") or []
    best = None
    scored = [t for t in trials if isinstance(t.get("score"), (int, float))]
    if scored:
        top = max(scored, key=lambda t: t["score"])
        best = {k: top[k] for k in BEST_KEYS if k in top}
    return {"trials": len(trials), "applied": bool(rep.get("applied")),
            "rolled_back": bool(rep.get("rolled_back")), "reasons": (rep.get("reasons") or [])[:5],
            "best": best, "ran": bool(rep)}


def tokens(root: Path = ROOT) -> dict:
    s = st.read_json("tokens.json", {}) or {}
    limit = int((_read(Path(root) / "config" / "llm.json", {}) or {}).get("daily_token_budget", 0))
    today = s.get("day") == _today()
    used = int(s.get("used", 0)) if today else 0
    return {"day": _today(), "used": used, "limit": limit,
            "ratio": round(used / limit, 3) if limit else 0,
            "calls": int(s.get("calls", 0)) if today else 0,
            "cached": len(s.get("cache", {})) if today else 0}


# ---------- validate job (the only write here) ----------

def validate_job() -> dict:
    return st.read_json(JOB_FILE, None) or {"state": "idle"}


def _job_running(job: dict) -> bool:
    if job.get("state") != "running":
        return False
    try:
        return st.minutes_since(job["started"]) < JOB_STALE_MIN
    except (KeyError, ValueError, TypeError):
        return False


def start_validate(root: Path = ROOT, *, background: bool = True,
                   command: list[str] | None = None) -> dict:
    """Run `algolab validate` (or `command`, for tests) and record the outcome. One job at a time.
    The caller must check DASH_ALLOW_WRITES first."""
    cmd = command or [sys.executable, "-m", "algolab.cli", "validate"]
    with _job_lock:
        if _job_running(validate_job()):
            raise JobBusy("a validation is already running")
        job = {"state": "running", "started": _now(), "finished": None, "returncode": None, "tail": ""}
        st.write_json(JOB_FILE, job)

    def run() -> None:
        try:
            p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=3600)
            done = {"state": "done" if p.returncode == 0 else "failed", "returncode": p.returncode,
                    "tail": (p.stdout + p.stderr)[-2000:]}
        except (OSError, subprocess.SubprocessError) as e:
            done = {"state": "failed", "returncode": None, "tail": f"{type(e).__name__}: {e}"}
        st.write_json(JOB_FILE, {**job, **done, "finished": _now()})

    if background:
        threading.Thread(target=run, daemon=True).start()
    else:
        run()
    return validate_job()
