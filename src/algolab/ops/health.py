"""Health checks. Writes state/health.json, alerts on each failing check, exits 1 if anything is down."""
import logging
import shutil
import socket
import sys
import time

import httpx

from algolab.ops import alerts
from algolab.ops.state import (ROOT, kill_switch_on, minutes_since, now_utc,
                               ops_config, read_json, write_json)
from algolab.profile import load_profile

log = logging.getLogger("algolab.health")


def _disk(ops):
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    return free_gb >= ops["health"]["min_free_disk_gb"], f"{free_gb:.1f} GB free"


def _market_data(p, ops):
    files = list((ROOT / p["data_dir"]).glob(f"*_{p['scanner']['htf']}.csv"))
    if not files:
        return False, "no HTF CSV files"
    age = (time.time() - max(f.stat().st_mtime for f in files)) / 60
    limit = ops["pipeline"]["max_data_age_minutes"]
    return age <= limit, f"newest data {age:.0f} min old (limit {limit})"


def _last_cycle():
    last = read_json("last_cycle.json")
    if not last:
        return False, "no cycle has run yet"
    if "error" in last:
        return False, f"last cycle error: {last['error']}"
    age = minutes_since(last["ts"])
    return age <= 45, f"last cycle {age:.0f} min ago"


def _account(ops):
    acct = read_json("account.json")
    if not acct:
        return False, "no account state"
    age = minutes_since(acct["ts"])
    limit = ops["pipeline"]["max_account_age_minutes"]
    return age <= limit, f"account state {age:.0f} min old (limit {limit})"


def _ollama(p):
    o = p["ollama"]
    if not o["url"]:
        return False, "OLLAMA_URL not set"
    headers = {"Authorization": f"Bearer {o['api_key']}"} if o["api_key"] else {}
    httpx.get(f"{o['url']}/api/tags", headers=headers, timeout=5).raise_for_status()
    return True, "reachable"


def _mcp(p):
    port = p["mcp"]["port"]
    with socket.create_connection(("127.0.0.1", port), timeout=2):
        pass
    return True, f"port {port} open"


def run_checks() -> dict:
    p = load_profile()
    ops = ops_config()
    checks = {
        "disk": lambda: _disk(ops),
        "market_data": lambda: _market_data(p, ops),
        "last_cycle": _last_cycle,
        "account_sync": lambda: _account(ops),
        "ollama": lambda: _ollama(p),
    }
    if p["mcp"]["transport"] != "stdio":
        checks["mcp"] = lambda: _mcp(p)

    results = {}
    for name, fn in checks.items():
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results[name] = {"ok": ok, "detail": detail}

    status = {"ts": now_utc().isoformat(),
              "healthy": all(r["ok"] for r in results.values()),
              "kill_switch": kill_switch_on(),
              "checks": results}
    write_json("health.json", status)
    for name, r in results.items():
        if not r["ok"]:
            alerts.send(f"health check failed: {name} - {r['detail']}", "CRITICAL",
                        key=f"health:{name}", cooldown_min=ops["alerts"]["dedupe_minutes"])
    return status


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    status = run_checks()
    log.info("health %s", "OK" if status["healthy"] else "FAILED")
    return 0 if status["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
