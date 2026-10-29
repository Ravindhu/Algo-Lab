"""One scan-and-review cycle. Run by a systemd timer every 15 minutes.

Fail-closed: if the kill switch is on, the account state is stale or missing,
the market data is stale, or a loss limit is hit, the cycle halts and sends no trade alerts.
This module never places orders. Execution is a separate, explicit step."""
import json
import logging
import sys

from algolab.agent.multi_agent import Orchestrator
from algolab.ops import alerts
from algolab.ops.risk import adjusted_equity, check_limits
from algolab.ops.state import (ROOT, engage_kill, kill_switch_on, minutes_since,
                               now_utc, ops_config, read_json, write_json)
from algolab.profile import load_profile

log = logging.getLogger("algolab.pipeline")
REPORT_DIR = ROOT / "reports"


def data_age_minutes(p: dict) -> float | None:
    files = list((ROOT / p["data_dir"]).glob(f"*_{p['scanner']['htf']}.csv"))
    if not files:
        return None
    newest = max(f.stat().st_mtime for f in files)
    return (now_utc().timestamp() - newest) / 60


def guard(p: dict, ops: dict) -> tuple[bool, str, dict]:
    """Return (may_trade, reason, account). Any doubt means no trade alerts."""
    if kill_switch_on():
        return False, "kill switch active (state/KILL)", {}

    acct = read_json("account.json")
    if not acct:
        return False, "no account state from the Windows sync", {}
    if minutes_since(acct["ts"]) > ops["pipeline"]["max_account_age_minutes"]:
        return False, "account state is stale", acct

    age = data_age_minutes(p)
    if age is None or age > ops["pipeline"]["max_data_age_minutes"]:
        return False, "market data missing or stale", acct

    equity = adjusted_equity(float(acct["equity"]), float(acct.get("net_deposits", 0.0)))
    ok_r, why_r, breach = check_limits(equity, ops["risk"])
    if breach:
        engage_kill(why_r)
        alerts.send("KILL SWITCH engaged: " + why_r +
                    ". Review, then delete state/KILL to resume.", "CRITICAL")
        return False, why_r, acct
    if not ok_r:
        return False, why_r, acct
    return True, "ok", acct



def run_cycle() -> dict:
    p = load_profile()
    ops = ops_config()
    may_trade, reason, acct = guard(p, ops)
    report = {"ts": now_utc().isoformat(), "may_trade": may_trade,
              "reason": reason, "dry_run": p["dry_run"], "take": []}

    if may_trade:
        result = Orchestrator(p).run(int(acct.get("open_positions", 0)))
        seen = read_json("seen_signals.json", {})
        mode = "DRY RUN" if p["dry_run"] else "LIVE"
        for d in result["take"]:
            key = f'{d["symbol"]}|{d["side"]}|{d["time"]}'
            if key in seen:
                continue                     # already alerted for this closed bar
            text = (f'{d["symbol"]} {d["side"]} on {p["scanner"]["htf"]} | close {d.get("close")} | '
                    f'ARMED at fib {d.get("entry_level", 0):.5f} (floor {d.get("floor", 0):.5f}) | '
                    f'conf {d.get("confidence")} | {d.get("reason", "")} | {mode}')
            delivered = alerts.send(text, "WARNING")
            report["take"].append({**d, "alerted": delivered})
            if delivered:                    # only a delivered alert counts as sent; otherwise retried next cycle
                seen[key] = now_utc().isoformat()
        # keep 7 days of dedupe history
        write_json("seen_signals.json", {k: v for k, v in seen.items() if minutes_since(v) < 7 * 24 * 60})
        report["reviewed"] = len(result["reviewed"])

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    (REPORT_DIR / f"cycle_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    write_json("last_cycle.json", report)
    log.info("cycle done: may_trade=%s reason=%s new_signals=%d",
             may_trade, reason, len(report["take"]))
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        run_cycle()
        return 0
    except Exception as e:
        log.exception("pipeline cycle failed")
        write_json("last_cycle.json", {"ts": now_utc().isoformat(),
                                       "error": f"{type(e).__name__}: {e}"})
        alerts.send(f"pipeline cycle failed: {type(e).__name__}", "CRITICAL",
                    key="pipeline-error", cooldown_min=60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
