"""One scan-and-review cycle. Run by a systemd timer every 15 minutes.

Fail-closed: if the kill switch is on, the account state is stale or missing,
the market data is stale, or a loss limit is hit, the cycle halts and sends no trade alerts.
This module never places orders. Execution is a separate, explicit step."""
import json
import logging
import sys

from algolab.agent.multi_agent import Orchestrator
from algolab.ops import alerts
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

    equity = float(acct["equity"])
    today = now_utc().date().isoformat()
    risk = read_json("risk.json", {})
    peak = max(float(risk.get("peak", equity)), equity)
    day_start = float(risk["day_start"]) if risk.get("day") == today else equity
    r = ops["risk"]

    if equity <= peak * (1 - r["max_drawdown_kill"]):
        reason = f"max drawdown {r['max_drawdown_kill']:.0%} breached"
        engage_kill(reason)
        alerts.send("KILL SWITCH engaged: " + reason +
                    ". Review, then delete state/KILL to resume.", "CRITICAL")
        return False, reason, acct

    write_json("risk.json", {"peak": peak, "day": today, "day_start": day_start})
    if equity <= day_start * (1 - r["max_daily_loss"]):
        return False, "daily loss limit reached (paused until next UTC day)", acct
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
        for d in result["take"]:
            key = f'{d["symbol"]}|{d["side"]}|{d["time"]}'
            if key in seen:
                continue                     # already alerted for this closed bar
            seen[key] = now_utc().isoformat()
            report["take"].append(d)
        # keep 7 days of dedupe history
        write_json("seen_signals.json", {k: v for k, v in seen.items() if minutes_since(v) < 7 * 24 * 60})
        report["reviewed"] = len(result["reviewed"])

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    (REPORT_DIR / f"cycle_{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    write_json("last_cycle.json", report)

    mode = "DRY RUN" if p["dry_run"] else "LIVE"
    for d in report["take"]:
        alerts.send(f'{d["symbol"]} {d["side"]} on {p["scanner"]["htf"]} | RSI {d["rsi"]} | '
                    f'conf {d["confidence"]} | {d.get("reason", "")} | {mode}', "WARNING")
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
