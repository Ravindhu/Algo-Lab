"""Live framework pipeline built on the MetaTrader5 package (through the adapter).
One cycle per run, normally every 5 minutes (scheduled with --framework flags baked in).

  1. self-test the MT5 adapter. Missing APIs halt the cycle (fail closed).
  2. read account and open positions. Apply kill switch, drawdown and daily loss guards.
  3. run the framework: manage open positions, reconcile, then let the Bot ask every strategy
     (scalp, engulfing, OCL) for signals. DRY_RUN=true -> alert only. DRY_RUN=false -> place orders.

Only one cycle may run at a time (state/framework.lock). Fills are recorded by the framework runner the
moment they happen, not at the end of the cycle.

Run with:  ALGOLAB_PROFILE=local DRY_RUN=true python -m algolab.live.pipeline --framework
"""
import asyncio
import json
import logging
import sys

from algolab.backtest.validate import validation_gate
from algolab.live.adapter import AccountState, AdapterError, Mt5Adapter
from algolab.live.framework_runner import run_framework
from algolab import strategy as S
from algolab.ops import alerts
from algolab.ops.risk import adjusted_equity, check_limits
from algolab.ops.state import (ROOT, LockBusy, cycle_lock, engage_kill, kill_switch_on,
                               now_utc, ops_config, write_json)
from algolab.profile import load_profile

log = logging.getLogger("algolab.live")
LIVE = {**json.loads((ROOT / "config" / "live.json").read_text()), **S.live_overrides()}


def load_symbols(ops: dict) -> list[str]:
    exported = ROOT / "config" / "symbols.json"
    if exported.exists():
        return json.loads(exported.read_text())["forex"]
    return ops["symbols"]


def guard(acct: AccountState, ops: dict) -> tuple[bool, str]:
    """May NEW entries be placed this cycle? Returns (False, reason) to block entries.

    This never stops position management: framework_cycle always runs managed exits (see run_framework).
    The drawdown and daily-loss baseline is updated first, so the peak and the day's start keep advancing
    even while validation or the kill switch is blocking trades."""
    ok_r, why_r, breach = check_limits(adjusted_equity(acct.equity, acct.net_deposits), ops["risk"])
    if kill_switch_on():
        return False, "kill switch active"
    if breach:
        engage_kill(why_r)
        alerts.send("KILL SWITCH engaged: " + why_r, "CRITICAL")
        return False, why_r
    ok_v, why_v = validation_gate()          # no new entries on unvalidated parameters
    if not ok_v:
        return False, why_v
    if acct.open_positions is None:
        return False, "open positions unknown (MT5 terminal did not answer) - failing closed"
    if not ok_r:
        return False, why_r
    if acct.open_positions >= LIVE["max_open_trades"]:
        return False, "max open trades reached"
    return True, "ok"


async def framework_cycle() -> dict:
    """Framework cycle (every 5 minutes). Open positions are always managed. The guard only decides whether
    new entries may be placed, so a halt never leaves managed exits (trailing stops, close-based stops, prop
    flatten) switched off."""
    p = load_profile()
    ops = ops_config()
    symbols = load_symbols(ops)
    report = {"ts": now_utc().isoformat(), "dry_run": p["dry_run"]}
    async with Mt5Adapter() as ad:
        problems, warnings = await ad.self_test()
        for w in warnings:
            log.warning(w)
        if problems:
            raise AdapterError("MT5 contract failed: " + "; ".join(problems))
        acct = await ad.account()
        ok, reason = guard(acct, ops)
        report.update(may_trade=ok, reason=reason, equity=acct.equity)
        if not ok:
            log.warning("new entries blocked: %s (open positions are still managed)", reason)
        fw = await run_framework(ad, p, symbols, acct.equity, allow_entries=ok, entry_block_reason=reason)
        if fw is not None:
            report["framework"] = fw
    write_json("framework_last_cycle.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        with cycle_lock("framework"):
            asyncio.run(framework_cycle())
        return 0
    except LockBusy as e:
        log.warning("skipped: %s", e)           # another cycle is trading right now
        return 0
    except Exception as e:
        log.exception("framework cycle failed")
        write_json("framework_last_cycle.json",
                   {"ts": now_utc().isoformat(), "error": f"{type(e).__name__}: {e}"})
        alerts.send(f"framework cycle failed: {type(e).__name__}: {e}",
                    "CRITICAL", key="live-error", cooldown_min=60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
