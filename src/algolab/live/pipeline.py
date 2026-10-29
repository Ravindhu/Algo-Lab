"""Windows pipeline built on the MetaTrader5 package (through the adapter). One cycle per run. The legacy cycle runs every 15 minutes. The framework cycle (--framework) runs every 5 minutes.

  1. self-test the MT5 adapter. Missing APIs halt the cycle (fail closed).
  2. read account and open positions. Apply kill switch, drawdown and daily loss guards.
  3. read candles for each symbol and apply the HTF engulfing + RSI rule.
  4. ask the Ollama analyst for TAKE / SKIP. Keep TAKE above min_confidence.
  5. size from an ATR stop. DRY_RUN=true -> alert only. DRY_RUN=false -> place the order.

Run with:  ALGOLAB_PROFILE=local DRY_RUN=true python -m algolab.live.pipeline
"""
import asyncio
import json
import logging
import math
import sys

import pandas as pd

from algolab.agent.multi_agent import AnalystAgent
from algolab.backtest.validate import validation_gate
from algolab.llm.budget import CachedAnalyst, budget_from_config
from algolab.live.adapter import AccountState, AdapterError, Mt5Adapter
from algolab.live.framework_runner import run_framework
from algolab.live.sizing import lots_from_risk, quote_to_account_rate
from algolab.llm.ollama_client import OllamaClient
from algolab import strategy as S
from algolab.ops import alerts, journal
from algolab.ops.risk import check_limits
from algolab.ops.state import (ROOT, engage_kill, kill_switch_on, now_utc,
                               ops_config, read_json, write_json)
from algolab.profile import load_profile
from algolab.scanner.scanner import scan_frame

log = logging.getLogger("algolab.live")
LIVE = {**json.loads((ROOT / "config" / "live.json").read_text()), **S.live_overrides()}


def load_symbols(ops: dict) -> list[str]:
    exported = ROOT / "config" / "symbols.json"
    if exported.exists():
        return json.loads(exported.read_text())["forex"]
    return ops["symbols"]


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def guard(acct: AccountState, ops: dict) -> tuple[bool, str]:
    if kill_switch_on():
        return False, "kill switch active"
    ok_v, why_v = validation_gate()          # no trading on unvalidated parameters
    if not ok_v:
        return False, why_v
    if acct.open_positions is None:
        return False, "open positions unknown (MT5 terminal did not answer) - failing closed"

    ok_r, why_r, breach = check_limits(acct.equity, ops["risk"])
    if breach:
        engage_kill(why_r)
        alerts.send("KILL SWITCH engaged: " + why_r, "CRITICAL")
        return False, why_r
    if not ok_r:
        return False, why_r
    if acct.open_positions >= LIVE["max_open_trades"]:
        return False, "max open trades reached"
    return True, "ok"


async def find_candidates(ad: Mt5Adapter, symbols: list[str], p: dict) -> tuple[list[dict], int]:
    hits, failures = [], 0
    for s in symbols:
        try:
            df = await ad.candles(s, p["scanner"]["htf"], 300)
        except AdapterError as e:
            failures += 1
            log.error("candles %s: %s", s, e)
            continue
        hit = scan_frame(s, df, p["scanner"])
        if hit:
            hits.append(hit)
    return hits, failures


def ml_accepts(hit: dict, df: pd.DataFrame, p: dict) -> bool:
    """ML filter. Threshold 0 = off. Threshold set but no model = reject (fail closed)."""
    thr = float(p["strategy"]["ml_threshold"])
    if thr <= 0:
        return True
    from algolab.ml import features as F
    from algolab.ml import model as M
    model = M.load()
    if model is None:
        return False
    feats = F.build(df, p["strategy"]["rsi_n"], p["strategy"]["atr_n"])
    row = feats.reindex([pd.to_datetime(hit["time"])])
    if row[F.FEATURES].isna().any(axis=None):
        return False
    return float(M.proba(model, row)[0]) >= thr


def plan_trade(df: pd.DataFrame, info: dict, equity: float, rate: float) -> dict | None:
    """rate = account currency per unit of the pair's quote currency (1.0 for USD-quoted pairs)."""
    a = float(atr(df, LIVE["atr_n"]).iloc[-2])          # last closed bar
    if not a > 0:
        return None
    sl_dist, tp_dist = LIVE["sl_mult"] * a, LIVE["tp_mult"] * a
    risk_money = equity * LIVE["risk_per_trade"]
    lots = lots_from_risk(risk_money, sl_dist, info["contract_size"], rate, LIVE["max_lots"],
                          volume_step=0.01)
    if lots < info["volume_min"]:
        return None                                      # risk too small for broker minimum
    return {"sl_dist": sl_dist, "tp_dist": tp_dist, "lots": round(lots, 2)}


def execute(ad: Mt5Adapter, hit: dict, plan: dict, info: dict, dry_run: bool) -> dict:
    sign = 1 if hit["side"] == "BUY" else -1
    t = ad.tick(hit["symbol"])
    entry = t["ask"] if sign > 0 else t["bid"]
    digits = info["digits"]
    sl = round(entry - sign * plan["sl_dist"], digits)
    tp = round(entry + sign * plan["tp_dist"], digits)
    summary = {"symbol": hit["symbol"], "side": hit["side"], "entry": entry,
               "sl": sl, "tp": tp, "lots": plan["lots"]}
    if dry_run:
        return {**summary, "executed": False}
    res = ad.market_order(hit["symbol"], hit["side"], plan["lots"], sl, tp,
                          LIVE["magic"], LIVE["deviation_points"])
    return {**summary, "executed": True, **res}


async def cycle() -> dict:
    p = load_profile()
    ops = ops_config()
    symbols = load_symbols(ops)
    report = {"ts": now_utc().isoformat(), "dry_run": p["dry_run"], "results": []}

    async with Mt5Adapter() as ad:
        problems, warnings = await ad.self_test()
        for w in warnings:
            log.warning(w)
        if problems:
            raise AdapterError("MT5 contract failed: " + "; ".join(problems))

        acct = await ad.account()
        ok, reason = guard(acct, ops)
        report.update(may_trade=ok, reason=reason, equity=acct.equity,
                      open_positions=acct.open_positions)
        if not ok:
            write_json("live_last_cycle.json", report)
            log.warning("halted: %s", reason)
            return report

        hits, failures = await find_candidates(ad, symbols, p)
        if failures:
            alerts.send(f"{failures} symbol(s) failed to load candles", "WARNING",
                        key="live-candles", cooldown_min=60)

        analyst = CachedAnalyst(
            AnalystAgent(OllamaClient(p["ollama"]["url"], p["ollama"]["api_key"],
                                      p["ollama"]["model"])),
            budget_from_config())
        seen = read_json("seen_signals.json", {})
        open_now = acct.open_positions
        account_ccy = ad.account_currency()

        for hit in hits:
            key = f'{hit["symbol"]}|{hit["side"]}|{hit["time"]}'
            if key in seen:
                continue
            seen[key] = now_utc().isoformat()
            write_json("seen_signals.json", seen)     # mark before acting: no duplicate orders on retry

            verdict = analyst.run(hit)
            if verdict.get("decision") != "TAKE" or float(verdict.get("confidence", 0)) < LIVE["min_confidence"]:
                continue
            if open_now >= LIVE["max_open_trades"]:
                break

            df = await ad.candles(hit["symbol"], p["scanner"]["htf"], 300)
            if not ml_accepts(hit, df, p):
                log.info("ML filter rejected %s %s", hit["symbol"], hit["side"])
                continue
            info = ad.symbol_info(hit["symbol"])
            rate = quote_to_account_rate(ad, hit["symbol"], account_ccy)
            if rate is None:
                log.warning("skip %s: no %s conversion pair (fail closed)", hit["symbol"], account_ccy)
                continue
            plan = plan_trade(df, info, acct.equity, rate)
            if plan is None:
                log.info("skip %s: size below broker minimum", hit["symbol"])
                continue

            result = execute(ad, hit, plan, info, p["dry_run"])
            if result["executed"]:
                open_now += 1
                journal.record_open(source="legacy", strategy="engulfing", symbol=result["symbol"],
                                    side=result["side"], ticket=result.get("ticket"), lots=result["lots"],
                                    entry=result["entry"], sl=result["sl"], tp=result["tp"])
            report["results"].append({**result, "reason": verdict.get("reason", "")})
            mode = "DRY RUN" if p["dry_run"] else "LIVE"
            alerts.send(f'{result["symbol"]} {result["side"]} {result["lots"]} lots | '
                        f'entry {result["entry"]} SL {result["sl"]} TP {result["tp"]} | {mode}',
                        "WARNING")

    write_json("live_last_cycle.json", report)
    log.info("cycle done: %d result(s)", len(report["results"]))
    return report


def _journal_framework(fw: dict) -> None:
    for rec in fw.get("results", []):
        if rec.get("executed"):
            journal.record_open(source="framework", strategy=rec["strategy"], symbol=rec["symbol"],
                                side=rec["side"], ticket=rec.get("ticket"), lots=rec["lots"],
                                entry=float(rec.get("fill_price") or rec.get("entry") or 0.0),
                                sl=rec["sl"], tp=rec.get("tp") or None)
    for ev in fw.get("events", []):
        journal.record_close(ev)


async def framework_cycle() -> dict:
    """Framework strategies on their own schedule (every 5 minutes), with the same guard as the full cycle.
    The full cycle no longer runs the framework, so each order is placed once."""
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
            write_json("framework_last_cycle.json", report)
            log.warning("framework halted: %s", reason)
            return report
        fw = await run_framework(ad, p, symbols, acct.equity)
        if fw is not None:
            report["framework"] = fw
            _journal_framework(fw)
    write_json("framework_last_cycle.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    framework_only = "--framework" in (sys.argv[1:] if argv is None else argv)
    name = "framework_last_cycle.json" if framework_only else "live_last_cycle.json"
    try:
        asyncio.run(framework_cycle() if framework_only else cycle())
        return 0
    except Exception as e:
        log.exception("%s cycle failed", "framework" if framework_only else "live")
        write_json(name, {"ts": now_utc().isoformat(), "error": f"{type(e).__name__}: {e}"})
        alerts.send(f"{'framework' if framework_only else 'live'} cycle failed: "
                    f"{type(e).__name__}: {e}", "CRITICAL", key="live-error", cooldown_min=60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
