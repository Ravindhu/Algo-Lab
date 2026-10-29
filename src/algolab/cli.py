"""`algolab` command. One entry point for setup checks, validation, optimization, dashboard, status and kill switch.

  algolab doctor      check install, config, data and validation
  algolab backtest    run the live rule on history and print metrics
  algolab validate    out-of-sample gate (required before live trading)
  algolab optimize    parameter search (report only)
  algolab dashboard   start the web dashboard
  algolab status      one-screen summary
  algolab kill        stop all trading now
  algolab resume      allow trading again (after you have reviewed the cause)
  algolab sealed      one-time test on the sealed final slice (never used by the optimizer)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from algolab.ops.state import STATE_DIR, engage_kill, kill_switch_on, read_json
from algolab.profile import ROOT, load_profile

REQUIRED_CONFIG = ["config/strategy.json", "config/autotune.json", "config/backtest.json",
                   "config/validation.json", "config/live.json", "config/llm.json",
                   "config/ops.json"]


def cmd_doctor(_) -> int:
    from algolab.backtest.validate import validation_gate
    from algolab.live.adapter import _installed_version
    from algolab.profile import ROOT as root

    p = load_profile()
    rows: list[tuple[bool, str, str]] = []
    rows.append((sys.version_info >= (3, 13), "python >= 3.13", sys.version.split()[0]))
    if sys.platform == "win32":
        ver = _installed_version()
        rows.append((ver != "not installed", "MetaTrader5 package (Windows)", ver))
    else:
        rows.append((True, "MetaTrader5 package (Windows only)", "not required on this OS"))
    for f in REQUIRED_CONFIG:
        rows.append(((root / f).exists(), f, "found" if (root / f).exists() else "missing"))
    rows.append((bool(p["ollama"]["url"]), "OLLAMA_URL set", p["ollama"]["url"] or "not set"))
    alert = bool(os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("ALERT_WEBHOOK_URL"))
    rows.append((alert, "alert channel configured", "telegram/webhook" if alert else "none"))
    csvs = len(list((root / p["data_dir"]).glob("*.csv")))
    rows.append((csvs > 0, "market data files", f"{csvs} csv"))
    ok_v, why_v = validation_gate()
    rows.append((ok_v, "strategy validation", why_v))
    from algolab.optimize.autotune import framework_drift
    drift = framework_drift()
    rows.append((not drift, "framework params match strategy", "; ".join(drift) or "in sync"))
    rows.append((p["dry_run"], "DRY_RUN (safe default is true)",
                 "true" if p["dry_run"] else "FALSE - live orders enabled"))

    for ok, name, detail in rows:
        print(f"{'OK  ' if ok else 'FAIL'} {name:<32} {detail}")
    return 0 if all(r[0] for r in rows) else 1


def cmd_backtest(_) -> int:
    from algolab.backtest.engine import Costs, load_all, run_backtest, trade_metrics
    import pandas as pd

    from algolab import strategy as S
    p = S.params()
    c = Costs(**json.loads((ROOT / "config" / "backtest.json").read_text())["costs"])
    dfs = load_all(load_profile()["scanner"]["htf"])
    tr = pd.concat([run_backtest(df, p, c).assign(symbol=s) for s, df in dfs.items()],
                   ignore_index=True)
    print(json.dumps(trade_metrics(tr, c.initial_balance), indent=2))
    return 0


def cmd_validate(_) -> int:
    from algolab.backtest.validate import run_validation
    r = run_validation()
    print(json.dumps({"passed": r["passed"], "reasons": r["reasons"],
                      "aggregate": r["aggregate"],
                      "positive_fold_ratio": r["positive_fold_ratio"]}, indent=2))
    return 0 if r["passed"] else 2


def cmd_sealed(args) -> int:
    from algolab.optimize.autotune import run_sealed_test
    r = run_sealed_test(force=getattr(args, "force", False))
    print(json.dumps({"passed": r["passed"], "already_run": r.get("already_run", False),
                      "reasons": r["reasons"], "metrics": r.get("metrics")}, indent=2, default=str))
    return 0 if r["passed"] else 2


def cmd_optimize(args) -> int:
    from algolab.optimize.autotune import run_autotune
    r = run_autotune(approve=getattr(args, "apply", False))
    print(json.dumps({"applied": r["applied"], "rolled_back": r["rolled_back"],
                      "reasons": r["reasons"], "current_holdout": r["current"]["holdout"],
                      "candidate_holdout": r["candidate"]["holdout"],
                      "candidate_params": r["candidate"]["params"],
                      "trials": r["trials_total"]}, indent=2, default=str))
    return 0


def cmd_dashboard(_) -> int:
    from algolab.dashboard.app import main as dash_main
    dash_main()
    return 0


def cmd_select(args) -> int:
    from algolab.optimize.selection import apply_selection
    r = apply_selection(dry_run=not getattr(args, "apply", False))
    print(json.dumps({"top_n": r["top_n"], "keep": r["keep"],
                      "changes": r["changes"], "ranks": r["ranks"]},
                     indent=2, default=str))
    return 0


def cmd_status(_) -> int:
    live = read_json("live_last_cycle.json", {})
    health = read_json("health.json", {})
    tokens = read_json("tokens.json", {})
    print(json.dumps({
        "kill_switch": kill_switch_on(),
        "live": {k: live.get(k) for k in ("ts", "may_trade", "reason", "equity", "open_positions", "error")},
        "health": health.get("healthy", "n/a (no VPS checks on this machine)"),
        "tokens": {k: tokens.get(k) for k in ("day", "used", "calls", "cached")},
    }, indent=2, default=str))
    return 0


def cmd_kill(_) -> int:
    engage_kill("manual kill (algolab kill)")
    print("kill switch engaged. Trading stops at the next cycle.")
    return 0


def cmd_resume(_) -> int:
    f = STATE_DIR / "KILL"
    if f.exists():
        f.unlink()
        print("kill switch removed. Trading may resume at the next cycle.")
    else:
        print("kill switch was not set.")
    return 0


COMMANDS = {"doctor": cmd_doctor, "backtest": cmd_backtest, "validate": cmd_validate,
            "sealed": cmd_sealed,
            "optimize": cmd_optimize, "dashboard": cmd_dashboard, "status": cmd_status,
            "kill": cmd_kill, "resume": cmd_resume, "select": cmd_select}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="algolab", description="AlgoLab command line")
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--force", action="store_true",
                        help="sealed: run again for the same parameters (weakens the sealed test)")
    parser.add_argument("--apply", action="store_true",
                        help="select: write framework.json (default: report only)")
    args = parser.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
