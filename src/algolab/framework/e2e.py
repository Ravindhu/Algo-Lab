"""End-to-end offline run of the framework: data -> OCL signals -> prop rules -> sizing -> simulated broker
-> position management -> closures -> report, with invariant checks.

    python -m algolab.framework.e2e --symbols EURUSD,USDJPY --bars 6000 --profile generic_2step_example

Data: CSV files {SYMBOL}_M5.csv and {SYMBOL}_M15.csv in --data-dir (the scanner's format: time, open,
high, low, close). With no files, a SYNTHETIC random walk is used. Synthetic results say nothing about
edge; they only prove the wiring works. The report says which data was used.

Writes state to a temporary directory (never the repo's state/), replaces alerts with a recorder, and
restores everything it patched. Exit code 0 when every invariant holds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from algolab.backtest.engine import load_csv
from algolab.framework.prop import load_prop_config, select_profile
from algolab.framework.replay import SimBroker, run_replay
from algolab.ops import alerts, state as ops_state
from algolab.profile import ROOT

log = logging.getLogger("algolab.e2e")
SPREADS = {"EURUSD": 0.00008, "GBPUSD": 0.00010, "USDJPY": 0.008, "EURJPY": 0.010}
START = {"EURUSD": (1.10, 0.0004), "GBPUSD": (1.27, 0.0005), "USDJPY": (150.0, 0.06), "EURJPY": (165.0, 0.07)}
KNOWN_REASONS = {"SL", "TRAIL", "TP", "DISASTER", "HTF", "PROP", "WEEKEND", "CLOSE", "EOD", "BROKER_CLOSE"}


def synthetic(symbol: str, n: int, seed: int) -> dict[str, pd.DataFrame]:
    start, vol = START.get(symbol, (1.0, 0.0004))
    rng = np.random.default_rng(seed + sum(map(ord, symbol)))
    close = start + np.cumsum(rng.normal(0, vol, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, vol, n)
    low = np.minimum(open_, close) - rng.uniform(0, vol, n)
    idx = pd.date_range("2026-01-05 00:00", periods=n, freq="5min")
    m5 = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    m15 = m5.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return {"M5": m5, "M15": m15}


def load_real(symbol: str, data_dir: Path) -> dict[str, pd.DataFrame]:
    return {"M5": load_csv(data_dir / f"{symbol}_M5.csv"), "M15": load_csv(data_dir / f"{symbol}_M15.csv")}


def build_data(symbols: list[str], bars: int, seed: int, data_dir: Path | None,
               keep_fraction: float = 1.0) -> tuple[dict, pd.DatetimeIndex, str]:
    """keep_fraction < 1 keeps only the first part of real history (validation excludes the sealed tail)."""
    real = data_dir is not None and all(
        (data_dir / f"{s}_M5.csv").exists() and (data_dir / f"{s}_M15.csv").exists() for s in symbols)
    if real:
        per = {s: load_real(s, data_dir) for s in symbols}
        index = per[symbols[0]]["M5"].index
        for s in symbols[1:]:
            index = index.intersection(per[s]["M5"].index)
        index = index.sort_values()
        if keep_fraction < 1.0:
            index = index[: max(int(len(index) * keep_fraction), 1)]
        for s in symbols:
            per[s]["M5"] = per[s]["M5"].loc[index]
        source = f"CSV in {data_dir}"
    else:
        per = {s: synthetic(s, bars, seed) for s in symbols}
        index = per[symbols[0]]["M5"].index
        source = "SYNTHETIC random walk (not market data; proves wiring only)"
    return per, index, source


def check_invariants(sim: SimBroker, curve: list[dict], prop_rules, cfg_ocl: dict) -> list[dict]:
    """Each check returns {name, ok, detail}. A failed check means the run must not be trusted."""
    checks = []

    total = sum(t["pnl"] for t in sim.trades)
    checks.append({"name": "balance reconciles with trades",
                   "ok": abs(sim.start_balance + total - sim.balance) < 1e-6,
                   "detail": f"start {sim.start_balance:.2f} + pnl {total:.2f} vs balance {sim.balance:.2f}"})
    checks.append({"name": "no position left open", "ok": not sim.positions,
                   "detail": f"open: {len(sim.positions)}"})
    checks.append({"name": "all close reasons known",
                   "ok": all(t["reason"] in KNOWN_REASONS for t in sim.trades),
                   "detail": sorted({t["reason"] for t in sim.trades})})

    if prop_rules is not None:
        failed_at = next((pt["t"] for pt in curve if pt["status"] == "FAILED"), None)
        if failed_at is not None:
            late = [o for o in sim.orders if o["time"] >= failed_at]
            checks.append({"name": "no order after prop FAILED", "ok": not late,
                           "detail": f"{len(late)} order(s) after {failed_at}"})
        floor = prop_rules.starting_balance * (1 - prop_rules.max_total_loss_pct / 100)
        breaches = [pt for pt in curve if pt["equity"] <= floor and pt["status"] != "FAILED"]
        checks.append({"name": "equity never below the total-loss floor unless FAILED",
                       "ok": not breaches,
                       "detail": f"floor {floor:.2f}; unflagged breaches: {len(breaches)}"})
        max_lots = prop_rules.max_lots
        bad_size = []
        for o in sim.orders:
            risk = (o["fill"] - o["sl"]) * o["lots"] * CONTRACT_SIZE * 1.0
            limit = o["equity"] * prop_rules.max_risk_per_trade_pct / 100
            if max_lots and o["lots"] > max_lots + 1e-9:
                bad_size.append(f"{o['ticket']} lots {o['lots']} > {max_lots}")
            if o["symbol"][3:6] == "USD" and risk > limit * 1.001:
                bad_size.append(f"{o['ticket']} risk {risk:.2f} > {limit:.2f}")
        checks.append({"name": "sizes within prop risk and lot caps", "ok": not bad_size,
                       "detail": bad_size[:3] or "ok"})
    checks.append({"name": "every order has its broker stop below the fill",
                   "ok": all(o["sl"] < o["fill"] for o in sim.orders),
                   "detail": f"{len(sim.orders)} order(s)"})
    return checks


CONTRACT_SIZE = 100000.0


def run_e2e(symbols: list[str], bars: int = 3000, seed: int = 7, data_dir: Path | None = None,
            profile: str | None = None, prop_enabled: bool = True, balance: float = 10000.0,
            out_dir: Path | None = None, prop_override: dict | None = None,
            warmup: int = 130, strategy_types: tuple[str, ...] = ("ocl",),
            keep_fraction: float = 1.0) -> dict:
    base = json.loads((ROOT / "config" / "framework.json").read_text())
    cfg = {**base, "enabled": True,
           "strategies": [s for s in base["strategies"]
                          if s["type"] in strategy_types and s.get("enabled", True)]}
    cfg["max_spread_atr"] = cfg.get("max_spread_atr", 0.25)

    pcfg = load_prop_config()
    if prop_override is not None:
        pcfg = {"enabled": prop_enabled, "active_profile": "override", "profiles": {"override": {
            "description": "test override", "rules": prop_override}}}
    else:
        pcfg = {**pcfg, "enabled": prop_enabled, "active_profile": profile or pcfg["active_profile"]}
    rules = None
    if prop_enabled:
        _, rules_obj = select_profile(pcfg)
        rules = rules_obj

    per, index, source = build_data(symbols, bars, seed, data_dir, keep_fraction)
    tmp = Path(tempfile.mkdtemp(prefix="algolab-e2e-"))
    old_state_dir, old_send = ops_state.STATE_DIR, alerts.send
    recorded: list[str] = []
    try:
        ops_state.STATE_DIR = tmp / "state"
        alerts.send = lambda text, *a, **k: recorded.append(text) or True
        sim = SimBroker(per, index, {s: SPREADS.get(s, 0.0002) for s in symbols}, balance)
        out = asyncio.run(run_replay(sim, symbols, cfg, pcfg if prop_enabled else None, warmup))
        checks = check_invariants(sim, out["curve"], rules, cfg)
    finally:
        ops_state.STATE_DIR, alerts.send = old_state_dir, old_send

    pnl = [t["pnl"] for t in sim.trades]
    report = {
        "data": source, "symbols": symbols, "bars": len(index), "seed": seed,
        "strategy_types": list(strategy_types),
        "profile": pcfg.get("active_profile") if prop_enabled else None,
        "balance": {"start": sim.start_balance, "end": round(sim.balance, 2)},
        "orders": len(sim.orders), "trades": len(sim.trades),
        "wins": int(sum(1 for x in pnl if x > 0)), "net": round(sum(pnl), 2),
        "reasons": pd.Series([t["reason"] for t in sim.trades]).value_counts().to_dict() if sim.trades else {},
        "framework_closures": pd.Series([e["reason"] for e in out["events"]]).value_counts().to_dict()
        if out["events"] else {},
        "prop_final": out["curve"][-1]["status"] if out["curve"] else None,
        "alerts": len(recorded),
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
        "trade_log": sim.trades,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "e2e_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m algolab.framework.e2e")
    ap.add_argument("--symbols", default="EURUSD,USDJPY")
    ap.add_argument("--bars", type=int, default=6000, help="M5 bars per symbol (synthetic only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--profile", default=None, help="prop profile name (default: active_profile)")
    ap.add_argument("--no-prop", action="store_true", help="run without prop rules")
    ap.add_argument("--balance", type=float, default=10000.0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "e2e"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    rep = run_e2e([s.strip() for s in args.symbols.split(",") if s.strip()], bars=args.bars,
                  seed=args.seed, data_dir=Path(args.data_dir), profile=args.profile,
                  prop_enabled=not args.no_prop, balance=args.balance, out_dir=Path(args.out))
    print(f"data: {rep['data']}  bars: {rep['bars']}  symbols: {', '.join(rep['symbols'])}")
    print(f"profile: {rep['profile']}  prop final: {rep['prop_final']}")
    print(f"orders: {rep['orders']}  trades: {rep['trades']}  wins: {rep['wins']}  net: {rep['net']}")
    print(f"broker exits: {rep['reasons']}")
    print(f"framework closures: {rep['framework_closures']}")
    print(f"balance: {rep['balance']['start']} -> {rep['balance']['end']}")
    for c in rep["checks"]:
        print(f"  [{'OK' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"report: {Path(args.out) / 'e2e_report.json'}")
    print("RESULT:", "PASS" if rep["passed"] else "FAIL")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
