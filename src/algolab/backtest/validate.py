"""Out-of-sample validation gate.

Two checks, both required before the live pipeline may trade:

  1. Legacy rule (H4 engulfing + Hilega-Milega, via the fixed backtest): the live parameters are tested on
     several folds of history, without fitting.
  2. Framework replay: every ENABLED framework strategy that the replay supports is run through the same
     run_framework() code path the live cycle uses, on REAL M5/M15 data (never synthetic data), with the
     broker simulated. It must pass the invariant checks and clear trade-count and profit-factor thresholds.
     An enabled strategy the replay cannot test fails the gate, so nothing untested can trade.

The result goes to reports/validation.json and is bound to a fingerprint of every setting that affects trading
(strategy parameters, framework config, enabled strategies, costs). Changing any of those makes the report stale
until `algolab validate` runs again. Passing does not guarantee future profit. It only stops the bot from trading
settings that have not held up on data they were not tuned on."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from algolab.backtest.engine import Costs, Params, load_all, run_backtest, trade_metrics
from algolab.ops.state import now_utc
from algolab.profile import ROOT, load_profile

WARMUP_BARS = 300
REPORT = ROOT / "reports" / "validation.json"
REPLAY_TYPES = ("scalp", "ocl", "ocl_sell_fail_top", "ocl_sell_fail_bottom", "discovered")
FRAMEWORK_CONFIG = ROOT / "config" / "framework.json"
REPLAY_DEFAULTS = {"framework_min_trades": 30, "framework_min_profit_factor": 1.2,
                   "framework_replay_bars": 6000, "framework_replay_seed": 7}


def load_config() -> tuple[Params, Costs, dict]:
    from algolab import strategy as S
    costs = json.loads((ROOT / "config" / "backtest.json").read_text())["costs"]
    cfg = json.loads((ROOT / "config" / "validation.json").read_text())
    return S.params(), Costs(**costs), cfg


def config_fingerprint() -> str:
    """Hash of every setting that affects what trades live. Any change makes an old validation stale."""
    from algolab import strategy as S
    cur = S.load()
    fw = json.loads(FRAMEWORK_CONFIG.read_text())
    payload = {
        "strategy": {k: cur[k] for k in S.KEYS}, "ml_threshold": cur.get("ml_threshold"),
        "framework": {k: fw.get(k) for k in ("enabled", "timeframe", "max_open", "risk_per_trade",
                                              "max_lots", "max_spread_atr", "max_margin_use", "magic")},
        "framework_strategies": [[e["type"], bool(e.get("enabled", True)), e.get("params", {})]
                                 for e in fw["strategies"]],
        "costs": json.loads((ROOT / "config" / "backtest.json").read_text())["costs"],
        "validation": json.loads((ROOT / "config" / "validation.json").read_text()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def validate(dfs: dict[str, pd.DataFrame], p: Params, c: Costs, cfg: dict) -> dict:
    parts, per_fold = [], []
    for sym, df in dfs.items():
        for idx in np.array_split(np.arange(len(df)), cfg["folds"]):
            start = max(idx[0] - WARMUP_BARS, 0)              # indicator warm-up, not counted
            tr = run_backtest(df.iloc[start:idx[-1] + 1], p, c)
            if not tr.empty:
                tr = tr[tr["entry_time"] >= df.index[idx[0]]]
            parts.append(tr.assign(symbol=sym))
            per_fold.append({"symbol": sym, "bars": int(len(idx)),
                             **trade_metrics(tr, c.initial_balance)})

    combined = pd.concat(parts, ignore_index=True)
    agg = trade_metrics(combined, c.initial_balance)
    positive = sum(1 for f in per_fold if f["trades"] > 0 and f["net"] > 0)
    ratio = positive / len(per_fold) if per_fold else 0.0

    reasons = []
    if agg["trades"] < cfg["min_trades_total"]:
        reasons.append(f"only {agg['trades']} trades (need {cfg['min_trades_total']})")
    pf = agg["profit_factor"]
    if agg["trades"] and pf is not None and pf < cfg["min_profit_factor"]:
        reasons.append(f"profit factor {pf} below {cfg['min_profit_factor']}")
    if agg["expectancy"] <= 0:
        reasons.append("expectancy not positive")
    if ratio < cfg["min_positive_fold_ratio"]:
        reasons.append(f"only {ratio:.0%} of folds positive (need {cfg['min_positive_fold_ratio']:.0%})")

    return {"passed": not reasons, "reasons": reasons or ["all gates passed"], "params": asdict(p),
            "costs": asdict(c), "aggregate": agg, "positive_fold_ratio": round(ratio, 3),
            "folds": per_fold, "symbols": sorted(dfs)}


def sealed_fraction() -> float:
    """Share of history at the end reserved for `algolab sealed`. Validation never reads it."""
    return float(json.loads((ROOT / "config" / "autotune.json").read_text()).get("sealed_frac", 0.1))


def pre_sealed(dfs: dict[str, pd.DataFrame], keep: float) -> dict[str, pd.DataFrame]:
    return {s: df.iloc[: max(int(len(df) * keep), 1)] for s, df in dfs.items()}


def framework_replay(cfg: dict, keep_fraction: float = 1.0) -> dict:
    """Replay the enabled framework strategies on real data. Returns passed=False with reasons when anything
    enabled cannot be tested, when data is missing, or when results miss the thresholds."""
    from algolab.framework.e2e import run_e2e

    fw = json.loads(FRAMEWORK_CONFIG.read_text())
    enabled = [e["type"] for e in fw["strategies"] if fw.get("enabled") and e.get("enabled", True)]
    if not enabled:
        return {"passed": True, "reasons": ["no framework strategies enabled"], "types": [],
                "trades": 0, "net": 0.0, "profit_factor": None}
    untested = sorted(set(enabled) - set(REPLAY_TYPES))
    if untested:
        return {"passed": False, "types": enabled, "trades": 0, "net": 0.0, "profit_factor": None,
                "reasons": [f"no replay validation for enabled strategy {t}: disable it or add replay support"
                            for t in untested]}

    symbols = load_profile_symbols()
    data = ROOT / "data"
    missing = [s for s in symbols for tf in ("M5", "M15") if not (data / f"{s}_{tf}.csv").exists()]
    if missing:
        return {"passed": False, "types": enabled, "trades": 0, "net": 0.0, "profit_factor": None,
                "reasons": [f"real M5/M15 data missing for replay: {sorted(set(missing))[:5]}"]}

    defaults = {**REPLAY_DEFAULTS, **{k: cfg[k] for k in REPLAY_DEFAULTS if k in cfg}}
    rep = run_e2e(symbols, bars=int(defaults["framework_replay_bars"]),
                  seed=int(defaults["framework_replay_seed"]), data_dir=data,
                  prop_enabled=False, strategy_types=tuple(enabled), keep_fraction=keep_fraction)
    pnls = [float(t["pnl"]) for t in rep["trade_log"]]
    gains, losses = sum(x for x in pnls if x > 0), -sum(x for x in pnls if x <= 0)
    pf = round(gains / losses, 3) if losses > 0 else None
    net = round(sum(pnls), 2)

    reasons = []
    if not str(rep["data"]).startswith("CSV"):
        reasons.append("replay did not use real data (synthetic data cannot validate a strategy)")
    failed = [c["name"] for c in rep["checks"] if not c["ok"]]
    if failed:
        reasons.append("replay invariant checks failed: " + ", ".join(failed))
    if len(pnls) < int(defaults["framework_min_trades"]):
        reasons.append(f"replay produced {len(pnls)} trades (need {defaults['framework_min_trades']})")
    if net <= 0:
        reasons.append(f"replay net {net} not positive")
    if pf is not None and pf < float(defaults["framework_min_profit_factor"]):
        reasons.append(f"replay profit factor {pf} below {defaults['framework_min_profit_factor']}")
    return {"passed": not reasons, "reasons": reasons or ["replay gates passed"], "types": enabled,
            "trades": len(pnls), "net": net, "profit_factor": pf, "data": rep["data"]}


def load_profile_symbols() -> list[str]:
    exported = ROOT / "config" / "symbols.json"
    if exported.exists():
        return json.loads(exported.read_text())["forex"]
    return json.loads((ROOT / "config" / "ops.json").read_text())["symbols"]


def run_validation() -> dict:
    p, c, cfg = load_config()
    keep = 1.0 - sealed_fraction()                 # the sealed tail is never used here
    dfs = pre_sealed(load_all(load_profile()["scanner"]["htf"]), keep)
    legacy = validate(dfs, p, c, cfg)
    framework = framework_replay(cfg, keep_fraction=keep)
    reasons = [f"legacy rule: {r}" for r in legacy["reasons"] if r != "all gates passed"]
    reasons += [f"framework replay: {r}" for r in framework["reasons"] if "passed" not in r]
    passed = legacy["passed"] and framework["passed"]
    result = {**legacy, "passed": passed, "reasons": reasons or ["all gates passed"],
              "framework": framework, "fingerprint": config_fingerprint(),
              "ts": now_utc().isoformat()}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, default=str))
    return result


def validation_gate() -> tuple[bool, str]:
    """Used by the live pipeline. Fail closed when the report is missing, failed, old, or for other settings."""
    max_age = json.loads((ROOT / "config" / "validation.json").read_text())["max_age_days"]
    if not REPORT.exists():
        return False, "strategy not validated (run: algolab validate)"
    res = json.loads(REPORT.read_text())
    age = (now_utc() - datetime.fromisoformat(res["ts"])).days
    if age > max_age:
        return False, f"validation is {age} days old (run: algolab validate)"
    if res.get("fingerprint") != config_fingerprint():
        return False, "settings changed since validation (run: algolab validate)"
    if not res["passed"]:
        return False, "validation failed: " + "; ".join(res["reasons"])
    return True, "validated"
