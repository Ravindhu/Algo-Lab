"""Out-of-sample validation gate.

Splits every symbol's history into folds and tests the LIVE parameters on each fold, without fitting.
The result goes to reports/validation.json. The live pipeline trades only while that file says
passed and is recent. Passing does not guarantee future profit. It only stops the bot from trading
parameters that have not held up on data they were not tuned on."""
from __future__ import annotations

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


def load_config() -> tuple[Params, Costs, dict]:
    from algolab import strategy as S
    costs = json.loads((ROOT / "config" / "backtest.json").read_text())["costs"]
    cfg = json.loads((ROOT / "config" / "validation.json").read_text())
    return S.params(), Costs(**costs), cfg


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

    return {"ts": now_utc().isoformat(), "passed": not reasons,
            "reasons": reasons or ["all gates passed"], "params": asdict(p),
            "costs": asdict(c), "aggregate": agg, "positive_fold_ratio": round(ratio, 3),
            "folds": per_fold, "symbols": sorted(dfs)}


def run_validation() -> dict:
    p, c, cfg = load_config()
    dfs = load_all(load_profile()["scanner"]["htf"])
    result = validate(dfs, p, c, cfg)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, default=str))
    return result


def validation_gate() -> tuple[bool, str]:
    """Used by the live pipeline. Fail closed when the report is missing, failed or old."""
    max_age = json.loads((ROOT / "config" / "validation.json").read_text())["max_age_days"]
    if not REPORT.exists():
        return False, "strategy not validated (run: algolab validate)"
    res = json.loads(REPORT.read_text())
    age = (now_utc() - datetime.fromisoformat(res["ts"])).days
    if age > max_age:
        return False, f"validation is {age} days old (run: algolab validate)"
    if not res["passed"]:
        return False, "validation failed: " + "; ".join(res["reasons"])
    return True, "validated"
