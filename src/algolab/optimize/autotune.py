"""Automatic tuning: Optuna search for strategy parameters and the ML threshold, with guards.

  1. Split history into three parts: search (first 70%), holdout (next 20%), and a SEALED final 10%.
     The sealed part is never used by the optimizer. It is only evaluated by `run_sealed_test`, once per
     parameter set. Holdout reuse is counted in state/holdout_ledger.json; after max_holdout_uses
     selections on the same window, the candidate can no longer be applied.
  2. Optuna (TPE) proposes parameters and an ML threshold (0.0 = ML off).
     Inside each trial, the ML model is trained on the early part of the search trades and scored
     on the later part, so the trial never sees the data it is scored on.
  3. The best set and the current set are both scored on the holdout.
  4. The candidate is applied only if it passes the holdout gate and beats the current set by the
     required margin.
  5. It is written to config/strategy.json (backup kept), the live model is retrained, the shared
     parameters are synced into config/framework.json, and the formal validation runs again. If that
     fails, the old parameters, model and framework file are restored.
  6. Results go to reports/autotune_latest.json, and an alert is sent.

Nothing here places orders. It only changes the parameter file that the live pipeline reads."""
from __future__ import annotations

import json
import math
from dataclasses import asdict

import pandas as pd

from algolab import strategy as S
from algolab.backtest.engine import Costs, Params, load_all, run_backtest, trade_metrics
from algolab.backtest.validate import run_validation
from algolab.ml import features as F
from algolab.ml import model as M
from algolab.ops import alerts
from algolab.ops.state import STATE_DIR, now_utc, read_json, write_json
from algolab.profile import ROOT, load_profile

WARMUP = 300
REPORTS = ROOT / "reports"
ML_CHOICES = [0.0, 0.52, 0.55, 0.60, 0.65]
FAILED = -1e12


def _cfg() -> dict:
    return json.loads((ROOT / "config" / "autotune.json").read_text())


def _costs() -> Costs:
    return Costs(**json.loads((ROOT / "config" / "backtest.json").read_text())["costs"])


def _split(dfs: dict[str, pd.DataFrame], frac: float, sealed_frac: float):
    """search: [0, frac). hold: [frac, 1 - sealed_frac). sealed: [1 - sealed_frac, end).
    Each later part starts WARMUP bars early so indicators are warm; trades before its start are dropped."""
    search, hold, hold_start, sealed, sealed_start = {}, {}, {}, {}, {}
    for s_, df in dfs.items():
        cut1 = int(len(df) * frac)
        cut2 = int(len(df) * (1 - sealed_frac))
        search[s_] = df.iloc[:cut1]
        hold[s_] = df.iloc[max(cut1 - WARMUP, 0):cut2]
        hold_start[s_] = df.index[cut1]
        sealed[s_] = df.iloc[max(cut2 - WARMUP, 0):]
        sealed_start[s_] = df.index[cut2]
    return search, hold, hold_start, sealed, sealed_start


def _trades(dfs, p: Params, c: Costs, start_at: dict | None = None):
    parts, feats = [], {}
    for s, df in dfs.items():
        tr = run_backtest(df, p, c)
        if start_at is not None and not tr.empty:
            tr = tr[tr["entry_time"] >= start_at[s]]
        if not tr.empty:
            parts.append(tr.assign(symbol=s))
        feats[s] = F.build(df, p.rsi_n, p.atr_n)
    tr = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=["entry_time", "signal_time", "pnl", "symbol"]))
    return tr, feats


def _inner_score(tr: pd.DataFrame, feats, thr: float, cfg: dict):
    """Score = expectancy * sqrt(trades) on the later part of the search data.
    The ML model is fitted only on the earlier part."""
    if tr.empty:
        return None, 0
    tr = tr.sort_values("entry_time")
    cut = int(len(tr) * cfg["inner_fit_frac"])
    fit_part, eval_part = tr.iloc[:cut], tr.iloc[cut:]
    if thr > 0:
        X, y = M.dataset(fit_part, feats)
        if len(X) < M.MIN_TRADES or y.nunique() < 2:
            return None, 0
        model = M.fit(X, y)
        Xe, _ = M.dataset(eval_part, feats)
        if Xe.empty:
            return None, 0
        accepted = eval_part.loc[Xe.index][M.proba(model, Xe) >= thr]
    else:
        accepted = eval_part
    n = len(accepted)
    if n < cfg["min_eval_trades"]:
        return None, n
    return float(accepted["pnl"].mean()) * math.sqrt(n), n


def _objective(search, c: Costs, cfg: dict):
    def objective(trial) -> float:
        p = Params(
            rsi_n=trial.suggest_int("rsi_n", 5, 14),
            ema_n=trial.suggest_int("ema_n", 2, 5),
            atr_n=trial.suggest_int("atr_n", 7, 21),
            sl_mult=trial.suggest_float("sl_mult", 0.8, 2.5),
            tp_mult=trial.suggest_float("tp_mult", 1.5, 4.5),
        )
        thr = trial.suggest_categorical("ml_threshold", ML_CHOICES)
        tr, feats = _trades(search, p, c)
        score, n = _inner_score(tr, feats, thr, cfg)
        trial.set_user_attr("trades", n)
        return FAILED if score is None else score
    return objective


def _evaluate(p: Params, thr: float, search, hold, hold_start, c: Costs):
    """Fit on search data, then score on the holdout. Returns (metrics or None, model or None)."""
    model = None
    if thr > 0:
        tr_s, feats_s = _trades(search, p, c)
        X, y = M.dataset(tr_s, feats_s)
        try:
            model = M.fit(X, y)
        except ValueError:
            return None, None
    tr_h, feats_h = _trades(hold, p, c, start_at=hold_start)
    accepted = tr_h
    if thr > 0 and not tr_h.empty:
        Xh, _ = M.dataset(tr_h, feats_h)
        accepted = tr_h.loc[Xh.index][M.proba(model, Xh) >= thr]
    return trade_metrics(accepted, c.initial_balance), model


def _gate(m: dict, cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    if m["trades"] < cfg["min_holdout_trades"]:
        reasons.append(f"holdout trades {m['trades']} below {cfg['min_holdout_trades']}")
    if m["expectancy"] <= 0:
        reasons.append("holdout expectancy not positive")
    pf = m["profit_factor"]
    if pf is not None and pf < cfg["min_profit_factor"]:
        reasons.append(f"holdout profit factor {pf} below {cfg['min_profit_factor']}")
    return not reasons, reasons


def _better(new: dict, cur: dict, cfg: dict) -> bool:
    ne, ce = new["expectancy"], cur["expectancy"]
    if ce <= 0:
        return ne > ce
    return ne >= ce * (1 + cfg["min_improvement_pct"] / 100)


def _trials(study) -> list[dict]:
    out = []
    for t in study.trials:
        if t.state.name == "COMPLETE" and t.value > FAILED:
            out.append({"number": t.number, "score": round(t.value, 4),
                        "trades": t.user_attrs.get("trades"), **t.params})
    return out


LEDGER = STATE_DIR / "holdout_ledger.json"
FW_CONFIG = ROOT / "config" / "framework.json"
# Parameters both the legacy path (strategy.json) and the framework's scalp strategy use, with the same meaning.
SHARED_KEYS = ("rsi_n", "atr_n", "sl_mult", "tp_mult")


def _window_key(hold_start: dict) -> str:
    return "|".join(f"{k}={v.isoformat()}" for k, v in sorted(hold_start.items()))


def _ledger_uses(key: str) -> int:
    return int(read_json("holdout_ledger.json", {}).get(key, 0))


def _ledger_bump(key: str) -> int:
    led = read_json("holdout_ledger.json", {})
    led[key] = int(led.get(key, 0)) + 1
    write_json("holdout_ledger.json", led)
    return led[key]


def framework_drift() -> list[str]:
    """Shared parameters that differ between strategy.json and the framework's scalp strategy."""
    cur = S.load()
    fw = json.loads(FW_CONFIG.read_text())
    scalp = next((e["params"] for e in fw["strategies"] if e["type"] == "scalp"), None)
    if scalp is None:
        return []
    return [f"{k}: strategy.json={cur[k]} framework.json={scalp[k]}"
            for k in SHARED_KEYS if k in cur and k in scalp and cur[k] != scalp[k]]


def _sync_framework(params: dict) -> bytes:
    """Write the shared parameters into the framework's scalp strategy. Returns the previous file bytes."""
    old = FW_CONFIG.read_bytes()
    fw = json.loads(old)
    for e in fw["strategies"]:
        if e["type"] == "scalp":
            for k in SHARED_KEYS:
                e["params"][k] = params[k]
    FW_CONFIG.write_text(json.dumps(fw, indent=2) + "\n")
    return old


def run_sealed_test(force: bool = False) -> dict:
    """One-time evaluation of the CURRENT parameters on the sealed final slice. The optimizer never sees
    this data. Refuses to run twice for the same parameters unless force=True (every extra look at the
    sealed slice weakens it)."""
    cfg, c = _cfg(), _costs()
    dfs = load_all(load_profile()["scanner"]["htf"])
    search, _, _, sealed, sealed_start = _split(dfs, cfg["train_frac"], cfg["sealed_frac"])
    cur = S.load()
    cp, cthr = S.params(cur), float(cur["ml_threshold"])
    fingerprint = json.dumps(cur, sort_keys=True, default=str)
    prev = read_json("sealed_result.json", None)
    if prev and prev.get("fingerprint") == fingerprint and not force:
        return {**prev, "already_run": True}
    m, _ = _evaluate(cp, cthr, search, sealed, sealed_start, c)
    ok, reasons = (False, ["ML model could not be fitted"]) if m is None else _gate(m, cfg)
    result = {"ts": now_utc().isoformat(), "fingerprint": fingerprint, "params": cur,
              "metrics": m, "passed": ok, "reasons": reasons or ["sealed test passed"],
              "already_run": False}
    write_json("sealed_result.json", result)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "sealed.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_autotune() -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg, c = _cfg(), _costs()
    dfs = load_all(load_profile()["scanner"]["htf"])
    search, hold, hold_start, _sealed, _sealed_start = _split(dfs, cfg["train_frac"], cfg["sealed_frac"])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(_objective(search, c, cfg), n_trials=cfg["n_trials"], show_progress_bar=False)
    if study.best_value <= FAILED / 2:
        raise ValueError("no trial produced enough trades; check data and search ranges")
    best = study.best_trial
    bp = Params(**{k: best.params[k] for k in S.KEYS})
    bthr = float(best.params["ml_threshold"])

    cur = S.load()
    cp, cthr = S.params(cur), float(cur["ml_threshold"])
    cur_m, _ = _evaluate(cp, cthr, search, hold, hold_start, c)
    new_m, _ = _evaluate(bp, bthr, search, hold, hold_start, c)

    # Every selection on this holdout window is counted. Past the limit it is no longer out-of-sample.
    key = _window_key(hold_start)
    uses = _ledger_bump(key)
    exhausted = uses > int(cfg["max_holdout_uses"])

    cur_ok = cur_m is not None and _gate(cur_m, cfg)[0]
    if new_m is None:
        new_ok, gate_reasons = False, ["ML model could not be fitted on search data"]
    else:
        new_ok, gate_reasons = _gate(new_m, cfg)
    better = new_ok and (not cur_ok or _better(new_m, cur_m, cfg))
    apply = better and not exhausted

    if not new_ok:
        reasons = gate_reasons
    elif exhausted:
        reasons = [f"holdout window already used {uses - 1} times (limit {cfg['max_holdout_uses']}); "
                   "it is no longer out-of-sample. Run `algolab sealed` before trusting a new candidate"]
    elif not better:
        reasons = ["candidate passed the holdout gate but does not beat current settings by "
                   f"{cfg['min_improvement_pct']}%"]
    else:
        reasons = ["candidate beats current settings on holdout"]

    result = {"ts": now_utc().isoformat(), "applied": False, "rolled_back": False,
              "reasons": reasons, "holdout_uses": uses,
              "current": {"params": cur, "holdout": cur_m},
              "candidate": {"params": {**asdict(bp), "ml_threshold": bthr}, "holdout": new_m},
              "trials": _trials(study), "trials_total": len(study.trials)}

    if apply:
        old_model = M.MODEL_FILE.read_bytes() if M.MODEL_FILE.exists() else None
        backup = S.write({**asdict(bp), "ml_threshold": bthr},
                         reason=f"autotune {now_utc().date().isoformat()}")
        old_fw = _sync_framework({**asdict(bp)})
        tr_all, feats_all = _trades(dfs, bp, c)
        if bthr > 0:
            X, y = M.dataset(tr_all, feats_all)
            M.save(M.fit(X, y))
        elif M.MODEL_FILE.exists():
            M.MODEL_FILE.unlink()

        formal = run_validation()
        if formal["passed"]:
            result["applied"] = True
            alerts.send(f"autotune applied: holdout expectancy {new_m['expectancy']} over "
                        f"{new_m['trades']} trades, threshold {bthr}. Backup {backup.name}",
                        "WARNING")
        else:
            S.restore(backup)
            FW_CONFIG.write_bytes(old_fw)
            if old_model is not None:
                M.MODEL_FILE.write_bytes(old_model)
            elif M.MODEL_FILE.exists():
                M.MODEL_FILE.unlink()
            result["rolled_back"] = True
            result["reasons"] = ["formal validation failed after update: "
                                 + "; ".join(formal["reasons"])]
            alerts.send("autotune rolled back: " + result["reasons"][0], "WARNING")

    REPORTS.mkdir(exist_ok=True)
    text = json.dumps(result, indent=2, default=str)
    (REPORTS / "autotune_latest.json").write_text(text)
    (REPORTS / f"autotune_{now_utc().strftime('%Y%m%dT%H%M%SZ')}.json").write_text(text)
    return result
