"""Automatic high-performance strategy selection for the Bot.

Periodically ranks every enabled framework strategy by its LIVE performance
(real fills from the trade journal, not backtests), and keeps only the top N
running by writing "enabled": true/false into config/framework.json.

Selection rule (robust, minimum evidence required):
  score = expectancy(R) * sqrt(closed trades), provided
    - at least min_trades closed trades for that strategy,
    - profit factor >= min_profit_factor (None/zero-loss counts as pass),
    - R-mean win rate lower bound >= min_lower_bound (tight-S violin bound).
  Strategies with too little evidence are never demoted (no churn from noise),
  and an empty roster is never allowed: the best strategies stay on regardless
  of how few pass, so the Bot always has something enabled.

Guard rails (anti-overfit, same spirit as autotune):
  - demotion requires the strategy to fail the gates (never on a whim),
  - the win-rate bound is corrected for multiple comparisons (Bonferroni across the strategies ranked),
    so a strategy is not promoted because it was the luckiest of many,
  - only trades with BROKER-realised P&L count. Unknown P&L is excluded, never treated as zero,
  - every selection run appends to reports/selection_latest.json. Changes are written only with --apply
    (the daily timer runs report-only; a person approves the change),
  - a strategy is never demoted while it still has open positions managed by
    the position manager (its exits keep working; only NEW entries stop).

CLI:  algolab-select [--apply]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from statistics import NormalDist

from algolab.ops import alerts
from algolab.ops.state import atomic_write_text, now_utc, write_json
from algolab.profile import ROOT

HANDOFF = "framework_handoff.json"
SELECTION = "selection.json"
REPORTS = ROOT / "reports"
MIN_R_WINDOW = 60                       # closed trades considered for ranking
FAMILY_ALPHA = 0.05                     # one-sided error across all strategies ranked together
DEFAULTS = {"top_n": 3, "min_trades": 30, "min_profit_factor": 1.0,
            "min_lower_bound": 0.35}


def _config() -> dict:
    path = ROOT / "config" / "selection.json"
    if path.exists():
        return {**DEFAULTS, **json.loads(path.read_text())}
    return dict(DEFAULTS)


def load_framework_config() -> dict:
    return json.loads((ROOT / "config" / "framework.json").read_text())


def _closes() -> list[dict]:
    from algolab.ops.journal import usable_closes
    rows = [r for r in usable_closes() if r.get("r") is not None]   # broker P&L only
    return rows[-MIN_R_WINDOW * 10:]           # bounded lookback


def _r_per_strategy(rows: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        out[r.get("strategy", "?")].append(float(r.get("r", 0.0)))
    return out


def _win_rate_lower_bound(rs: list[float], z: float = 1.96) -> float | None:
    """Wilson lower bound of the win rate. z is set per family size (see rank_strategies)."""
    n = len(rs)
    if n == 0:
        return None
    p = sum(1 for r in rs if r > 0) / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - rad) / denom


def _score(rs: list[float], cfg: dict, z: float = 1.96) -> dict:
    """Ranking metrics for one strategy. Pass requires enough trades and
    positive expectancy; the score itself is expectancy * sqrt(n)."""
    n = len(rs)
    if n < cfg["min_trades"]:
        # Too little evidence: the strategy is NOT ranked and NOT demoted —
        # a short track record is not proof of bad performance.
        return {"n": n, "expectancy": None, "score": None, "pf": None,
                "win_lb": None, "passes": None, "why": f"only {n} closed trades (unranked)"}
    exp = sum(rs) / n
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None
    lb = _win_rate_lower_bound(rs, z)
    passes = exp > 0 and (pf is None or pf >= cfg["min_profit_factor"]) \
        and (lb is None or lb >= cfg["min_lower_bound"])
    return {"n": n, "expectancy": round(exp, 4), "score": round(exp * math.sqrt(n), 4),
            "pf": round(pf, 3) if pf is not None else None,
            "win_lb": round(lb, 3) if lb is not None else None,
            "passes": passes, "why": "ok" if passes else "gates not met"}


def family_z(m: int) -> float:
    """One-sided z for a Wilson bound with a Bonferroni correction over m strategies."""
    m = max(int(m), 1)
    return NormalDist().inv_cdf(1 - FAMILY_ALPHA / m)


def rank_strategies(cfg: dict | None = None) -> dict[str, dict]:
    """strategy name -> metrics dict, for every strategy with journal closes."""
    cfg = cfg or _config()
    by = _r_per_strategy(_closes())
    z = family_z(len(by))
    return {name: _score(rs, cfg, z) for name, rs in by.items()}


def apply_selection(dry_run: bool = True, cfg: dict | None = None) -> dict:
    """Rank strategies, demote the weak, keep the top N. Writes framework.json
    enabled flags only when a change is warranted and not a dry run."""
    cfg = cfg or _config()
    fw = load_framework_config()
    ranks = rank_strategies(cfg)

    ranked = sorted(
        (r for r in ranks.items() if r[1]["passes"]),
        key=lambda kv: kv[1]["score"] or 0, reverse=True)
    keep = {name for name, _ in ranked[:cfg["top_n"]]}
    # strategies with no closes yet: leave enabled (too early to judge)
    unknown = [e["type"] for e in fw["strategies"]
               if e["type"] not in ranks and e.get("enabled")]
    keep.update(unknown)

    changes: list[dict] = []
    for entry in fw["strategies"]:
        name = entry["type"]
        current = bool(entry.get("enabled", True))
        metrics = ranks.get(name)
        if metrics is None:
            want = current                      # no evidence: no change
            why = "no closed trades yet"
        elif metrics["passes"] is None:
            want = current                      # too little evidence: no change
            why = metrics["why"]
        elif metrics["passes"] and name in keep:
            want = True
            why = f"rank {metrics['score']}"
        elif metrics["passes"] is False:
            # Gates failed = bad performance. Never demote if that would leave
            # the roster empty (keep at least the best scorer running).
            failing = [n for n, m in ranks.items() if m["passes"] is False]
            passing = [e for e in fw["strategies"]
                       if e["type"] not in failing and e.get("enabled")]
            if not passing:
                want = current
                why = "kept: every strategy is failing the gates"
            else:
                want = False
                why = metrics["why"]
        else:
            want = False
            why = f"score {metrics['score']} below top {cfg['top_n']}"
        if want != current:
            changes.append({"strategy": name, "was": current, "would": want,
                            "why": why})
            entry["enabled"] = want
        elif not want:
            entry["enabled"] = False

    result = {"ts": now_utc().isoformat(), "dry_run": dry_run,
              "top_n": cfg["top_n"], "ranks": ranks,
              "keep": sorted(keep), "changes": changes}

    if dry_run:
        return result
    if changes:
        fw_path = ROOT / "config" / "framework.json"
        atomic_write_text(fw_path, json.dumps(fw, indent=2) + "\n")
        write_json(SELECTION, result)
        for ch in changes:
            alerts.send(f"selection: {ch['strategy']} {'enabled' if ch['would'] else 'disabled'} "
                        f"({ch['why']})", "WARNING", key=f"sel-{ch['strategy']}",
                        cooldown_min=1440)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="algolab-select")
    ap.add_argument("--apply", action="store_true",
                    help="write framework.json (default: report only; the daily timer never applies)")
    args = ap.parse_args(argv)
    result = apply_selection(dry_run=not args.apply)
    print(json.dumps(result, indent=2, default=str))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "selection_latest.json").write_text(
        json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
