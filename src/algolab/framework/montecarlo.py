"""Monte Carlo check of a trade log, and a win-rate gate for entries.

simulate(): bootstrap the realised trade P&Ls (sampling with replacement) many times and look at the
spread of outcomes: final equity, maximum drawdown, probability of profit, probability of ruin.
Assumes trades are independent and that the future looks like the sample. Both are approximations.
A thin or unlucky sample gives a misleading picture, so read the trade count next to every number.

win_lower_bound(): one-sided Wilson lower bound on the win rate. The gate uses the LOWER bound, so a
signal is taken only when the evidence for its win rate is strong, not just when the point estimate is high.

Nothing here predicts certainty. There is no entry at probability one. The best achievable target is a
positive lower bound on expectancy after costs, checked out of sample.

CLI:  python -m algolab.framework.montecarlo trades.csv --balance 10000 --runs 10000
      (trades.csv needs a `pnl` column, e.g. the output of run_ocl_backtest)
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

Z_ONE_SIDED_95 = 1.6448536269514722


def simulate(pnls, start_balance: float, runs: int = 10_000, seed: int = 0,
             ruin_pct: float = 0.5, dd_limit_pct: float | None = None, chunk: int = 1000) -> dict:
    arr = np.asarray(pnls, dtype=float)
    if arr.size == 0:
        raise ValueError("no trades to simulate")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trade P&L contains NaN or infinity")
    if start_balance <= 0 or runs <= 0:
        raise ValueError("start_balance and runs must be positive")
    rng = np.random.default_rng(seed)
    finals, max_dds, ruined = [], [], []
    done = 0
    while done < runs:
        m = min(chunk, runs - done)
        paths = arr[rng.integers(0, arr.size, size=(m, arr.size))]
        eq = start_balance + np.cumsum(paths, axis=1)
        eq_full = np.concatenate([np.full((m, 1), start_balance), eq], axis=1)
        peak = np.maximum.accumulate(eq_full, axis=1)
        dd = np.max((peak - eq_full) / peak, axis=1)
        finals.append(eq_full[:, -1])
        max_dds.append(dd)
        ruined.append(eq_full.min(axis=1) <= start_balance * (1 - ruin_pct))
        done += m
    final = np.concatenate(finals)
    dd = np.concatenate(max_dds)
    ruin = np.concatenate(ruined)
    out = {
        "trades": int(arr.size),
        "runs": int(runs),
        "start_balance": float(start_balance),
        "mean_trade_pnl": float(arr.mean()),
        "p_profit": float(np.mean(final > start_balance)),
        "final_equity": {"p5": float(np.percentile(final, 5)), "median": float(np.median(final)),
                         "p95": float(np.percentile(final, 95))},
        "max_drawdown_pct": {"median": float(np.median(dd) * 100), "p95": float(np.percentile(dd, 95) * 100)},
        "p_ruin": {"threshold_pct": ruin_pct * 100, "probability": float(np.mean(ruin))},
    }
    if dd_limit_pct is not None:
        out["p_drawdown_over_limit"] = {"limit_pct": dd_limit_pct,
                                        "probability": float(np.mean(dd * 100 > dd_limit_pct))}
    return out


def win_lower_bound(wins: int, n: int, z: float = Z_ONE_SIDED_95) -> float:
    """One-sided Wilson lower bound on the win rate. Returns 0.0 when n is 0."""
    if n <= 0:
        return 0.0
    if not 0 <= wins <= n:
        raise ValueError("wins must be between 0 and n")
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def entry_allowed(recent_pnls, min_trades: int = 30, min_lower_bound: float = 0.5) -> tuple[bool, str]:
    """Gate a new entry on the lower bound of the recent win rate. Too few trades -> not allowed."""
    arr = np.asarray(recent_pnls, dtype=float)
    if arr.size < min_trades:
        return False, f"only {arr.size} recent trades (need {min_trades})"
    wins = int(np.sum(arr > 0))
    lb = win_lower_bound(wins, arr.size)
    if lb < min_lower_bound:
        return False, f"win-rate lower bound {lb:.3f} below {min_lower_bound:.3f}"
    return True, f"win-rate lower bound {lb:.3f} meets {min_lower_bound:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Monte Carlo of a trade log (bootstrap of realised P&L).")
    ap.add_argument("trades_csv")
    ap.add_argument("--balance", type=float, required=True)
    ap.add_argument("--runs", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ruin-pct", type=float, default=50.0, help="ruin = equity falls this % below start")
    ap.add_argument("--dd-limit-pct", type=float, default=None)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.trades_csv)
    if "pnl" not in df.columns:
        print("trades file has no 'pnl' column", file=sys.stderr)
        return 2
    rep = simulate(df["pnl"], args.balance, runs=args.runs, seed=args.seed,
                   ruin_pct=args.ruin_pct / 100, dd_limit_pct=args.dd_limit_pct)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
