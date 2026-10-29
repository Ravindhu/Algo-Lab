"""Backtest with the SAME rule the live scanner uses: higher-timeframe engulfing confirmed by
Hilega-Milega RSI/EMA direction, ATR stop and target, spread and commission. Entry is at the next
bar's open. One position at a time.

Fill model (conservative):
  - A position entered at bar i's open is checked against bar i's own high and low (no skipped stops).
  - Spread is paid on entry AND exit (half the spread per side).
  - A gap through the stop fills at the bar's open (worse), a gap through the target fills at the open (better).
  - If stop and target are both inside one bar, the stop is assumed first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from algolab.profile import ROOT
from algolab.framework.strategies.engulfing_strategy import engulfing
from algolab.framework.strategies.indicators import hm_direction


@dataclass(frozen=True)
class Params:
    rsi_n: int = 9
    ema_n: int = 3
    atr_n: int = 14
    sl_mult: float = 1.5
    tp_mult: float = 3.0


@dataclass(frozen=True)
class Costs:
    initial_balance: float = 10000.0
    spread: float = 0.00012
    commission_per_lot: float = 3.5
    contract_size: float = 100000.0
    risk_per_trade: float = 0.01
    max_lots: float = 1.0


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close"]].dropna().sort_index()


def load_all(htf: str, data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    folder = Path(data_dir) if data_dir else ROOT / "data"
    out = {f.stem.rsplit("_", 1)[0]: load_csv(f) for f in sorted(folder.glob(f"*_{htf}.csv"))}
    if not out:
        raise FileNotFoundError(f"no {htf} CSV files in {folder}")
    return out


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Simple-average true range over n bars. Identical to hilega_milega.atr, which the live
    position manager uses, so a stop sized in backtest is sized the same way live."""
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def signals(df: pd.DataFrame, p: Params) -> np.ndarray:
    """Signal per bar: the rule evaluated on that bar's closed data (no look-ahead)."""
    eng = engulfing(df).to_numpy()
    conf = hm_direction(df["close"].to_numpy(), p.rsi_n, p.ema_n)
    sig = np.zeros(len(df), dtype=int)
    sig[(eng > 0) & (conf == 1)] = 1
    sig[(eng < 0) & (conf == -1)] = -1
    return sig


def run_backtest(df: pd.DataFrame, p: Params, c: Costs) -> pd.DataFrame:
    sig = signals(df, p)
    a = atr(df, p.atr_n).to_numpy()
    o, h, l, cl = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    times = df.index
    half = c.spread / 2
    eq = float(c.initial_balance)
    trades: list[dict] = []
    pos: dict | None = None

    def close(i: int, px: float, reason: str) -> None:
        nonlocal eq, pos
        gross = pos["dir"] * (px - pos["entry"]) * pos["lots"] * c.contract_size
        net = gross - 2 * c.commission_per_lot * pos["lots"]
        eq += net
        trades.append({**pos, "exit_time": times[i], "exit": px, "pnl": net, "reason": reason})
        pos = None

    for i in range(1, len(df)):
        # 1. Entry at this bar's open, decided on the previous closed bar.
        if pos is None and sig[i - 1] != 0 and a[i - 1] > 0:
            d = int(sig[i - 1])
            sl_d = p.sl_mult * a[i - 1]
            lots = min(math.floor(eq * c.risk_per_trade / (sl_d * c.contract_size) * 100) / 100,
                       c.max_lots)
            if lots > 0:
                entry = o[i] + d * half              # buy at ask, sell at bid
                pos = {"dir": d, "entry": entry, "lots": lots, "entry_time": times[i],
                       "signal_time": times[i - 1],
                       "sl": entry - d * sl_d, "tp": entry + d * p.tp_mult * a[i - 1]}
        # 2. Broker stops and targets inside this bar, including the bar the position was entered on.
        if pos is not None:
            d = pos["dir"]
            if d > 0:
                if l[i] <= pos["sl"]:
                    close(i, min(o[i], pos["sl"]) - half, "SL")      # exit at bid
                elif h[i] >= pos["tp"]:
                    close(i, max(o[i], pos["tp"]) - half, "TP")
            else:
                if h[i] >= pos["sl"]:
                    close(i, max(o[i], pos["sl"]) + half, "SL")      # exit at ask
                elif l[i] <= pos["tp"]:
                    close(i, min(o[i], pos["tp"]) + half, "TP")

    if pos is not None:
        close(len(df) - 1, float(cl[-1]) - pos["dir"] * half, "EOD")
    return pd.DataFrame(trades)


def trade_metrics(tr: pd.DataFrame, initial: float) -> dict:
    if tr.empty:
        return {"trades": 0, "net": 0.0, "win_rate": 0.0, "profit_factor": None,
                "expectancy": 0.0, "max_dd_pct": 0.0}
    pnl = tr["pnl"].astype(float)
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    curve = initial + pnl.cumsum()
    dd = float((curve / curve.cummax() - 1).min())
    return {"trades": int(len(pnl)), "net": round(float(pnl.sum()), 2),
            "win_rate": round(float((pnl > 0).mean()), 3),
            "profit_factor": round(float(gp / gl), 3) if gl > 0 else None,
            "expectancy": round(float(pnl.mean()), 2),
            "max_dd_pct": round(dd * 100, 2)}
