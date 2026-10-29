"""scanner.py - the engulfing + Hilega-Milega HTF scanner.

Finds HTF engulfing candles confirmed by the Hilega-Milega RSI/EMA rule on the
last CLOSED bar, and arms them at the fib entry level (the OCL touch model).
Moved here from signals/engulfing.py; signals/engulfing.py re-exports these
names for back-compat.

Pure rules: reads CSVs/profile only, never touches a broker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from algolab.framework.strategies.engulfing_strategy import (DEFAULT_MIN_BODY_PCT,
                                                             FIB_ENTRY, engulfing,
                                                             fib_entry_level)
from algolab.framework.strategies.indicators import hm_direction
from algolab.profile import ROOT, load_profile


def scan_frame(symbol: str, df: pd.DataFrame, cfg: dict) -> dict | None:
    """One symbol's candles -> a signal dict for the last CLOSED bar, or None.

    The last row of the frame is the bar still forming. It is dropped before anything is computed,
    so neither the engulfing pattern nor the HM state can read a price that is not final yet."""
    rsi_period = int(cfg.get("hm_rsi_period", 9))
    ema_period = int(cfg.get("hm_ema_period", 3))
    if len(df) < max(rsi_period + ema_period + 3, 5):
        return None
    closed = df.iloc[:-1].copy()           # drop the forming bar
    closed["eng"] = engulfing(closed, float(cfg.get("min_body_pct", DEFAULT_MIN_BODY_PCT)))
    bar = closed.iloc[-1]                  # last closed bar
    conf = int(hm_direction(closed["close"].to_numpy(), rsi_period, ema_period)[-1])
    if bar["eng"] > 0 and conf == 1:
        side = "BUY"
    elif bar["eng"] < 0 and conf == -1:
        side = "SELL"
    else:
        return None
    fib = float(cfg.get("fib_pct", FIB_ENTRY))
    direction = 1 if side == "BUY" else -1
    level = fib_entry_level(float(bar["high"]), float(bar["low"]), direction, fib)
    floor = float(bar["low"]) if side == "BUY" else float(bar["high"])
    return {"symbol": symbol, "side": side, "time": str(closed.index[-1]),
            "close": float(bar["close"]),
            # The setup is armed at the fib level. The trade waits for the lower-timeframe touch (OCL model).
            "status": "ARMED", "entry_level": level, "floor": floor, "fib_pct": fib}


def scan_all(htf: str | None = None, data_dir: str | Path | None = None) -> list[dict]:
    """Offline helper: scan exported CSV files (alert tools, agent pipeline, MCP server)."""
    p = load_profile()
    cfg = p["scanner"]
    htf = htf or cfg["htf"]
    folder = Path(data_dir) if data_dir else ROOT / p["data_dir"]
    hits = []
    for f in sorted(folder.glob(f"*_{htf}.csv")):
        symbol = f.stem.rsplit("_", 1)[0]
        df = pd.read_csv(f, parse_dates=["time"], index_col="time")
        df.columns = [c.lower() for c in df.columns]
        hit = scan_frame(symbol, df[["open", "high", "low", "close"]].dropna().sort_index(), cfg)
        if hit:
            hits.append(hit)
    return hits


if __name__ == "__main__":
    print(json.dumps(scan_all(sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
