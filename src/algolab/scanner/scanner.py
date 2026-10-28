"""Scan all forex pairs for higher-timeframe engulfing candles.

Rules (last CLOSED bar):
  BUY  : bullish engulfing AND RSI below the oversold level
  SELL : bearish engulfing AND RSI above the overbought level
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algolab.profile import ROOT, load_profile


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    po, pc = o.shift(), c.shift()
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=df.index)


def scan_frame(symbol: str, df: pd.DataFrame, cfg: dict) -> dict | None:
    if len(df) < cfg["rsi_n"] + 3:
        return None
    df = df.copy()
    df["rsi"] = rsi(df["close"], cfg["rsi_n"])
    df["eng"] = engulfing(df)
    bar = df.iloc[-2]                      # last closed bar, not the forming one
    if bar["eng"] > 0 and bar["rsi"] < cfg["oversold"]:
        side = "BUY"
    elif bar["eng"] < 0 and bar["rsi"] > cfg["overbought"]:
        side = "SELL"
    else:
        return None
    return {"symbol": symbol, "side": side, "time": str(df.index[-2]),
            "rsi": round(float(bar["rsi"]), 2), "close": float(bar["close"])}


def scan_all(htf: str | None = None, data_dir: str | Path | None = None) -> list[dict]:
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
