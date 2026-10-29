"""Signal rules shared by every consumer (framework strategies, backtest, alert tools).

Rules (last CLOSED bar):
  BUY  : bullish engulfing AND RSI above EMA(RSI) (Hilega-Milega)
  SELL : bearish engulfing AND RSI below EMA(RSI) (Hilega-Milega)

Pure math only: no broker calls, no orders, no file writes (except the CLI helper below).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algolab.hilega_milega import ema, hilega_milega_rsi
from algolab.profile import ROOT, load_profile


def hm_confirmed(closes: list[float], rsi_period: int = 9, ema_period: int = 3) -> int | None:
    """Hilega-Milega direction: +1 when RSI is above EMA(RSI), -1 when below, None when unknown."""
    rsi_values = hilega_milega_rsi(closes, rsi_period=rsi_period)
    line = ema(rsi_values, ema_period)
    r, e = rsi_values[-1] if rsi_values else None, line[-1] if line else None
    if r is None or e is None:
        return None
    return 1 if r > e else (-1 if r < e else None)


def engulfing(df: pd.DataFrame) -> pd.Series:
    """+1 bullish engulfing, -1 bearish engulfing, 0 none (evaluated per bar)."""
    o, c = df["open"], df["close"]
    po, pc = o.shift(), c.shift()
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=df.index)


def scan_frame(symbol: str, df: pd.DataFrame, cfg: dict) -> dict | None:
    """One symbol's candles -> a signal dict for the last closed bar, or None."""
    rsi_period = int(cfg.get("hm_rsi_period", 9))
    ema_period = int(cfg.get("hm_ema_period", 3))
    if len(df) < max(rsi_period + ema_period + 3, 5):
        return None
    df = df.copy()
    df["eng"] = engulfing(df)
    bar = df.iloc[-2]                      # last closed bar, not the forming one
    conf = hm_confirmed([float(x) for x in df["close"]], rsi_period, ema_period)
    if bar["eng"] > 0 and conf == 1:
        side = "BUY"
    elif bar["eng"] < 0 and conf == -1:
        side = "SELL"
    else:
        return None
    return {"symbol": symbol, "side": side, "time": str(df.index[-2]),
            "close": float(bar["close"])}


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
