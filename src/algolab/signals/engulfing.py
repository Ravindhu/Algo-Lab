"""Signal rules shared by every consumer (framework strategies, backtest, alert tools).

Rules (last CLOSED bar):
  BUY  : bullish engulfing AND RSI above EMA(RSI) (Hilega-Milega)
  SELL : bearish engulfing AND RSI below EMA(RSI) (Hilega-Milega)

Both the live scanner and the backtest use hm_direction_series(), so a bar's HM state is the same
whether it is evaluated live or replayed on history. A bar's state uses closes up to and including
that bar only.

Pure math only: no broker calls, no orders, no file writes (except the CLI helper below).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algolab.profile import ROOT, load_profile


def hm_direction_series(closes, rsi_period: int = 9, ema_period: int = 3) -> np.ndarray:
    """Per-bar Hilega-Milega direction: +1 when RSI > EMA(RSI), -1 when RSI < EMA(RSI), 0 when undefined
    or equal. Entry j uses closes[0..j] only, so the series is safe to read at any bar.

    RSI is the simple-average RSI of the last rsi_period price changes (the same convention as
    hilega_milega.hilega_milega_rsi). The EMA is seeded with the first RSI value (as hilega_milega.ema)."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    p, e = int(rsi_period), int(ema_period)
    out = np.zeros(n, dtype=int)
    if p <= 0 or e <= 0 or n <= p:
        return out
    d = np.diff(c)
    g = np.concatenate([[0.0], np.cumsum(np.clip(d, 0, None))])
    lo = np.concatenate([[0.0], np.cumsum(np.clip(-d, 0, None))])
    idx = np.arange(p, n)
    gains = g[idx] - g[idx - p]
    losses = lo[idx] - lo[idx - p]
    safe = np.where(losses > 0, losses, 1.0)
    rs = np.where(losses > 0, gains / safe, np.inf)
    rsi = np.where(gains + losses == 0, 50.0, 100.0 - 100.0 / (1.0 + rs))
    k = 2.0 / (e + 1.0)
    ema_val = None
    for j, r in enumerate(rsi):
        ema_val = float(r) if ema_val is None else k * float(r) + (1 - k) * ema_val
        if j + 1 >= e:
            i = p + j
            out[i] = 1 if r > ema_val else (-1 if r < ema_val else 0)
    return out


def hm_confirmed(closes: list[float], rsi_period: int = 9, ema_period: int = 3) -> int | None:
    """Hilega-Milega direction of the LAST element: +1, -1, or None when unknown or equal."""
    if len(closes) == 0:
        return None
    d = int(hm_direction_series(closes, rsi_period, ema_period)[-1])
    return d if d != 0 else None


DEFAULT_MIN_BODY_PCT = 0.5


def body_ratio(df: pd.DataFrame) -> pd.Series:
    """Candle body as a share of its full range (high - low). 0 where the range is 0."""
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    return (body / rng.where(rng > 0)).fillna(0.0)


def engulfing(df: pd.DataFrame, min_body_pct: float = DEFAULT_MIN_BODY_PCT) -> pd.Series:
    """+1 bullish engulfing, -1 bearish engulfing, 0 none (evaluated per bar).

    Both candles of the pattern (the engulfed one and the engulfing one) must have a body larger than
    min_body_pct of their range. The default is 0.5, i.e. more than 50 percent. Set it to 0 to ignore body size."""
    o, c = df["open"], df["close"]
    po, pc = o.shift(), c.shift()
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    if min_body_pct > 0:
        if not {"high", "low"} <= set(df.columns):
            raise ValueError("body filter needs high and low columns")
        ok = (body_ratio(df) > min_body_pct) & (body_ratio(df).shift() > min_body_pct)
        bull, bear = bull & ok, bear & ok
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=df.index)


def scan_frame(symbol: str, df: pd.DataFrame, cfg: dict) -> dict | None:
    """One symbol's candles -> a signal dict for the last CLOSED bar, or None.

    The last row of the frame is the bar still forming. It is dropped before anything is computed,
    so neither the engulfing pattern nor the HM state can read a price that is not final yet."""
    rsi_period = int(cfg.get("hm_rsi_period", 9))
    ema_period = int(cfg.get("hm_ema_period", 3))
    if len(df) < max(rsi_period + ema_period + 3, 5):
        return None
    closed = df.iloc[:-1].copy()           # drop the forming bar
    closed["eng"] = engulfing(closed)
    bar = closed.iloc[-1]                  # last closed bar
    conf = int(hm_direction_series(closed["close"].to_numpy(), rsi_period, ema_period)[-1])
    if bar["eng"] > 0 and conf == 1:
        side = "BUY"
    elif bar["eng"] < 0 and conf == -1:
        side = "SELL"
    else:
        return None
    return {"symbol": symbol, "side": side, "time": str(closed.index[-1]),
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
