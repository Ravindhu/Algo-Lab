"""Live position manager for OCL trades.

Broker side: each entry is a market order with a disaster stop (broker_sl) and the fixed target (tp, if any).
Strategy side: the LTF stop, the ATR trail, breakeven and the HTF floor are judged on CLOSED bars, as in
ocl_keylevel.py. This manager closes at market when one of those rules fires.

Each cycle, per tracked position:
  1. Broker no longer holds it: classify the closure from the bars since the last cycle
     (DISASTER / TP / BROKER_CLOSE). No order is sent.
  2. Otherwise run protect.Trade.on_closed_bar() on each new closed LTF bar. SL or TRAIL closes at market.
  3. The last closed HTF bar closed below the HTF floor: close at market (HTF).

State is only written after a broker call succeeds. If a close fails, the next cycle re-evaluates the same bar.
"""
from __future__ import annotations

import logging

import pandas as pd

from algolab.framework.protect import Protection, Trade
from algolab.framework.timeframe import TimeFrame
from algolab.hilega_milega import atr as atr_series
from algolab.live.adapter import AdapterError

log = logging.getLogger("algolab.positions")


def new_record(rec: dict, *, strategy: str, ltf: str, htf: str) -> dict:
    """Build the tracked record from a Bot result that really filled."""
    fill = float(rec["fill_price"])
    tp = float(rec["tp"]) if float(rec.get("tp") or 0) > 0 else None
    protect = Protection.from_dict(rec["protect"])
    trade = Trade.open(entry=fill, trigger=float(rec["trigger"]), htf_sl=float(rec["htf_sl"]),
                       broker_sl=float(rec["sl"]), tp=tp, p=protect)
    delta = TimeFrame(ltf).seconds
    return {
        "ticket": int(rec["ticket"]), "strategy": strategy, "symbol": rec["symbol"],
        "ltf": ltf, "htf": htf, "volume": float(rec["lots"]),
        "protect": protect.to_dict(), "trade": trade.to_dict(),
        "atr_period": int(rec.get("atr_period", 14)),
        # The entry bar is the one that was forming when the signal was made: rec["bar"] + one bar.
        "last_bar": str(pd.Timestamp(rec["bar"]) + pd.Timedelta(seconds=delta)),
        "delta_s": delta,
    }


def _event(rec: dict, reason: str, price: float) -> dict:
    return {"strategy": rec["strategy"], "symbol": rec["symbol"], "reason": reason,
            "price": float(price), "ticket": rec["ticket"]}


def _classify(trade: Trade, new_bars: pd.DataFrame) -> tuple[str, float]:
    for _, row in new_bars.iterrows():
        if float(row["low"]) <= trade.broker_sl:
            return "DISASTER", trade.broker_sl
        if trade.tp is not None and float(row["high"]) >= trade.tp:
            return "TP", trade.tp
    last = float(new_bars["close"].iloc[-1]) if len(new_bars) else trade.entry
    return "BROKER_CLOSE", last


async def _manage_one(ad, rec: dict, live_pos, magic: int, deviation: int, lookback: int) -> dict | None:
    sym, ltf, htf = rec["symbol"], rec["ltf"], rec["htf"]
    protect = Protection.from_dict(rec["protect"])
    trade = Trade.from_dict(rec["trade"])
    delta = pd.Timedelta(seconds=rec["delta_s"])

    bars = await ad.candles(sym, ltf, lookback)
    closed = bars.iloc[:-1]
    highs = [float(x) for x in closed["high"]]
    lows = [float(x) for x in closed["low"]]
    closes = [float(x) for x in closed["close"]]
    atr_vals = atr_series(highs, lows, closes, period=rec.get("atr_period", 14))
    start = pd.Timestamp(rec["last_bar"])
    positions = [i for i, t in enumerate(closed.index) if t >= start]
    new_bars = closed.iloc[positions]

    if live_pos is None:
        reason, price = _classify(trade, new_bars)
        return _event(rec, reason, price)

    last_bar = rec["last_bar"]
    for pos in positions:
        idx = closed.index[pos]
        high, close = float(closed["high"].iloc[pos]), float(closed["close"].iloc[pos])
        reason = trade.on_closed_bar(high=high, close=close, atr=atr_vals[pos], p=protect)
        last_bar = str(idx + delta)
        if reason:
            ad.close_position(rec["ticket"], sym, trade.side, rec["volume"], magic, deviation)
            rec["last_bar"], rec["trade"] = last_bar, trade.to_dict()
            return _event(rec, reason, close)

    hbars = await ad.candles(sym, htf, 10)
    hclosed = hbars.iloc[:-1]
    if len(hclosed) and float(hclosed["close"].iloc[-1]) < trade.htf_sl:
        hclose = float(hclosed["close"].iloc[-1])
        ad.close_position(rec["ticket"], sym, trade.side, rec["volume"], magic, deviation)
        rec["last_bar"], rec["trade"] = last_bar, trade.to_dict()
        return _event(rec, "HTF", hclose)

    rec["last_bar"], rec["trade"] = last_bar, trade.to_dict()
    return None


async def manage(ad, tracked: dict, *, magic: int, deviation: int, lookback: int = 120) -> list[dict]:
    """Mutates `tracked` in place. Returns closure events for the strategies to act on.
    A failed candle or close call keeps the position tracked and is retried next cycle."""
    live = {int(p["ticket"]): p for p in ad.positions_by_magic(magic)}
    events: list[dict] = []
    for key in list(tracked):
        rec = tracked[key]
        try:
            ev = await _manage_one(ad, rec, live.get(int(rec["ticket"])), magic, deviation, lookback)
        except AdapterError as e:
            log.error("position %s: %s (kept tracking; retried next cycle)", key, e)
            continue
        if ev is not None:
            events.append(ev)
            del tracked[key]
    return events


def flatten(ad, tracked: dict, *, magic: int, deviation: int, reason: str) -> list[dict]:
    """Close every tracked position now (prop-firm rule). Positions the broker already closed are
    recorded as BROKER_CLOSE and dropped."""
    live = {int(p["ticket"]) for p in ad.positions_by_magic(magic)}
    events: list[dict] = []
    for key in list(tracked):
        rec = tracked[key]
        if int(rec["ticket"]) in live:
            try:
                res = ad.close_position(rec["ticket"], rec["symbol"], 1, rec["volume"], magic, deviation)
            except AdapterError as e:
                log.error("flatten %s failed: %s (kept tracking)", key, e)
                continue
            events.append(_event(rec, reason, float(res.get("fill_price", 0.0))))
        else:
            events.append(_event(rec, "BROKER_CLOSE", float(rec["trade"]["entry"])))
        del tracked[key]
    return events
