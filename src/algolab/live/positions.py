"""Live position manager for OCL trades.

Broker side: each entry is a market order with a disaster stop (broker_sl) and the fixed target (tp, if any).
Strategy side: the LTF stop, the ATR trail, breakeven and the HTF floor are judged on CLOSED bars, as in
ocl_keylevel.py. This manager closes at market when one of those rules fires.

Each cycle, per tracked position:
  1. Broker no longer holds it: classify the closure from the bars since the last cycle
     (DISASTER / TP / BROKER_CLOSE). No order is sent. P&L is the broker's realised profit.
  2. Otherwise run protect.Trade.on_closed_bar() on each new closed LTF bar. SL or TRAIL closes at market.
  3. The last closed HTF bar closed below the HTF floor: close at market (HTF).

P&L is never estimated: it is the broker's realised profit from the deal history, or None (unknown).
A partial close keeps the position tracked with the remaining volume and retries next cycle.
reconcile() reports broker positions with this project's magic number that are not tracked (orphans).
State is only written after a broker call succeeds. If a close fails, the next cycle re-evaluates the same bar.
"""
from __future__ import annotations

import logging

import pandas as pd

from algolab.framework.protect import Protection, Trade
from algolab.framework.symbol import ForexSymbol
from algolab.framework.timeframe import TimeFrame
from algolab.hilega_milega import atr as atr_series
from algolab.live.adapter import AdapterError, PartialClose

log = logging.getLogger("algolab.positions")


def _mirror_of(side: str) -> int:
    """+1 for BUY, -1 for SELL. A SELL trade is tracked in mirrored prices (see the module note)."""
    return -1 if side == "SELL" else 1


def _broker_side(rec: dict) -> int:
    """The side the broker sees, used to close: +1 for a BUY position, -1 for a SELL position."""
    return _mirror_of(rec.get("side", "BUY"))


def new_record(rec: dict, *, strategy: str, ltf: str, htf: str) -> dict:
    """Build the tracked record from a Bot result that really filled.

    A SELL trade is stored with every price negated. Protect.Trade then runs its BUY rules unchanged, and the
    manager feeds it negated bars, so the same code protects both sides."""
    side = rec.get("side", "BUY")
    m = _mirror_of(side)
    fill = float(rec["fill_price"])
    tp = float(rec["tp"]) if float(rec.get("tp") or 0) > 0 else None
    protect = Protection.from_dict(rec["protect"])
    trade = Trade.open(entry=m * fill, trigger=m * float(rec["trigger"]), htf_sl=m * float(rec["htf_sl"]),
                       broker_sl=m * float(rec["sl"]), tp=None if tp is None else m * tp, p=protect,
                       pip=ForexSymbol(rec["symbol"]).pip)
    delta = TimeFrame(ltf).seconds
    return {
        "ticket": int(rec["ticket"]), "strategy": strategy, "symbol": rec["symbol"], "side": side,
        "ltf": ltf, "htf": htf, "volume": float(rec["lots"]),
        "protect": protect.to_dict(), "trade": trade.to_dict(),
        "atr_period": int(rec.get("atr_period", 14)),
        # The entry bar is the one that was forming when the signal was made: rec["bar"] + one bar.
        "last_bar": str(pd.Timestamp(rec["bar"]) + pd.Timedelta(seconds=delta)),
        "delta_s": delta,
    }


def new_plain_record(rec: dict, *, strategy: str, tf: str, hold_bars: int | None = None) -> dict:
    """Tracking record for a plain (non-OCL) strategy. Exits are the broker's stop and target, plus an optional
    time exit (hold_bars). No trail, no breakeven, no HTF floor. The manager's job is to classify the close and
    record P&L, so the trade reaches the journal and the entry gate, and it stops blocking new entries as an orphan.

    hold_bars follows the discovery backtest exactly: the signal is on closed bar i, the entry is at the open of
    bar i+1, and the trade closes at the close of bar i+1+hold_bars (run_bt_loop, `last = start + max_hold`).
    exit_after is the open time of that last bar; the close happens once that bar has closed."""
    tp = float(rec["tp"]) if float(rec.get("tp") or 0) > 0 else None
    delta = TimeFrame(tf).seconds
    signal_bar = pd.Timestamp(rec["bar"])
    exit_after = None
    if hold_bars is not None:
        exit_after = str(signal_bar + pd.Timedelta(seconds=delta * (1 + int(hold_bars))))
    return {
        "kind": "plain", "ticket": int(rec["ticket"]), "strategy": strategy, "symbol": rec["symbol"],
        "side": rec.get("side", "BUY"), "ltf": tf, "volume": float(rec["lots"]),
        "sl": float(rec["sl"]), "tp": tp, "entry": float(rec["fill_price"]),
        "last_bar": str(signal_bar + pd.Timedelta(seconds=delta)), "delta_s": delta,
        "exit_after": exit_after,
    }


def _event(rec: dict, reason: str, price: float, pnl: float | None) -> dict:
    return {"strategy": rec["strategy"], "symbol": rec["symbol"], "reason": reason,
            "price": float(price), "ticket": rec["ticket"], "pnl": pnl}


def _realized(ad, ticket: int) -> float | None:
    """Broker realised P&L of a closed position, or None when it cannot be read (recorded as unknown)."""
    try:
        return ad.realized_pnl(int(ticket))
    except AdapterError as e:
        log.error("realised P&L for ticket %s unavailable: %s", ticket, e)
        return None


def _classify(trade: Trade, lows: list[float], highs: list[float], closes: list[float]) -> tuple[str, float]:
    """Closure reason from the bars since the last cycle. Inputs and the returned price are in trade space
    (mirrored for SELL, so the caller converts the price back)."""
    for lo, hi in zip(lows, highs):
        if lo <= trade.broker_sl:
            return "DISASTER", trade.broker_sl
        if trade.tp is not None and hi >= trade.tp:
            return "TP", trade.tp
    last = closes[-1] if closes else trade.entry
    return "BROKER_CLOSE", last


class _StillOpen(Exception):
    """A partial close left the position open. The record keeps its old trade state and is re-evaluated."""


def _close_at_market(ad, rec: dict, side: int, magic: int, deviation: int) -> tuple[float, float | None]:
    """Close the whole position. Returns (fill_price, realised_pnl). Raises _StillOpen after a partial close,
    with rec["volume"] reduced to what is still open."""
    try:
        res = ad.close_position(rec["ticket"], rec["symbol"], side, rec["volume"], magic, deviation)
    except PartialClose as e:
        rec["volume"] = float(e.remaining)
        log.error("ticket %s partially closed; %s lots still open, retrying next cycle", rec["ticket"], e.remaining)
        raise _StillOpen() from e
    return float(res["fill_price"]), _realized(ad, rec["ticket"])


async def _manage_plain(ad, rec: dict, live_pos, lookback: int, magic: int, deviation: int) -> dict | None:
    """Plain record: the broker holds the stop and target. While it is open, advance the bar marker and apply the
    time exit (exit_after) once its bar has closed. Once the position is gone, classify the close from the bars
    since the last cycle and book the realised P&L."""
    bars = await ad.candles(rec["symbol"], rec["ltf"], lookback)
    closed = bars.iloc[:-1]
    delta = pd.Timedelta(seconds=rec["delta_s"])
    start = pd.Timestamp(rec["last_bar"])
    new = [i for i, t in enumerate(closed.index) if t >= start]
    last_bar = str(closed.index[-1] + delta) if len(closed) else rec["last_bar"]
    if live_pos is not None:
        rec["last_bar"] = last_bar
        if rec.get("exit_after") and len(closed) and closed.index[-1] >= pd.Timestamp(rec["exit_after"]):
            try:
                fill, pnl = _close_at_market(ad, rec, _broker_side(rec), magic, deviation)
            except _StillOpen:
                return None                    # partial close: retried next cycle
            return _event(rec, "TIME", fill, pnl)
        return None
    buy = rec["side"] == "BUY"
    sl, tp = rec["sl"], rec["tp"]
    reason, price = "BROKER_CLOSE", (float(closed["close"].iloc[-1]) if len(closed) else rec["entry"])
    for i in new:
        lo, hi = float(closed["low"].iloc[i]), float(closed["high"].iloc[i])
        if buy:
            stop, target = lo <= sl, tp is not None and hi >= tp
        else:
            stop, target = hi >= sl, tp is not None and lo <= tp
        if stop:
            reason, price = "DISASTER", sl
            break
        if target:
            reason, price = "TP", tp
            break
    return _event(rec, reason, price, _realized(ad, rec["ticket"]))


async def _manage_one(ad, rec: dict, live_pos, magic: int, deviation: int, lookback: int) -> dict | None:
    if rec.get("kind") == "plain":
        return await _manage_plain(ad, rec, live_pos, lookback, magic, deviation)
    sym, ltf, htf = rec["symbol"], rec["ltf"], rec["htf"]
    protect = Protection.from_dict(rec["protect"])
    trade = Trade.from_dict(rec["trade"])
    delta = pd.Timedelta(seconds=rec["delta_s"])
    m = _mirror_of(rec.get("side", "BUY"))        # trade space: prices negated for SELL
    side = _broker_side(rec)

    bars = await ad.candles(sym, ltf, lookback)
    closed = bars.iloc[:-1]
    if m > 0:
        highs = [float(x) for x in closed["high"]]
        lows = [float(x) for x in closed["low"]]
        closes = [float(x) for x in closed["close"]]
    else:                                         # SELL: the mirrored chart swaps high and low
        highs = [-float(x) for x in closed["low"]]
        lows = [-float(x) for x in closed["high"]]
        closes = [-float(x) for x in closed["close"]]
    atr_vals = atr_series(highs, lows, closes, period=rec.get("atr_period", 14))
    start = pd.Timestamp(rec["last_bar"])
    positions = [i for i, t in enumerate(closed.index) if t >= start]

    if live_pos is None:
        new = positions
        reason, price = _classify(trade, [lows[i] for i in new], [highs[i] for i in new],
                                  [closes[i] for i in new])
        return _event(rec, reason, m * price, _realized(ad, rec["ticket"]))

    last_bar = rec["last_bar"]
    for pos in positions:
        idx = closed.index[pos]
        reason = trade.on_closed_bar(high=highs[pos], close=closes[pos], atr=atr_vals[pos], p=protect)
        last_bar = str(idx + delta)
        if reason:
            try:
                fill, pnl = _close_at_market(ad, rec, side, magic, deviation)
            except _StillOpen:
                return None                  # nothing saved: the same bar is re-evaluated next cycle
            rec["last_bar"], rec["trade"] = last_bar, trade.to_dict()
            return _event(rec, reason, fill, pnl)

    hbars = await ad.candles(sym, htf, 10)
    hclosed = hbars.iloc[:-1]
    if len(hclosed) and m * float(hclosed["close"].iloc[-1]) < trade.htf_sl:
        try:
            fill, pnl = _close_at_market(ad, rec, side, magic, deviation)
        except _StillOpen:
            return None
        rec["last_bar"], rec["trade"] = last_bar, trade.to_dict()
        return _event(rec, "HTF", fill, pnl)

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


def reconcile(ad, tracked: dict, *, magic: int) -> list[int]:
    """Tickets that the broker holds for this project's magic number but that are NOT tracked.
    Such a position has no trailing stop or HTF exit. The caller must alert and block new entries."""
    tracked_tickets = {int(r["ticket"]) for r in tracked.values()}
    return sorted(ad.position_tickets(magic) - tracked_tickets)


def flatten(ad, tracked: dict, *, magic: int, deviation: int, reason: str) -> list[dict]:
    """Close every tracked position now (prop-firm rule). Positions the broker already closed are
    recorded as BROKER_CLOSE and dropped. A position that could not be closed stays tracked."""
    live = {int(p["ticket"]): p for p in ad.positions_by_magic(magic)}
    events: list[dict] = []
    for key in list(tracked):
        rec = tracked[key]
        if int(rec["ticket"]) in live:
            side = int(live[int(rec["ticket"])]["side"])
            try:
                fill, pnl = _close_at_market(ad, rec, side, magic, deviation)
            except (AdapterError, _StillOpen) as e:
                log.error("flatten %s failed: %s (kept tracking)", key, e)
                continue
            events.append(_event(rec, reason, fill, pnl))
        else:
            entry = float(rec["entry"]) if "entry" in rec else float(rec["trade"]["entry"])
            events.append(_event(rec, "BROKER_CLOSE", entry, _realized(ad, rec["ticket"])))
        del tracked[key]
    return events
