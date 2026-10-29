"""OCL backtest. It drives the SAME strategy object, protect.py and prop.py that the live bot uses, so the
validation gate tests the rules that trade. Bar ordering mirrors the live cycle:

For each LTF bar i (decision at its open), in this order:
  1. Broker exits during bar i-1 (disaster stop first, then target). Intrabar, from bar i-1's high/low.
  2. Close-based rules for bar i-1 (LTF stop, trail). They execute at the open of bar i (the next poll).
  3. HTF floor: the last closed HTF bar closed below it -> close at the open of bar i.
  4. Prop rules: mark equity, force a flatten if required (FAILED or weekend), then check the entry.
  5. Entry: strategy signal at the open of bar i, filled at open + half spread.

Costs: spread (half each side) and commission per lot per side. Account P&L is in account currency.
quote_to_account: 1.0 when the pair's quote currency is the account currency. Otherwise a float or a
Series aligned to the LTF index (account currency per quote unit), e.g. 1/USDJPY for JPY quotes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from algolab.backtest.engine import Costs
from algolab.framework.strategies.ocl import OclKeylevel, SetupState
from algolab.framework.prop import PropGuard
from algolab.framework.protect import Trade
from algolab.framework.timeframe import TimeFrame
from algolab.hilega_milega import atr as atr_series


def _with_forming(df: pd.DataFrame, pos: int, lookback: int) -> pd.DataFrame:
    """Rows up to pos inclusive, where row pos is the bar still forming. Only its OPEN is kept, so the
    frame carries no data from inside the forming bar. The strategy drops that last row."""
    view = df.iloc[max(0, pos - lookback): pos + 1].copy()
    o = float(df["open"].iloc[pos])
    for col in ("high", "low", "close", "open"):
        view.iloc[-1, view.columns.get_loc(col)] = o
    return view


def run_ocl_backtest(strategy: OclKeylevel, htf: pd.DataFrame, ltf: pd.DataFrame, costs: Costs,
                     quote_to_account: float | pd.Series = 1.0,
                     prop: PropGuard | None = None) -> pd.DataFrame:
    strategy.st = SetupState()
    p = strategy.p
    protection = p.protection
    lb = p.lookback
    htf_idx = htf.index
    htf_delta = pd.Timedelta(seconds=TimeFrame(strategy.htf_tf).seconds)
    times = ltf.index
    o = ltf["open"].to_numpy(float)
    h = ltf["high"].to_numpy(float)
    l = ltf["low"].to_numpy(float)
    c = ltf["close"].to_numpy(float)
    atr_arr = np.array([np.nan if v is None else v for v in atr_series(
        list(h), list(l), list(c), period=p.atr_period)], dtype=float)
    half = costs.spread / 2
    eq = float(costs.initial_balance)
    trades: list[dict] = []
    trade: Trade | None = None
    meta: dict = {}

    def rate(i: int) -> float:
        if isinstance(quote_to_account, pd.Series):
            return float(quote_to_account.iloc[i])
        return float(quote_to_account)

    def floating(price: float) -> float:
        if trade is None:
            return 0.0
        return trade.side * (price - trade.entry) * meta["lots"] * costs.contract_size * meta["rate"]

    def close(i: int, px: float, reason: str, stop_close: float | None = None) -> None:
        nonlocal eq, trade, meta
        gross = trade.side * (px - trade.entry) * meta["lots"] * costs.contract_size * rate(i)
        net = gross - 2 * costs.commission_per_lot * meta["lots"]
        eq += net
        trades.append({**meta, "exit_time": times[i], "exit": px, "pnl": net, "reason": reason})
        # On an LTF stop-out the ladder uses the close of the bar that closed below the stop.
        strategy.on_exit(reason, stop_close if stop_close is not None else px)
        trade, meta = None, {}

    for i in range(1, len(ltf)):
        bar = i - 1
        # 1) broker exits during bar i-1
        if trade is not None:
            ev = trade.intrabar_exit(h[bar], l[bar])
            if ev is not None:
                px = trade.broker_sl if ev == "DISASTER" else trade.tp
                close(bar, px, ev)
        # 2) close-based rules for bar i-1, executed at the open of bar i
        if trade is not None:
            ev = trade.on_closed_bar(high=h[bar], close=c[bar],
                                     atr=None if np.isnan(atr_arr[bar]) else float(atr_arr[bar]),
                                     p=protection)
            if ev is not None:
                close(i, o[i], ev, stop_close=c[bar] if ev == "SL" else None)
            else:
                meta["trail"] = trade.trail
        # HTF bar forming at the open of bar i. Every HTF bar before it has closed (its end <= t_i).
        f = htf_idx.searchsorted(times[i], side="right") - 1
        k = f - 1
        # 3) HTF floor: last closed HTF bar closed below it
        if trade is not None and k >= 0 and htf["close"].iloc[k] < trade.htf_sl:
            close(i, o[i], "HTF")

        # 4) prop rules
        if prop is not None:
            eq_mark = eq + floating(c[bar])
            prop.update(times[i], eq_mark)
            if trade is not None:
                why = prop.must_flatten(times[i])
                if why:
                    close(i, o[i], "PROP" if "FAILED" in why else "WEEKEND")
                    eq_mark = eq
        # 5) entry
        if trade is not None:
            continue
        if f < 1:
            continue
        frames = {strategy.htf_tf: _with_forming(htf, f, lb), strategy.ltf_tf: _with_forming(ltf, i, lb)}
        if prop is not None and prop.s.status != "ACTIVE":
            continue
        sig = strategy.signal_frames(frames)
        if sig is None:
            continue
        if costs.spread > sig["max_spread_atr"] * sig["atr"]:
            continue
        fill = o[i] + half
        broker_sl = sig["sl_price"]
        sl_dist = fill - broker_sl
        if sl_dist <= 0:
            continue
        r = rate(i)
        risk_per_lot = sl_dist * costs.contract_size * r
        lots = min(math.floor(eq * costs.risk_per_trade / risk_per_lot * 100) / 100, costs.max_lots)
        if prop is not None:
            cap = prop.max_lots_for(eq_mark, risk_per_lot)
            lots = min(lots, math.floor(cap * 100) / 100)
            ok, _ = prop.check_entry(times[i], eq_mark, 0, lots * risk_per_lot)
            if not ok:
                continue
            prop.record_entry(times[i])
        if lots <= 0:
            continue
        tp = fill + p.target_rr * (fill - sig["htf_sl"]) if sig["tp_dist"] is not None else None
        trade = Trade.open(entry=fill, trigger=sig["trigger"], htf_sl=sig["htf_sl"],
                           broker_sl=broker_sl, tp=tp, p=protection, pip=strategy.symbol.pip)
        meta = {"entry_time": times[i], "signal_time": pd.Timestamp(sig["time"]), "entry": fill,
                "lots": lots, "rate": r, "sl": broker_sl, "tp": tp, "htf_sl": sig["htf_sl"],
                "exit_mode": sig["exit_mode"], "retries": sig["retries"], "confirm_bar": sig["confirm_bar"],
                "trail": None}
        strategy.on_entry()

    if trade is not None:
        close(len(ltf) - 1, float(c[-1]), "EOD")
    return pd.DataFrame(trades)
