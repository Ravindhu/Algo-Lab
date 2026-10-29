"""Bot: runs a list of strategies, dedupes signals, and places orders only when allowed.

Safety defaults (same as the live pipeline):
  - dry_run=True: signals are reported, nothing is sent to the broker.
  - An error in one strategy is recorded and the others still run.
  - If the open-position count is unknown, no order is placed (fail closed).
  - Each (strategy, symbol, side, bar time) is acted on once per process.
  - gate(rec) runs before sizing (e.g. prop-firm rules). A False result skips the signal with a reason.
  - sizer(rec) may return {"skip": reason} to skip with a specific reason, or None for a generic skip.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from algolab.framework.strategies import MultiTimeframeStrategy, Strategy

log = logging.getLogger("algolab.bot")
REQUIRED_ORDER_KEYS = ("lots", "sl", "tp")


@dataclass
class Bot:
    adapter: Any                                    # Mt5Adapter, or any object with the same methods
    strategies: list[Strategy] = field(default_factory=list)
    dry_run: bool = True
    max_open: int = 3
    magic: int = 0
    deviation: int = 20
    on_signal: Callable[[dict], Any] | None = None  # optional hook (alerts, logging); may be async
    sizer: Callable[[dict], Any] | None = None      # signal -> {"lots", "sl", "tp"} | {"skip": why} | None
    gate: Callable[[dict], tuple[bool, str]] | None = None
    _seen: set = field(default_factory=set, repr=False)

    def add(self, strategy: Strategy) -> "Bot":
        self.strategies.append(strategy)
        return self

    async def _place(self, rec: dict) -> dict:
        if self.gate:
            ok, why = self.gate(rec)
            if not ok:
                return {**rec, "executed": False, "why": why}
        if self.sizer and any(k not in rec for k in REQUIRED_ORDER_KEYS):
            plan = self.sizer(rec)
            if inspect.isawaitable(plan):
                plan = await plan
            if isinstance(plan, dict) and "skip" in plan:
                return {**rec, "executed": False, "why": plan["skip"]}
            if not plan:
                return {**rec, "executed": False, "why": "size below broker minimum or invalid stop"}
            rec = {**rec, **plan}
        missing = [k for k in REQUIRED_ORDER_KEYS if k not in rec]
        if missing:
            return {**rec, "executed": False, "why": f"order skipped, missing {missing}"}
        if self.dry_run:
            return {**rec, "executed": False, "why": "dry run"}
        open_now = await self.adapter.open_positions()
        if open_now is None:
            return {**rec, "executed": False, "why": "open positions unknown - failing closed"}
        if open_now >= self.max_open:
            return {**rec, "executed": False, "why": "max open positions reached"}
        res = await asyncio.to_thread(self.adapter.market_order, rec["symbol"], rec["side"],
                                      rec["lots"], rec["sl"], rec["tp"],
                                      self.magic, self.deviation)
        return {**rec, "executed": True, **res}

    async def run_once(self) -> list[dict]:
        results: list[dict] = []
        for s in self.strategies:
            try:
                if isinstance(s, MultiTimeframeStrategy):
                    frames = {tf: await s.symbol.candles(self.adapter, tf, s.count)
                              for tf in s.timeframes}
                    sig = s.signal_frames(frames)
                    df = frames[s.timeframes[-1]]       # bar identity comes from the lower timeframe
                else:
                    df = await s.symbol.candles(self.adapter, s.timeframe, s.count)
                    sig = s.signal(df)
            except Exception as e:                      # one broken strategy must not stop the rest
                log.error("strategy %s failed: %s", s.name, e)
                results.append({"strategy": s.name, "error": f"{type(e).__name__}: {e}"})
                continue
            if not sig:
                continue
            bar = str(df.index[-2]) if len(df) >= 2 else str(df.index[-1])
            key = (s.name, s.symbol.name, sig["side"], bar)
            if key in self._seen:
                continue
            self._seen.add(key)                         # mark before acting: no duplicate orders on retry

            rec = {"strategy": s.name, "symbol": s.symbol.name, "bar": bar, **sig}
            rec = await self._place(rec)
            if self.on_signal:
                out = self.on_signal(rec)
                if inspect.isawaitable(out):
                    await out
            results.append(rec)
        return results
