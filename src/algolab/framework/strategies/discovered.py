# LIVE strategy built from a discovery candidate. One file serves every promoted genome: the genome
# (entry family, filter, SL/TP) comes from config/framework.json, and the signal logic is the SAME code
# discovery backtested (discovery_factory.sig_and_fill_for_genome). So live cannot drift from the test.
#
# Rules:
#   - Only the last CLOSED bar counts. The forming bar is dropped before the signal is computed.
#   - Entry is a market order (next bar open, the same as the backtest's next-open entry).
#   - Broker-side SL and TP (ATR multiples from the genome), plus the genome's fixed hold as a time exit.
#     The position manager applies the hold (positions.new_plain_record hold_bars), so live closes on the same
#     bar the backtest does. No trail and no breakeven: the backtest does not model them, so live must not.
#   - Only the symbol named in params["only_symbol"] trades. Promotion writes it, so one candidate
#     never trades symbols it was not tested on.
#   - Genomes whose entry needs a resting limit (fib families) are refused at construction.
from __future__ import annotations

import math

import pandas as pd

from algolab.framework.strategies.base import Strategy

MIN_BARS = 300          # enough history for the indicators in any family (EMA 200, Ichimoku, ...)


class DiscoveredStrategy(Strategy):
    name = "discovered"

    def __init__(self, symbol, timeframe="M15", count=400, genome=None, only_symbol=None, **extra):
        if not isinstance(genome, dict):
            raise ValueError("discovered: params.genome is required (a candidate from reports/discovery.json)")
        from algolab import discovery_factory as df_mod           # lazy: keeps MT5-free imports clean
        ok, why = df_mod.live_compatible(genome)
        if not ok:
            raise ValueError(f"discovered: genome not live-compatible: {why}")
        super().__init__(symbol, timeframe, count, genome=genome, only_symbol=only_symbol, **extra)
        self.genome = genome
        self.only_symbol = only_symbol
        self._df = df_mod

    def signal(self, df: pd.DataFrame) -> dict | None:
        if self.only_symbol and self.symbol.name != self.only_symbol:
            return None
        if len(df) < MIN_BARS + 1:
            return None
        closed = df.iloc[:-1]                                      # drop the still-forming bar
        sig, _fill = self._df.sig_and_fill_for_genome(closed, self.genome)
        s = int(sig[-1])
        if s == 0:
            return None
        atr_v = float(self._df.Ind(closed).atr(14)[-1])
        if not math.isfinite(atr_v) or atr_v <= 0:
            return None
        return {"side": "BUY" if s == 1 else "SELL", "atr": atr_v,
                "sl_dist": float(self.genome["sl"]) * atr_v,
                "tp_dist": float(self.genome["tp"]) * atr_v,
                "time": str(closed.index[-1])}


# Plugin registry export. Promoted candidates are config entries of type "discovered".
STRATEGY = DiscoveredStrategy
