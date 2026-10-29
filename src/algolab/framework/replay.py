"""Replay: runs the LIVE framework code path over historical bars with a simulated broker.

SimBroker implements the adapter methods the framework calls (tick, symbol_info, account_currency,
positions_by_magic, market_order, close_position, open_positions, candles). It is NOT MT5: fills, spread
and commission are simplified. What it exercises for real is run_framework(): the sizer, the prop gate,
Bot dedupe, the position manager and state persistence.

Bar ordering per LTF bar i (decision at its open), the same as the backtest and the live cycle:
  1. broker-side exits during bar i-1 (disaster stop, target), from bar i-1's high and low;
  2. run_framework at the open of bar i: manage closed bars through i-1, prop rules, entries.
"""
from __future__ import annotations

import pandas as pd

from algolab.live.adapter import AdapterError
from algolab.framework.prop import PropGuard  # noqa: F401  (re-exported type for callers)

CONTRACT = 100000.0


class SimBroker:
    def __init__(self, data: dict[str, dict[str, pd.DataFrame]], index: pd.DatetimeIndex,
                 spreads: dict[str, float], balance: float, ltf: str = "M5",
                 commission_per_lot: float = 3.5):
        self.data = data                  # symbol -> {"M5": df, "M15": df}, all on the same LTF index
        self.index = index
        self.spreads = spreads
        self.ltf = ltf
        self.start_balance = float(balance)
        self.balance = float(balance)
        self.commission = commission_per_lot
        self.i = 0
        self.now = index[0]
        self.positions: dict[int, dict] = {}
        self.trades: list[dict] = []
        self.orders: list[dict] = []
        self._ticket = 1000

    # ---- time ----
    def set_index(self, i: int) -> None:
        self.i = i
        self.now = self.index[i]

    def _ohlc(self, sym: str, tf: str, j: int) -> tuple[float, float, float, float]:
        r = self.data[sym][tf].iloc[j]
        return float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])

    def _mid(self, sym: str) -> float:
        return self._ohlc(sym, self.ltf, self.i)[0]

    def rate(self, sym: str) -> float:
        """Account currency (USD) per unit of the pair's quote currency, at the current bar."""
        quote = sym[3:6]
        if quote == "USD":
            return 1.0
        if quote == "JPY":
            return 1.0 / self._mid("USDJPY")
        raise AdapterError(f"sim has no conversion for quote {quote}")

    # ---- adapter surface used by the framework ----
    def tick(self, symbol: str) -> dict:
        mid = self._mid(symbol)
        half = self.spreads[symbol] / 2
        return {"bid": mid - half, "ask": mid + half}

    def symbol_info(self, symbol: str) -> dict:
        return {"contract_size": CONTRACT, "volume_min": 0.01, "volume_step": 0.01,
                "digits": 3 if symbol[3:6] == "JPY" else 5}

    def account_currency(self) -> str:
        return "USD"

    async def open_positions(self) -> int:
        return len(self.positions)

    def positions_by_magic(self, magic: int) -> list[dict]:
        return [dict(p) for p in self.positions.values() if p["magic"] == int(magic)]

    async def candles(self, symbol: str, tf_name: str, count: int) -> pd.DataFrame:
        """Closed bars plus a forming row. The forming row carries only its OPEN, so no future data leaks."""
        df = self.data[symbol][tf_name]
        if tf_name == self.ltf:
            rows = df.iloc[: self.i + 1].copy()
            pos = self.i
        else:
            pos = df.index.searchsorted(self.now, side="right") - 1
            rows = df.iloc[: pos + 1].copy()
        if pos < 0:
            raise AdapterError(f"no {tf_name} bar for {symbol} at {self.now}")
        o = float(df["open"].iloc[pos])
        rows.iloc[-1, rows.columns.get_loc("high")] = o
        rows.iloc[-1, rows.columns.get_loc("low")] = o
        rows.iloc[-1, rows.columns.get_loc("close")] = o
        rows.iloc[-1, rows.columns.get_loc("open")] = o
        return rows.iloc[-count:]

    def market_order(self, symbol: str, side: str, lots: float, sl: float, tp: float,
                     magic: int, deviation: int) -> dict:
        if side != "BUY":
            raise AdapterError("sim supports BUY only (the OCL strategy is BUY only)")
        fill = self.tick(symbol)["ask"]
        if not sl < fill:
            raise AdapterError(f"invalid stop {sl} for fill {fill}")
        self._ticket += 1
        t = self._ticket
        self.positions[t] = {"ticket": t, "symbol": symbol, "side": 1, "price_open": fill,
                             "sl": float(sl), "tp": float(tp) if tp and tp > 0 else None,
                             "volume": float(lots), "magic": int(magic), "time": self.now}
        self.orders.append({"ticket": t, "symbol": symbol, "lots": float(lots), "fill": fill,
                            "sl": float(sl), "tp": tp, "time": self.now, "equity": self.equity_mark()})
        return {"ticket": t, "fill_price": fill}

    def close_position(self, ticket: int, symbol: str, side: int, volume: float,
                       magic: int, deviation: int) -> dict:
        if int(ticket) not in self.positions:
            raise AdapterError(f"close_position: ticket {ticket} not open")
        px = self.tick(symbol)["bid"] if side > 0 else self.tick(symbol)["ask"]
        self._close(int(ticket), px, "CLOSE")
        return {"ticket": int(ticket), "fill_price": px}

    # ---- broker-side fills and accounting ----
    def _close(self, ticket: int, px: float, reason: str) -> None:
        pos = self.positions.pop(ticket)
        gross = (px - pos["price_open"]) * pos["volume"] * CONTRACT * self.rate(pos["symbol"])
        net = gross - 2 * self.commission * pos["volume"]
        self.balance += net
        self.trades.append({"ticket": ticket, "symbol": pos["symbol"], "lots": pos["volume"],
                            "entry_time": str(pos["time"]), "exit_time": str(self.now),
                            "entry": pos["price_open"], "exit": px, "pnl": net, "reason": reason})

    def intrabar(self, j: int) -> None:
        """Broker stops and targets that fired inside bar j (its high and low)."""
        for t in list(self.positions):
            pos = self.positions.get(t)
            if pos is None:
                continue
            _, high, low, _ = self._ohlc(pos["symbol"], self.ltf, j)
            if low <= pos["sl"]:
                self._close(t, pos["sl"], "DISASTER")
            elif pos["tp"] is not None and high >= pos["tp"]:
                self._close(t, pos["tp"], "TP")

    def equity_mark(self) -> float:
        floating = 0.0
        for pos in self.positions.values():
            floating += (self._mid(pos["symbol"]) - pos["price_open"]) * pos["volume"] * CONTRACT \
                * self.rate(pos["symbol"])
        return self.balance + floating

    # ---- adapter methods added for the live safety checks ----
    def margin_for(self, symbol: str, side: str, lots: float) -> float:
        """Margin at 1:100 leverage in account currency (approximation for the simulation)."""
        mid = self._mid(symbol)
        return float(lots) * CONTRACT * mid * self.rate(symbol) / 100.0

    def free_margin(self) -> float:
        used = sum(self.margin_for(p["symbol"], "BUY", p["volume"]) for p in self.positions.values())
        return max(self.balance - used, 0.0)

    def position_tickets(self, magic: int) -> set[int]:
        return {t for t, p in self.positions.items() if p["magic"] == int(magic)}

    def realized_pnl(self, ticket: int) -> float | None:
        for t in reversed(self.trades):
            if t["ticket"] == int(ticket):
                return float(t["pnl"])
        return None

    def close_all(self, reason: str) -> None:
        for t in list(self.positions):
            pos = self.positions[t]
            self._close(t, self._mid(pos["symbol"]), reason)


async def run_replay(sim: SimBroker, symbols: list[str], cfg: dict, prop_cfg: dict | None,
                     warmup: int, on_bar=None) -> dict:
    """Run run_framework() bar by bar over the simulated broker. Returns the equity curve and events.
    on_bar(sim, i), if given, runs before each cycle. Tests use it to inject losses at a known bar."""
    from algolab.live.framework_runner import run_framework

    events: list[dict] = []
    results: list[dict] = []
    curve: list[dict] = []
    for i in range(warmup, len(sim.index)):
        sim.set_index(i)
        sim.intrabar(i - 1)
        if on_bar is not None:
            on_bar(sim, i)
        eq = sim.equity_mark()
        fw = await run_framework(sim, {"dry_run": False}, symbols, eq, now=sim.now,
                                 cfg=cfg, prop_cfg=prop_cfg)
        status = None
        if fw is not None:
            events.extend(fw["events"])
            results.extend(fw["results"])
            status = fw["prop"]["status"] if fw["prop"] else None
        curve.append({"t": sim.now, "equity": eq, "status": status, "orders": len(sim.orders)})
    sim.set_index(len(sim.index) - 1)
    sim.close_all("EOD")
    return {"curve": curve, "events": events, "results": results}
