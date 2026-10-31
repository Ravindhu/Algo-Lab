"""The ONLY module that talks to the MetaTrader 5 terminal.

Uses the official `MetaTrader5` package directly (Windows only). No third-party MT5 wrapper.

Design rules:
  - The MetaTrader5 package is imported lazily, so the rest of the project (backtests, tests,
    dashboard, VPS code) runs on Linux without it.
  - Its calls are blocking. They run in a worker thread via asyncio.to_thread, so the async
    pipeline never freezes while the terminal responds.
  - Every call that can fail checks for None and raises AdapterError with mt5.last_error().
    Callers treat an AdapterError as "do not trade" (fail closed).
  - self_test() checks that every function and constant the project uses exists in the installed
    package. The pipeline refuses to trade if anything is missing.

Check the connection:  python -m algolab.live.adapter
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.metadata as md
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType

import pandas as pd

# Functions the project calls on the MetaTrader5 module.
REQUIRED_FUNCTIONS = (
    "initialize", "shutdown", "last_error", "account_info", "symbols_get", "symbol_select",
    "symbol_info", "symbol_info_tick", "copy_rates_from_pos", "positions_get", "order_send",
    "order_calc_margin", "history_deals_get",
)
# Constants the project reads from the module.
REQUIRED_CONSTANTS = (
    "TRADE_ACTION_DEAL", "TRADE_RETCODE_DONE", "ORDER_TYPE_BUY", "ORDER_TYPE_SELL",
    "ORDER_FILLING_IOC", "ORDER_TIME_GTC",
)
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
# Broker-side filling modes come as a bit mask on symbol_info().filling_mode (MT5 SYMBOL_FILLING_*).
SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2
# Retcodes that mean the order (or part of it) was executed.
FILLED_RETCODES_NAME = ("TRADE_RETCODE_DONE", "TRADE_RETCODE_DONE_PARTIAL")
# Deal entry values for closing deals (MT5 DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT).
CLOSING_ENTRIES = (1, 2, 3)
DEAL_TYPE_BALANCE = 2     # deposits, withdrawals and credits: not trading results


class AdapterError(RuntimeError):
    pass


class PartialClose(AdapterError):
    """The broker closed only part of a position. The position is still open with less volume."""

    def __init__(self, msg: str, remaining: float):
        super().__init__(msg)
        self.remaining = remaining


@dataclass
class AccountState:
    equity: float
    balance: float
    open_positions: int | None      # None = unknown -> pipeline fails closed
    net_deposits: float = 0.0       # deposits minus withdrawals to date (not trading results)


def missing_api(mod: ModuleType) -> list[str]:
    """Names the project needs that the installed MetaTrader5 module does not have."""
    names = [n for n in REQUIRED_FUNCTIONS if not callable(getattr(mod, n, None))]
    names += [n for n in REQUIRED_CONSTANTS if not hasattr(mod, n)]
    names += [f"TIMEFRAME_{t}" for t in TIMEFRAMES if not hasattr(mod, f"TIMEFRAME_{t}")]
    return names


def _field(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


def server_utc_offset(server_epoch: float, utc_epoch: float) -> int:
    """Seconds the broker's server clock runs ahead of UTC, rounded to the nearest 30 minutes.
    MT5 stamps bars and ticks with the broker's server clock, not UTC. server_epoch is the tick time, and
    utc_epoch is this machine's clock (epoch seconds are the same everywhere). The rounding absorbs local
    clock skew of up to 15 minutes."""
    return int(round((server_epoch - utc_epoch) / 1800.0)) * 1800


def _installed_version() -> str:
    try:
        return md.version("MetaTrader5")
    except md.PackageNotFoundError:
        return "not installed"


class Mt5Adapter:
    """Async context manager. Logs in with MT5_LOGIN / MT5_PASSWORD / MT5_SERVER (optional
    MT5_PATH to the terminal exe). With no login set, it uses the terminal's current session."""

    def __init__(self):
        self.version = _installed_version()
        self.mt5: ModuleType | None = None

    # --- lifecycle -----------------------------------------------------------------------
    async def __aenter__(self):
        try:
            self.mt5 = importlib.import_module("MetaTrader5")
        except ImportError as e:
            raise AdapterError("MetaTrader5 package not installed (Windows only): "
                               "pip install MetaTrader5") from e
        kwargs = {}
        if os.getenv("MT5_PATH"):
            kwargs["path"] = os.getenv("MT5_PATH")
        if os.getenv("MT5_LOGIN"):
            kwargs.update(login=int(os.environ["MT5_LOGIN"]),
                          password=os.getenv("MT5_PASSWORD", ""),
                          server=os.getenv("MT5_SERVER", ""))
        ok = await self._run(self.mt5.initialize, **kwargs)
        if not ok:
            raise AdapterError(f"MT5 initialize failed: {self.mt5.last_error()}. "
                               "Open the MT5 terminal and check the login in .env.")
        return self

    async def __aexit__(self, *exc):
        if self.mt5 is not None:
            await self._run(self.mt5.shutdown)

    @staticmethod
    async def _run(fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _err(self, what: str) -> AdapterError:
        return AdapterError(f"{what} failed: {self.mt5.last_error()}")

    # --- contract ------------------------------------------------------------------------
    async def self_test(self) -> tuple[list[str], list[str]]:
        """Returns (problems, warnings). Any problem means: do not trade."""
        if self.mt5 is None:
            return ["MT5 not initialised"], []
        problems = [f"MetaTrader5 missing {n}" for n in missing_api(self.mt5)]
        warnings = []
        if self.version == "not installed":
            warnings.append("MetaTrader5 package version unknown")
        return problems, warnings

    # --- reads ---------------------------------------------------------------------------
    async def symbols(self) -> list[dict]:
        raw = await self._run(self.mt5.symbols_get)
        if raw is None:
            raise self._err("symbols_get")
        return [{"name": _field(s, "name", default=""),
                 "path": _field(s, "path", default=""),
                 "trade_mode": _field(s, "trade_mode", default=None)} for s in raw]

    async def account(self) -> AccountState:
        info = await self._run(self.mt5.account_info)
        if info is None:
            raise self._err("account_info")
        flows = await self._run(self.net_deposits)
        return AccountState(float(info.equity), float(info.balance), await self.open_positions(),
                            net_deposits=flows)

    async def open_positions(self) -> int | None:
        """Number of open positions, or None if the terminal could not answer."""
        positions = await self._run(self.mt5.positions_get)
        if positions is None:          # None = error. An empty tuple means "no positions".
            return None
        return len(positions)

    async def candles(self, symbol: str, tf_name: str, count: int) -> pd.DataFrame:
        """Closed and forming bars, oldest first. Index = bar time in UTC (naive).
        MT5 bar times are on the broker's server clock, so they are converted with the offset from a live tick.
        A tick with no server time fails closed: the bars cannot be placed in UTC."""
        if tf_name not in TIMEFRAMES:
            raise AdapterError(f"unknown timeframe {tf_name}")
        if not await self._run(self.mt5.symbol_select, symbol, True):
            raise self._err(f"symbol_select({symbol})")
        tick = await self._run(self.mt5.symbol_info_tick, symbol)
        server_time = float(_field(tick, "time", default=0) or 0) if tick is not None else 0.0
        if server_time <= 0:
            raise AdapterError(f"{symbol}: no server time from tick; bar times cannot be converted to UTC")
        offset = server_utc_offset(server_time, time.time())
        rates = await self._run(self.mt5.copy_rates_from_pos, symbol,
                                getattr(self.mt5, f"TIMEFRAME_{tf_name}"), 0, count)
        if rates is None or len(rates) == 0:
            raise self._err(f"copy_rates_from_pos({symbol})")
        df = pd.DataFrame(rates)
        need = {"time", "open", "high", "low", "close"}
        if not need <= set(df.columns):
            raise AdapterError(f"{symbol}: candle columns missing {need - set(df.columns)}")
        df["time"] = pd.to_datetime(df["time"] - offset, unit="s")
        return df.set_index("time")[["open", "high", "low", "close"]].sort_index()

    # --- instrument data and orders (synchronous: the pipeline calls them one at a time) ---
    def symbol_info(self, symbol: str) -> dict:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise self._err(f"symbol_info({symbol})")
        return {"contract_size": info.trade_contract_size, "volume_min": info.volume_min,
                "volume_step": info.volume_step,
                "volume_max": float(getattr(info, "volume_max", 100.0)),
                "digits": info.digits,
                "filling_mode": int(getattr(info, "filling_mode", 0) or 0)}

    def filling_type(self, symbol: str) -> int:
        """The order filling mode the symbol accepts. Brokers differ; a hard-coded mode is rejected by some."""
        mode = self.symbol_info(symbol)["filling_mode"]
        if mode & SYMBOL_FILLING_IOC:
            return int(getattr(self.mt5, "ORDER_FILLING_IOC", 1))
        if mode & SYMBOL_FILLING_FOK:
            return int(getattr(self.mt5, "ORDER_FILLING_FOK", 0))
        return int(getattr(self.mt5, "ORDER_FILLING_RETURN", 2))

    def free_margin(self) -> float:
        info = self.mt5.account_info()
        if info is None:
            raise self._err("account_info")
        return float(info.margin_free)

    def margin_for(self, symbol: str, side: str, lots: float) -> float:
        """Margin the broker would require for this order (account currency)."""
        t = self.tick(symbol)
        mt5 = self.mt5
        action = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        price = t["ask"] if side == "BUY" else t["bid"]
        m = mt5.order_calc_margin(action, symbol, float(lots), float(price))
        if m is None:
            raise self._err(f"order_calc_margin({symbol})")
        return float(m)

    def net_deposits(self) -> float:
        """Deposits minus withdrawals, from the broker's balance deals. Used so the kill switch measures
        trading results, not money moved in or out of the account."""
        now = int(datetime.now(timezone.utc).timestamp())
        deals = self.mt5.history_deals_get(0, now + 86400)
        if deals is None:
            raise self._err("history_deals_get")
        return float(sum(float(getattr(d, "profit", 0.0)) for d in deals
                         if int(getattr(d, "type", -1)) == DEAL_TYPE_BALANCE))

    def realized_pnl(self, ticket: int) -> float | None:
        """Realised profit of a closed position from the broker's deal history (profit + commission +
        swap + fee). None when the history has no closing deal yet. The journal then records P&L as unknown
        instead of guessing it."""
        deals = self.mt5.history_deals_get(position=int(ticket))
        if deals is None:
            raise self._err(f"history_deals_get(position={ticket})")
        if not any(int(getattr(d, "entry", -1)) in CLOSING_ENTRIES for d in deals):
            return None
        return float(sum(float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0))
                         + float(getattr(d, "swap", 0.0)) + float(getattr(d, "fee", 0.0))
                         for d in deals))

    def tick(self, symbol: str) -> dict:
        t = self.mt5.symbol_info_tick(symbol)
        if t is None:
            raise self._err(f"tick({symbol})")
        return {"bid": t.bid, "ask": t.ask}

    def market_order(self, symbol: str, side: str, lots: float, sl: float, tp: float,
                     magic: int, deviation: int) -> dict:
        t = self.tick(symbol)
        mt5 = self.mt5
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": t["ask"] if side == "BUY" else t["bid"],
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": "algolab",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_type(symbol),
        }
        res = mt5.order_send(req)
        if res is None:
            # The request may or may not have reached the broker. Do not assume either way: the next
            # cycle's reconciliation looks at real positions before anything else is decided.
            raise AdapterError(f"order_send({symbol}) returned no result: state unknown, check positions")
        if res.retcode not in self._filled_codes():
            raise AdapterError(f"order_send failed: retcode {res.retcode} {getattr(res, 'comment', '')}")
        volume = float(getattr(res, "volume", 0.0) or lots)
        return {"ticket": res.order, "fill_price": res.price, "volume": volume,
                "partial": res.retcode != mt5.TRADE_RETCODE_DONE}

    def _filled_codes(self) -> set:
        codes = {int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009))}
        codes.add(int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)))
        return codes


    # --- account and framework position management (synchronous, like the other order calls) ---
    def account_currency(self) -> str:
        info = self.mt5.account_info()
        if info is None:
            raise self._err("account_info")
        return str(info.currency)

    def positions_by_magic(self, magic: int) -> list[dict]:
        """Open positions opened by this project (matching magic). Raises on a terminal error so the
        caller fails closed. An error is never reported as an empty list."""
        raw = self.mt5.positions_get()
        if raw is None:
            raise self._err("positions_get")
        out = []
        for pos in raw:
            if int(getattr(pos, "magic", 0)) != int(magic):
                continue
            out.append({"ticket": int(pos.ticket), "symbol": str(pos.symbol),
                        "side": 1 if pos.type == self.mt5.ORDER_TYPE_BUY else -1,
                        "price_open": float(pos.price_open), "sl": float(pos.sl),
                        "tp": float(pos.tp), "volume": float(pos.volume)})
        return out

    def position_tickets(self, magic: int) -> set[int]:
        return {p["ticket"] for p in self.positions_by_magic(magic)}

    def close_position(self, ticket: int, symbol: str, side: int, volume: float,
                       magic: int, deviation: int) -> dict:
        """Close an open position at market. A partial close raises PartialClose with the volume still open."""
        t = self.tick(symbol)
        mt5 = self.mt5
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "position": int(ticket), "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_SELL if side > 0 else mt5.ORDER_TYPE_BUY,
            "price": t["bid"] if side > 0 else t["ask"],
            "deviation": deviation, "magic": magic, "comment": "algolab-close",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": self.filling_type(symbol),
        }
        res = mt5.order_send(req)
        if res is None:
            raise AdapterError(f"close_position({ticket}) returned no result: check positions")
        if res.retcode not in self._filled_codes():
            raise AdapterError(f"close_position failed for ticket {ticket}: retcode {res.retcode}")
        closed = float(getattr(res, "volume", 0.0) or volume)
        if closed + 1e-9 < float(volume):
            raise PartialClose(f"partial close of ticket {ticket}: {closed} of {volume}",
                               remaining=round(float(volume) - closed, 8))
        return {"ticket": int(ticket), "fill_price": float(res.price), "volume": closed}

async def _main() -> int:
    async with Mt5Adapter() as ad:
        problems, warnings = await ad.self_test()
        print(json.dumps({"metatrader5": ad.version, "problems": problems,
                          "warnings": warnings}, indent=2))
        return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
