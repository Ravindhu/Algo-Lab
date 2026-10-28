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
from dataclasses import dataclass
from types import ModuleType

import pandas as pd

# Functions the project calls on the MetaTrader5 module.
REQUIRED_FUNCTIONS = (
    "initialize", "shutdown", "last_error", "account_info", "symbols_get", "symbol_select",
    "symbol_info", "symbol_info_tick", "copy_rates_from_pos", "positions_get", "order_send",
)
# Constants the project reads from the module.
REQUIRED_CONSTANTS = (
    "TRADE_ACTION_DEAL", "TRADE_RETCODE_DONE", "ORDER_TYPE_BUY", "ORDER_TYPE_SELL",
    "ORDER_FILLING_IOC", "ORDER_TIME_GTC",
)
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")


class AdapterError(RuntimeError):
    pass


@dataclass
class AccountState:
    equity: float
    balance: float
    open_positions: int | None      # None = unknown -> pipeline fails closed


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
        return AccountState(float(info.equity), float(info.balance), await self.open_positions())

    async def open_positions(self) -> int | None:
        """Number of open positions, or None if the terminal could not answer."""
        positions = await self._run(self.mt5.positions_get)
        if positions is None:          # None = error. An empty tuple means "no positions".
            return None
        return len(positions)

    async def candles(self, symbol: str, tf_name: str, count: int) -> pd.DataFrame:
        """Closed and forming bars, oldest first. Index = bar time (UTC, naive)."""
        if tf_name not in TIMEFRAMES:
            raise AdapterError(f"unknown timeframe {tf_name}")
        if not await self._run(self.mt5.symbol_select, symbol, True):
            raise self._err(f"symbol_select({symbol})")
        rates = await self._run(self.mt5.copy_rates_from_pos, symbol,
                                getattr(self.mt5, f"TIMEFRAME_{tf_name}"), 0, count)
        if rates is None or len(rates) == 0:
            raise self._err(f"copy_rates_from_pos({symbol})")
        df = pd.DataFrame(rates)
        need = {"time", "open", "high", "low", "close"}
        if not need <= set(df.columns):
            raise AdapterError(f"{symbol}: candle columns missing {need - set(df.columns)}")
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.set_index("time")[["open", "high", "low", "close"]].sort_index()

    # --- instrument data and orders (synchronous: the pipeline calls them one at a time) ---
    def symbol_info(self, symbol: str) -> dict:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise self._err(f"symbol_info({symbol})")
        return {"contract_size": info.trade_contract_size, "volume_min": info.volume_min,
                "volume_step": info.volume_step, "digits": info.digits}

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
            "type_filling": mt5.ORDER_FILLING_IOC,   # some brokers need FOK or RETURN
        }
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            raise AdapterError(f"order_send failed: {res}")
        return {"ticket": res.order, "fill_price": res.price}


async def _main() -> int:
    async with Mt5Adapter() as ad:
        problems, warnings = await ad.self_test()
        print(json.dumps({"metatrader5": ad.version, "problems": problems,
                          "warnings": warnings}, indent=2))
        return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
