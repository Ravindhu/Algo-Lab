"""WINDOWS ONLY. Run every 15 minutes with Task Scheduler:

  schtasks /Create /SC MINUTE /MO 15 /TN AlgoLab-Sync ^
    /TR "C:\\algolab\\.venv\\Scripts\\python.exe -m algolab.bridge.windows_sync"

Each run:
  1) fetches the latest candles for every symbol in config/ops.json into data/
  2) writes state/account.json (equity, balance, open positions, UTC timestamp)
  3) copies data/ and state/account.json to the VPS over SSH (scp)

Required .env keys: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, VPS_USER, VPS_HOST, VPS_PATH.
Use an SSH key (no password prompt) so the task can run unattended.
Symbol names must match your broker (some brokers add suffixes such as EURUSDm)."""
import json
import logging
import os
import subprocess
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from algolab.ops.state import ROOT, ops_config

log = logging.getLogger("algolab.sync")
TIMEFRAMES = ["H4"]        # must match scanner.htf in the profile
BARS = 1000


def fetch_symbol(symbol: str, tf_name: str):
    rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{tf_name}"), 0, BARS)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"no rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    out = ROOT / "data" / f"{symbol}_{tf_name}.csv"
    out.parent.mkdir(exist_ok=True)
    df[["time", "open", "high", "low", "close", "tick_volume"]].to_csv(out, index=False)
    return out


def account_snapshot() -> dict:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()}")
    positions = mt5.positions_get() or []
    return {"ts": datetime.now(timezone.utc).isoformat(), "equity": info.equity,
            "balance": info.balance, "open_positions": len(positions)}


def write_account(snap: dict) -> None:
    state = ROOT / "state"
    state.mkdir(exist_ok=True)
    tmp = state / "account.json.tmp"
    tmp.write_text(json.dumps(snap))
    os.replace(tmp, state / "account.json")   # atomic on Windows too


def symbols() -> list[str]:
    exported = ROOT / "config" / "symbols.json"
    if exported.exists():
        return json.loads(exported.read_text())["forex"]
    return ops_config()["symbols"]


def push_to_vps() -> None:
    if not os.getenv("VPS_HOST"):
        log.info("VPS_HOST not set: standalone mode, nothing pushed")
        return
    dest = f"{os.environ['VPS_USER']}@{os.environ['VPS_HOST']}:{os.environ['VPS_PATH']}"
    for csv in (ROOT / "data").glob("*.csv"):
        subprocess.run(["scp", "-q", str(csv), f"{dest}/data/"], check=True)
    subprocess.run(["scp", "-q", str(ROOT / "state" / "account.json"), f"{dest}/state/"],
                   check=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not mt5.initialize(login=int(os.environ["MT5_LOGIN"]),
                          password=os.environ["MT5_PASSWORD"],
                          server=os.environ["MT5_SERVER"]):
        log.error("MT5 init failed: %s", mt5.last_error())
        return 1
    failures = 0
    try:
        for sym in symbols():
            for tf in TIMEFRAMES:
                try:
                    fetch_symbol(sym, tf)
                except Exception as e:
                    failures += 1
                    log.error("%s %s: %s", sym, tf, e)
        snap = account_snapshot()
    finally:
        mt5.shutdown()
    write_account(snap)
    push_to_vps()
    log.info("sync done: %d symbol failures", failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
