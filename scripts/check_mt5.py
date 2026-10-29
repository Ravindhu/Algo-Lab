"""Checks the MetaTrader 5 connection: package present, terminal logged in, every API the project
uses available. Used by install_windows.ps1 after install. Never blocks the install: a failed
check prints what to fix and returns 0 (the scheduled tasks will report problems later).

Run:  python scripts/check_mt5.py
"""
import asyncio
import json
import sys

from algolab.live.adapter import Mt5Adapter


async def main() -> int:
    try:
        async with Mt5Adapter() as ad:
            problems, warnings = await ad.self_test()
            acct = await ad.account()
    except Exception as e:
        print(f"NOT READY: {type(e).__name__}: {e}")
        print("Open the MT5 terminal, log in to your broker once, then re-run the installer.")
        return 0
    if problems:
        print("MT5 API problems (trading stays off):", "; ".join(problems))
        return 0
    print(json.dumps({"status": "OK", "metatrader5": ad.version, "warnings": warnings,
                      "account_login_ok": True, "equity": acct.equity}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
