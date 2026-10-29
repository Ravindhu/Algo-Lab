"""WINDOWS. Read the forex symbols your broker offers and write config/symbols.json.

Run after MT5 login (ALGOLAB_PROFILE=local python scripts/export_symbols.py).
Review the output once: broker folder names differ, and some CFDs can be included or missed."""
import asyncio
import json
from datetime import datetime, timezone

from algolab.live.adapter import Mt5Adapter
from algolab.profile import ROOT

CURRENCIES = {"EUR", "USD", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "NOK", "SEK",
              "SGD", "MXN", "ZAR", "TRY", "HKD", "PLN", "CZK", "HUF", "DKK", "CNH"}


def is_forex(name: str, path: str) -> bool:
    base = name[:6]          # brokers often add suffixes, e.g. EURUSDm
    is_pair = (len(base) == 6 and base.isalpha() and base.isupper()
               and base[:3] in CURRENCIES and base[3:] in CURRENCIES)
    return is_pair or "forex" in path.lower()


async def main() -> None:
    async with Mt5Adapter() as ad:
        symbols = await ad.symbols()
        forex = sorted({s["name"] for s in symbols
                        if s["name"] and is_forex(s["name"], s["path"] or "")
                        and s["trade_mode"] != 0})          # 0 = trading disabled
        out = {"generated": datetime.now(timezone.utc).isoformat(),
               "count": len(forex), "forex": forex}
        (ROOT / "config" / "symbols.json").write_text(json.dumps(out, indent=2))
        print(f"{len(forex)} forex symbols written to config/symbols.json")


if __name__ == "__main__":
    asyncio.run(main())
