"""Shared sizing helpers. Both the legacy pipeline and the framework size from a risk budget in the
account currency, so the quote-to-account conversion lives in one place."""
from __future__ import annotations

import math

from algolab.live.adapter import AdapterError


def quote_to_account_rate(ad, symbol: str, account: str) -> float | None:
    """Account currency per 1 unit of the pair's QUOTE currency. None if no conversion pair exists.
    EURUSD on a USD account -> 1.0. USDJPY on a USD account -> 1 / USDJPY mid (JPY quote).
    GBPJPY on a EUR account needs EURJPY; the function tries both directions of account/quote."""
    base, suffix = symbol[:6], symbol[6:]
    quote = base[3:6]
    if quote == account:
        return 1.0
    for pair, invert in ((account + quote, True), (quote + account, False)):
        try:
            t = ad.tick(pair + suffix)
        except AdapterError:
            continue
        mid = (t["bid"] + t["ask"]) / 2
        if mid > 0:
            return 1.0 / mid if invert else mid
    return None


def lots_from_risk(risk_money: float, sl_dist: float, contract_size: float, rate: float,
                   max_lots: float, volume_step: float = 0.01) -> float:
    """Lots so that a stop at sl_dist (price units) risks risk_money in account currency.
    sl_dist * contract_size is the loss per lot in the QUOTE currency; rate converts it to the account."""
    if sl_dist <= 0 or contract_size <= 0 or rate <= 0:
        return 0.0
    risk_per_lot = sl_dist * contract_size * rate
    # Floor with a small epsilon: float division can land just below a whole step (2.9999... for 3.0),
    # and int() would then drop a step. Same rule as framework_runner._snap_volume.
    steps = math.floor(risk_money / risk_per_lot / volume_step + 1e-9)
    return round(min(steps * volume_step, max_lots), 8)
