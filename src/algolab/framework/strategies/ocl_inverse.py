"""Inverse (SELL) versions of the OCL blueprints: ocl_inv, ocl_sell_fail_top_inv, ocl_sell_fail_bottom_inv.

Mirror principle. Negate every price (open, high, low, close) and swap high and low. A bullish pattern on
the negated chart is exactly the bearish pattern on the real chart. The strategy code is reused unchanged:
  - red and green swap, so "C1 red, C2 red, C3 green" becomes "green, green, red".
  - RSI becomes 100 - RSI, and the HM gate "RSI above EMA" becomes "RSI below EMA".
  - ATR is unchanged. Levels, stops and targets change sign.
The wrapper negates the frames on the way in, and the signal's prices on the way out, so the Bot, the sizer,
the backtest and the live position manager all see real prices with side SELL. The strategy's own state stays
in negated space, so on_exit() prices are negated too.

Each inverse blueprint keeps its own handoff namespace ("SELL:") so it never shares state with the BUY versions.
"""
from __future__ import annotations

import pandas as pd

from algolab.framework.strategies.multi import MultiTimeframeStrategy
from algolab.framework.strategies.ocl import OclKeylevel
from algolab.framework.strategies.engulfing_ocl import EngulfingOcl
from algolab.framework.strategies.ocl_sell_fail import OclSellFailBottom, OclSellFailTop

# Signal keys holding a price level. Distances (sl_dist, tp_dist), ATR and protection settings are not mirrored.
PRICE_KEYS = ("sl_price", "htf_sl", "trigger", "aggressive", "conservative", "entry_ref")


def mirror_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["open"] = -df["open"]
    out["close"] = -df["close"]
    out["high"] = -df["low"]
    out["low"] = -df["high"]
    return out


def mirror_frames(frames: dict) -> dict:
    return {tf: mirror_frame(df) for tf, df in frames.items()}


def mirror_signal(sig: dict) -> dict:
    out = dict(sig)
    out["side"] = "SELL"
    for k in PRICE_KEYS:
        if out.get(k) is not None:
            out[k] = -float(out[k])
    if out.get("rsi") is not None:                  # RSI of the mirrored chart is 100 - RSI
        out["rsi"] = 100.0 - float(out["rsi"])
    return out


class _Inverted(MultiTimeframeStrategy):
    """Runs a BUY blueprint on mirrored prices. Subclasses set `name` and `base`."""

    name = "inverted"
    base: type = OclKeylevel

    def __init__(self, symbol, htf="15minute", ltf="5minute", count=None, **params):
        self.inner = self.base(symbol, htf=htf, ltf=ltf, count=count, **params)
        self.symbol = self.inner.symbol
        self.htf_tf = self.inner.htf_tf
        self.ltf_tf = self.inner.ltf_tf
        self.timeframes = self.inner.timeframes
        self.count = self.inner.count
        self.p = self.inner.p
        self.params = params

    @property
    def st(self):
        return self.inner.st

    def state(self) -> dict:
        return self.inner.state()

    def restore(self, d: dict) -> None:
        self.inner.restore(d)

    def signal_frames(self, frames: dict) -> dict | None:
        sig = self.inner.signal_frames(mirror_frames(frames))
        return None if sig is None else mirror_signal(sig)

    def exit_check(self, frames: dict) -> bool:
        return self.inner.exit_check(mirror_frames(frames))

    def on_entry(self) -> None:
        self.inner.on_entry()

    def on_exit(self, reason: str, price: float) -> None:
        self.inner.on_exit(reason, -float(price))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.symbol.name}, {self.params})"


class _SellFailTopInner(OclSellFailTop):
    handoff_scope = "SELL:"


class _SellFailBottomInner(OclSellFailBottom):
    handoff_scope = "SELL:"


class OclInv(_Inverted):
    """Inverse of the normal OCL blueprint: SELL on a failed-up structure with bearish HM."""
    name = "ocl_inv"
    base = OclKeylevel


class OclSellFailTopInv(_Inverted):
    name = "ocl_sell_fail_top_inv"
    base = _SellFailTopInner


class OclSellFailBottomInv(_Inverted):
    name = "ocl_sell_fail_bottom_inv"
    base = _SellFailBottomInner


class EngulfingOclInv(_Inverted):
    """SELL side of the engulfing OCL model: bearish engulfing, fib level above, OCL SELL entry."""
    name = "engulfing_ocl_inv"
    base = EngulfingOcl


STRATEGY = [OclInv, OclSellFailTopInv, OclSellFailBottomInv, EngulfingOclInv]
