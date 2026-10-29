"""Engulfing body filter and fib entry, and the SELL (inverse) OCL blueprints.

Inverse rule under test: the SELL blueprint on real candles equals the BUY blueprint on mirrored candles,
with every price negated. Mirrored state is kept, so the two strategies stay in lockstep bar by bar.
"""
import numpy as np
import pandas as pd
import pytest

from algolab.framework.bot import Bot  # noqa: F401  (import check: the Bot module loads with the new strategies)
from algolab.framework.ocl_backtest import run_inverse_backtest, run_ocl_backtest
from algolab.framework.protect import Protection, Trade
from algolab.framework.strategies import engulfing_strategy as es
from algolab.framework.strategies.engulfing_strategy import Engulfing, fib_entry_level
from algolab.framework.strategies.ocl import OclKeylevel
from algolab.framework.strategies.ocl_inverse import OclInv, OclSellFailTopInv, mirror_frame
from algolab.framework.strategies.ocl_sell_fail import OclSellFailTop
from algolab.framework.symbol import ForexSymbol
from algolab.backtest.engine import Costs
from algolab.framework.strategies.engulfing_strategy import engulfing

EngulfingHmStrategy = Engulfing   # back-compat name used throughout this file


def frame(rows, start="2026-01-05", freq="h"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


# ---------- engulfing body filter -------------------------------------------------

def _pair(cur_high):
    # previous candle: red, body 0.8 of range 1.2. Current: green engulfing, body 1.0 of range 1.2 or wider.
    return frame([(10.0, 10.2, 9.0, 9.2), (9.1, cur_high, 9.0, 10.1)])


def test_engulfing_accepts_bodies_over_half_range():
    assert int(engulfing(_pair(10.2), min_body_pct=0.5).iloc[-1]) == 1


def test_engulfing_rejects_small_body_on_the_engulfing_candle():
    # body 1.0, range 2.5 -> 40 percent: too small
    assert int(engulfing(_pair(11.5), min_body_pct=0.5).iloc[-1]) == 0


def test_engulfing_body_filter_can_be_switched_off():
    assert int(engulfing(_pair(11.5), min_body_pct=0.0).iloc[-1]) == 1


def test_engulfing_body_filter_checks_the_engulfed_candle_too():
    rows = [(10.0, 10.5, 9.0, 9.2), (9.1, 10.2, 9.0, 10.1)]     # previous body 0.8 of range 1.5 -> 53% passes
    assert int(engulfing(frame(rows)).iloc[-1]) == 1
    rows = [(10.0, 11.0, 9.0, 9.2), (9.1, 10.2, 9.0, 10.1)]     # previous body 0.8 of range 2.0 -> 40% fails
    assert int(engulfing(frame(rows)).iloc[-1]) == 0


def test_fib_entry_level_is_88_6_percent_into_the_candle():
    assert fib_entry_level(10.2, 9.0, +1) == pytest.approx(10.2 - 0.886 * 1.2)
    assert fib_entry_level(10.2, 9.0, -1) == pytest.approx(9.0 + 0.886 * 1.2)


# ---------- engulfing fib entry (HM gate is set per test) --------------------------

def _eng_frame(direction_rows, last_rows=()):
    """25 green prefix bars, a red bar, the engulfing pattern, then the given closed bars and a forming bar."""
    prefix = [(10.0, 10.1, 9.9, 10.05)] * 25
    rows = prefix + [(10.0, 10.2, 9.0, 9.2)] + list(direction_rows) + list(last_rows)
    rows = rows + [rows[-1]]          # the forming bar at the very end
    return frame(rows)


def _set_hm(monkeypatch, value):
    monkeypatch.setattr(es, "hm_direction",
                        lambda closes, r, e: np.full(len(closes), value, dtype=int))


ENG = (9.1, 10.2, 9.0, 10.1)          # bullish engulfing, body 1.0 of 1.2
TOUCH = (10.1, 10.3, 9.0, 10.2)       # low 9.0 reaches the 88.6% level (9.137)
NO_TOUCH = (10.1, 10.3, 9.5, 10.2)    # low 9.5 does not


IDX = pd.date_range("2026-01-05", periods=400, freq="h")


def test_fib_entry_fires_on_the_first_touch_bar(monkeypatch):
    _set_hm(monkeypatch, +1)
    df = _eng_frame([ENG], [TOUCH])
    sig = Engulfing(ForexSymbol("EURUSD"), "H1").signal(df)
    assert sig is not None and sig["side"] == "BUY"
    assert sig["entry"] == pytest.approx(fib_entry_level(10.2, 9.0, +1))


def test_no_signal_when_the_level_was_never_touched(monkeypatch):
    _set_hm(monkeypatch, +1)
    df = _eng_frame([ENG], [NO_TOUCH])
    assert Engulfing(ForexSymbol("EURUSD"), "H1").signal(df) is None


def test_earlier_touch_voids_the_pattern(monkeypatch):
    _set_hm(monkeypatch, +1)
    df = _eng_frame([ENG, TOUCH], [TOUCH])          # touched on the bar after the pattern, and again later
    assert Engulfing(ForexSymbol("EURUSD"), "H1").signal(df) is None


def test_sell_side_fib_entry_is_the_mirror(monkeypatch):
    _set_hm(monkeypatch, -1)
    df = mirror_frame(_eng_frame([ENG], [TOUCH]))
    sig = Engulfing(ForexSymbol("EURUSD"), "H1").signal(df)
    assert sig is not None and sig["side"] == "SELL"
    # mirrored prices: the level sits fib * range above the low of the mirrored candle
    assert sig["entry"] == pytest.approx(-fib_entry_level(10.2, 9.0, +1))


# ---------- inverse OCL: mirror identity ------------------------------------------

def _walk(seed, n):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.4, n))
    opens = np.concatenate([[closes[0]], closes[:-1]]) + rng.normal(0, 0.1, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.3, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.3, n))
    return np.column_stack([opens, highs, lows, closes])


def _tf_frames(rows, htf_name, ltf_name, n_htf_per_ltf=3):
    ltf = frame(rows, freq="5min")
    htf_rows = []
    for i in range(0, len(rows) - n_htf_per_ltf + 1, n_htf_per_ltf):
        chunk = rows[i:i + n_htf_per_ltf]
        htf_rows.append((chunk[0, 0], chunk[:, 1].max(), chunk[:, 2].min(), chunk[-1, 3]))
    htf = frame(np.array(htf_rows), freq="15min")
    return {htf_name: htf, ltf_name: ltf}


def _same_mirrored(sig_base, sig_inv):
    if sig_base is None or sig_inv is None:
        return sig_base is None and sig_inv is None
    assert sig_inv["side"] == "SELL" and sig_base["side"] == "BUY"
    for k in ("sl_price", "htf_sl", "trigger", "aggressive", "conservative"):
        if k in sig_base:
            assert sig_inv[k] == pytest.approx(-sig_base[k], abs=1e-9), k
    assert sig_inv["sl_dist"] == pytest.approx(sig_base["sl_dist"], abs=1e-9)
    return True


def test_inverse_ocl_is_the_buy_blueprint_on_mirrored_candles():
    sym = ForexSymbol("EURUSD")
    base = OclKeylevel(sym, htf="15minute", ltf="5minute")
    inv = OclInv(sym, htf="15minute", ltf="5minute")
    checked = fired = 0
    for seed in range(6):
        rows = _walk(seed, 900)
        frames = _tf_frames(rows, base.htf_tf, base.ltf_tf)
        base.st = type(base.st)()
        inv.restore({})
        for end in range(60, len(rows) - 1):
            cut = {tf: df.iloc[: max(1, int(len(df) * end / len(rows)))] for tf, df in frames.items()}
            cut = {tf: df for tf, df in cut.items() if len(df) >= 30}
            if len(cut) < 2:
                continue
            a = base.signal_frames({tf: mirror_frame(df) for tf, df in cut.items()})
            b = inv.signal_frames(cut)
            _same_mirrored(a, b)
            checked += 1
            fired += a is not None
    assert checked > 500
    assert fired > 0, "fixture never produced an OCL setup; widen the walk"


def test_inverse_sell_fail_top_is_the_buy_blueprint_on_mirrored_candles(monkeypatch):
    # The sell-failure blueprints record exhausted cores in a shared handoff file. Keep it in memory here.
    from algolab.framework.strategies import ocl_sell_fail as sf
    memory = {}
    monkeypatch.setattr(sf.HandoffStore, "exhausted",
                        lambda self, symbol, key: memory.get(self._scope + symbol, {}).get("key") == key)
    monkeypatch.setattr(sf.HandoffStore, "mark_exhausted",
                        lambda self, symbol, key: memory.__setitem__(self._scope + symbol, {"key": key}))
    sym = ForexSymbol("EURUSD")
    base = OclSellFailTop(sym, htf="15minute", ltf="5minute")
    inv = OclSellFailTopInv(sym, htf="15minute", ltf="5minute")
    checked = fired = 0
    for seed in range(4):
        rows = _walk(seed + 50, 900)
        frames = _tf_frames(rows, base.htf_tf, base.ltf_tf)
        base.st = base._new_state()
        inv.restore({})
        for end in range(60, len(rows) - 1, 3):
            cut = {tf: df.iloc[: max(1, int(len(df) * end / len(rows)))] for tf, df in frames.items()}
            cut = {tf: df for tf, df in cut.items() if len(df) >= 30}
            if len(cut) < 2:
                continue
            a = base.signal_frames({tf: mirror_frame(df) for tf, df in cut.items()})
            b = inv.signal_frames(cut)
            _same_mirrored(a, b)
            checked += 1
            fired += a is not None
    assert checked > 100
    assert fired > 0, "fixture never produced a sell-failure setup; widen the walk"


def test_inverse_names_and_config_types_resolve():
    from algolab.framework.strategies import registry
    for name in ("ocl_inv", "ocl_sell_fail_top_inv", "ocl_sell_fail_bottom_inv"):
        assert registry.load(name).name == name


# ---------- SELL trade protection and backtest -------------------------------------

def test_sell_trade_is_protected_in_mirrored_space():
    from algolab.live.positions import new_record
    rec = {"fill_price": 1.2000, "ticket": 1, "symbol": "EURUSD", "lots": 0.1, "side": "SELL",
           "trigger": 1.2050, "htf_sl": 1.2100, "sl": 1.2130, "tp": 1.1800,
           "bar": "2026-01-05 01:00", "protect": Protection().to_dict()}
    r = new_record(rec, strategy="ocl_inv", ltf="M5", htf="M15")
    t = Trade.from_dict(r["trade"])
    # stored negated: the stop (1.2130) must sit above entry 1.2000 for a short
    assert -t.broker_sl > -t.entry
    # a close above the stop (in real prices) is an SL for a SELL
    assert t.on_closed_bar(high=-1.1990, close=-1.2140, atr=0.001, p=Protection()) == "SL"


def test_inverse_backtest_takes_a_short_on_a_bearish_market():
    # test_ocl_backtest builds a bullish scenario that the BUY blueprint takes to target. Mirror it into a
    # bearish market: the inverse blueprint should take the same setup as a SELL and reach its target.
    from test_ocl_backtest import ENTRY, scenario, strategy
    htf, ltf = scenario([ENTRY, (1.092, 1.145, 1.091, 1.140)])
    bearish_htf, bearish_ltf = mirror_frame(htf), mirror_frame(ltf)
    costs = Costs()
    tr = run_inverse_backtest(strategy(), bearish_htf, bearish_ltf, costs)
    assert len(tr) == 1, "the inverse backtest should take exactly one short"
    t = tr.iloc[0]
    assert t["side"] == "SELL" and t["reason"] == "TP"
    assert t["sl"] > t["entry"], "a short's stop sits above its entry"
    assert t["tp"] < t["entry"], "a short's target sits below its entry"
    assert t["pnl"] > 0
