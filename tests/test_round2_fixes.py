"""Regression tests for the audit findings. Each test pins one specific failure that was found in the code."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from algolab.framework import Bot, ForexSymbol
from algolab.framework.strategies.base import Strategy
from algolab.hilega_milega import atr as hm_atr
from algolab.hilega_milega import ema, hilega_milega_rsi
from algolab.live import adapter as A
from algolab.live import framework_runner as FR
from algolab.live import positions
from algolab.ops import journal, risk
from algolab.ops import alerts
from algolab.ops import state as ops_state
import importlib
E = importlib.import_module("algolab.signals.engulfing")   # the package re-exports a function of the same name
from algolab.backtest import engine
from algolab.backtest.engine import Costs, Params, run_backtest

sys.path.insert(0, os.path.dirname(__file__))
from test_adapter_contract import fake_mt5  # noqa: E402


def ohlc(close, spread=0.001):
    close = np.asarray(close, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close},
                        index=pd.date_range("2024-01-01", periods=len(close), freq="4h"))


def random_walk(n, seed):
    rng = np.random.default_rng(seed)
    return 1.1 + np.cumsum(rng.normal(0, 0.003, n))


# ---------- C1 / C2: one HM definition, per bar, shared by live and backtest ----------

def reference_hm(closes, p, e):
    rsi = hilega_milega_rsi(list(closes), rsi_period=p)
    line = ema(rsi, e)
    out = []
    for r, m in zip(rsi, line):
        out.append(0 if (r is None or m is None) else (1 if r > m else (-1 if r < m else 0)))
    return np.array(out)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_hm_series_matches_the_reference_at_every_bar(seed):
    closes = random_walk(400, seed)
    assert np.array_equal(E.hm_direction_series(closes, 9, 3), reference_hm(closes, 9, 3))


def test_hm_series_is_not_a_constant_so_both_sides_can_trade():
    s = E.hm_direction_series(random_walk(600, 4), 9, 3)
    assert (s == 1).any() and (s == -1).any()                  # the old scalar gave one side only


def test_backtest_produces_sell_signals_when_the_market_falls():
    closes = np.r_[np.linspace(1.2, 1.0, 200), np.linspace(1.0, 1.2, 200)]
    df = ohlc(closes + np.random.default_rng(0).normal(0, 0.002, len(closes)))
    sig = engine.signals(df, Params())
    assert (sig == 1).any() and (sig == -1).any()


@pytest.mark.parametrize("seed", [5, 6])
def test_live_scanner_agrees_with_backtest_signal_on_every_closed_bar(seed):
    """The live scanner must decide each closed bar exactly as the backtest did for that bar."""
    df = ohlc(random_walk(260, seed))
    p = Params(rsi_n=9, ema_n=3)
    sig = engine.signals(df, p)
    cfg = {"hm_rsi_period": 9, "hm_ema_period": 3}
    checked = 0
    for k in range(60, len(df) - 1):
        hit = E.scan_frame("X", df.iloc[: k + 2], cfg)     # frame whose last CLOSED bar is k
        want = {1: "BUY", -1: "SELL"}.get(int(sig[k]))
        got = hit["side"] if hit else None
        assert got == want, (k, got, want)
        checked += 1
    assert checked > 100


def test_live_scanner_ignores_the_forming_bar():
    """Changing only the still-forming bar must not change today's signal (it was read before)."""
    df = ohlc(random_walk(120, 9))
    cfg = {"hm_rsi_period": 9, "hm_ema_period": 3}
    base = E.scan_frame("X", df, cfg)
    spiked = df.copy()
    spiked.iloc[-1, spiked.columns.get_loc("close")] += 0.5
    spiked.iloc[-1, spiked.columns.get_loc("high")] += 0.5
    assert E.scan_frame("X", spiked, cfg) == base


# ---------- H1: backtest fill realism ----------

ENTRY = 21          # signal on bar 20 -> entry at bar 21's open (ATR needs 14 bars of history first)


def flat_frame(n=40, price=1.0, spread=0.001):
    """Bars with a real range (so ATR is defined) and no drift (so only the modelled costs move the P&L)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame({"open": [price] * n, "high": [price + spread] * n,
                         "low": [price - spread] * n, "close": [price] * n}, index=idx)


def one_signal(monkeypatch, n=40):
    sig = np.zeros(n, dtype=int)
    sig[ENTRY - 1] = 1
    monkeypatch.setattr(engine, "signals", lambda d, p: sig)


def test_stop_inside_the_entry_bar_is_honoured(monkeypatch):
    df = flat_frame()
    df.iloc[ENTRY, df.columns.get_loc("low")] = 0.5             # entry bar crashes through the stop
    one_signal(monkeypatch)
    tr = run_backtest(df, Params(), Costs(risk_per_trade=0.01))
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "SL"
    assert tr.iloc[0]["exit_time"] == df.index[ENTRY]


def test_exit_pays_half_the_spread_on_the_way_out(monkeypatch):
    df = flat_frame(spread=0.001)                               # range for ATR; no excursion reaches stop or target
    one_signal(monkeypatch)
    c = Costs(spread=0.002, commission_per_lot=0.0, risk_per_trade=0.01)
    tr = run_backtest(df, Params(tp_mult=3.0), c)
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "EOD"
    # entry pays half a spread, exit pays half a spread: round trip = full spread
    assert tr.iloc[0]["exit"] == pytest.approx(1.0 - 0.001)
    assert tr.iloc[0]["entry"] == pytest.approx(1.0 + 0.001)


def test_gap_through_the_stop_fills_at_the_open_not_the_stop(monkeypatch):
    df = flat_frame(spread=0.001)
    g = ENTRY + 2                                               # a bar after entry gaps below the stop
    df.iloc[g, df.columns.get_loc("open")] = 0.80
    df.iloc[g, df.columns.get_loc("high")] = 0.80
    df.iloc[g, df.columns.get_loc("low")] = 0.79
    df.iloc[g, df.columns.get_loc("close")] = 0.80
    one_signal(monkeypatch)
    c = Costs(spread=0.0, commission_per_lot=0.0, risk_per_trade=0.01)
    tr = run_backtest(df, Params(), c)
    assert tr.iloc[0]["reason"] == "SL" and tr.iloc[0]["exit"] == pytest.approx(0.80)


def test_backtest_atr_matches_the_live_position_manager_atr():
    df = ohlc(random_walk(200, 11), spread=0.002)
    ours = engine.atr(df, 14).to_numpy()
    live = hm_atr(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period=14)
    live = np.array([np.nan if v is None else v for v in live])
    assert np.allclose(ours[13:], live[13:], equal_nan=True)


# ---------- H3 / H2: LLM budget wired, ops alert no longer crashes ----------

def test_orchestrator_calls_the_analyst_through_the_budget_cache():
    from algolab.agent.multi_agent import Orchestrator
    from algolab.llm.budget import CachedAnalyst
    o = Orchestrator({"ollama": {"url": "http://localhost:11434", "api_key": "", "model": "m"},
                      "risk": {"max_open_trades": 3}, "analyst": {"min_confidence": 0.6},
                      "scanner": {"htf": "H4"}, "dry_run": True})
    assert isinstance(o.analyst, CachedAnalyst)


def test_budget_cache_key_identifies_a_signal_by_symbol_side_and_bar():
    from algolab.llm.budget import CachedAnalyst
    a = {"symbol": "EURUSD", "side": "BUY", "time": "t1", "close": 1.1}
    b = {"symbol": "EURUSD", "side": "BUY", "time": "t1", "close": 1.2}   # same signal, other fields
    assert CachedAnalyst.key(a) == CachedAnalyst.key(b)
    assert CachedAnalyst.key(a) != CachedAnalyst.key({**a, "side": "SELL"})


def test_ops_cycle_alerts_without_a_rsi_field_and_marks_seen_only_after_delivery(monkeypatch):
    from algolab.agent import multi_agent
    from algolab.ops import pipeline as OP

    class FakeOrch:
        def __init__(self, p=None):
            pass

        def run(self, open_positions=0):
            take = [{"symbol": "EURUSD", "side": "BUY", "time": "t9", "close": 1.1,
                     "decision": "TAKE", "confidence": 0.9, "reason": "ok"}]
            return {"take": take, "reviewed": take}

    sent = []
    monkeypatch.setattr(OP, "Orchestrator", FakeOrch)
    monkeypatch.setattr(OP, "guard", lambda p, ops: (True, "ok", {"open_positions": 0}))
    monkeypatch.setattr(alerts, "send", lambda text, *a, **k: sent.append(text) or True)
    rep = OP.run_cycle()
    assert len(sent) == 1 and "EURUSD BUY" in sent[0]
    assert rep["take"][0]["alerted"] is True
    assert "EURUSD|BUY|t9" in ops_state.read_json("seen_signals.json", {})


def test_ops_cycle_retries_when_alert_delivery_fails(monkeypatch):
    from algolab.ops import pipeline as OP

    class FakeOrch:
        def __init__(self, p=None):
            pass

        def run(self, open_positions=0):
            take = [{"symbol": "GBPUSD", "side": "SELL", "time": "t2", "close": 1.3,
                     "decision": "TAKE", "confidence": 0.8, "reason": "x"}]
            return {"take": take, "reviewed": take}

    monkeypatch.setattr(OP, "Orchestrator", FakeOrch)
    monkeypatch.setattr(OP, "guard", lambda p, ops: (True, "ok", {}))
    monkeypatch.setattr(alerts, "send", lambda text, *a, **k: False)     # channel down
    OP.run_cycle()
    assert "GBPUSD|SELL|t2" not in ops_state.read_json("seen_signals.json", {})


# ---------- H4: broker P&L, never estimated ----------

def test_realized_pnl_sums_profit_commission_swap_and_fee():
    deals = [SimpleNamespace(entry=0, profit=0.0, commission=-1.0, swap=0.0, fee=0.0),
             SimpleNamespace(entry=1, profit=12.0, commission=-1.0, swap=-0.5, fee=0.0)]
    ad = A.Mt5Adapter()
    ad.mt5 = fake_mt5(history_deals_get=lambda *a, **k: deals)
    assert ad.realized_pnl(7) == pytest.approx(9.5)


def test_realized_pnl_is_none_until_a_closing_deal_exists():
    ad = A.Mt5Adapter()
    ad.mt5 = fake_mt5(history_deals_get=lambda *a, **k: [SimpleNamespace(entry=0, profit=0.0,
                                                                          commission=-1.0, swap=0.0, fee=0.0)])
    assert ad.realized_pnl(7) is None


def test_history_error_is_raised_not_treated_as_zero():
    ad = A.Mt5Adapter()
    ad.mt5 = fake_mt5(history_deals_get=lambda *a, **k: None, last_error=lambda: (1, "x"))
    with pytest.raises(A.AdapterError):
        ad.realized_pnl(7)


def test_positions_record_broker_pnl_on_a_stop_out():
    class Ad:
        def __init__(self):
            self.realized = {1: 42.5}

        def realized_pnl(self, ticket):
            return self.realized.get(int(ticket))

    ev = positions._event({"strategy": "ocl", "symbol": "EURUSD", "ticket": 1}, "SL", 1.09,
                          Ad().realized_pnl(1))
    assert ev["pnl"] == pytest.approx(42.5)


# ---------- H5 / H6: volume step, margin, filling mode ----------

def test_lots_are_snapped_down_to_the_volume_step():
    assert FR._snap_volume(0.37, {"volume_step": 0.1, "volume_max": 10}) == pytest.approx(0.3)
    assert FR._snap_volume(0.37, {"volume_step": 0.01, "volume_max": 0.2}) == pytest.approx(0.2)


class SizerAd:
    def __init__(self, margin, free, step=0.1):
        self._m, self._f, self._step = margin, free, step

    def symbol_info(self, sym):
        return {"contract_size": 100000, "volume_min": 0.1, "volume_step": self._step,
                "volume_max": 10.0, "digits": 5}

    def tick(self, sym):
        return {"bid": 1.1000, "ask": 1.1001}

    def account_currency(self):
        return "USD"

    def margin_for(self, sym, side, lots):
        return self._m * lots

    def free_margin(self):
        return self._f


def test_sizer_skips_when_margin_exceeds_the_share_of_free_margin():
    ad = SizerAd(margin=10_000.0, free=1_000.0)                 # enormous margin per lot
    sizer = FR.make_sizer(ad, 10000, {"risk_per_trade": 0.01, "max_lots": 5.0, "max_margin_use": 0.5})
    plan = sizer({"symbol": "EURUSD", "side": "BUY", "sl_dist": 0.001, "tp_dist": 0.003})
    assert "margin" in plan["skip"]


def test_sizer_returns_a_step_multiple_when_margin_fits():
    ad = SizerAd(margin=10.0, free=10_000.0)
    sizer = FR.make_sizer(ad, 10000, {"risk_per_trade": 0.01, "max_lots": 5.0, "max_margin_use": 0.5})
    plan = sizer({"symbol": "EURUSD", "side": "BUY", "sl_dist": 0.001, "tp_dist": 0.003})
    assert round(plan["lots"] / 0.1, 6) == round(plan["lots"] / 0.1)


@pytest.mark.parametrize("mode,expected", [(2, 1), (1, 0), (0, 2)])
def test_filling_mode_is_chosen_from_the_symbol(mode, expected):
    ad = A.Mt5Adapter()
    ad.mt5 = fake_mt5(symbol_info=lambda s: SimpleNamespace(filling_mode=mode, volume_step=0.01,
                                                            volume_min=0.01, volume_max=5.0, digits=5,
                                                            trade_contract_size=100000),
                      ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ORDER_FILLING_RETURN=2)
    assert ad.filling_type("EURUSD") == expected


def _mt5_for_orders(retcode, volume=None):
    return fake_mt5(
        symbol_info=lambda s: SimpleNamespace(filling_mode=2, volume_step=0.01, volume_min=0.01,
                                              volume_max=5.0, digits=5, trade_contract_size=100000),
        symbol_info_tick=lambda s: SimpleNamespace(bid=1.0, ask=1.0001),
        order_send=lambda req: SimpleNamespace(retcode=retcode, order=5, price=1.0001,
                                               volume=volume, comment=""),
        TRADE_RETCODE_DONE=10009, TRADE_RETCODE_DONE_PARTIAL=10010,
        ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ORDER_FILLING_RETURN=2)


def test_partial_fill_is_reported_with_the_volume_that_filled():
    ad = A.Mt5Adapter()
    ad.mt5 = _mt5_for_orders(10010, volume=0.3)
    out = ad.market_order("EURUSD", "BUY", 0.5, 0.99, 0, 1, 20)
    assert out["volume"] == pytest.approx(0.3) and out["partial"] is True


def test_no_order_result_is_raised_as_unknown_state():
    ad = A.Mt5Adapter()
    ad.mt5 = _mt5_for_orders(10009)
    ad.mt5.order_send = lambda req: None
    with pytest.raises(A.AdapterError, match="state unknown"):
        ad.market_order("EURUSD", "BUY", 0.5, 0.99, 0, 1, 20)


def test_partial_close_reports_the_volume_still_open():
    ad = A.Mt5Adapter()
    ad.mt5 = _mt5_for_orders(10010, volume=0.2)
    with pytest.raises(A.PartialClose) as e:
        ad.close_position(9, "EURUSD", 1, 0.5, 1, 20)
    assert e.value.remaining == pytest.approx(0.3)


def test_net_deposits_are_balance_deals_only():
    deals = [SimpleNamespace(type=2, profit=1000.0), SimpleNamespace(type=2, profit=-200.0),
             SimpleNamespace(type=0, profit=50.0)]
    ad = A.Mt5Adapter()
    ad.mt5 = fake_mt5(history_deals_get=lambda *a, **k: deals)
    assert ad.net_deposits() == pytest.approx(800.0)


# ---------- H7: kill switch measures trading, not deposits ----------

def test_a_withdrawal_does_not_look_like_a_drawdown(tmp_path):
    # equity 9000 after withdrawing 1000 of a 10000 account: trading is flat
    assert risk.adjusted_equity(9000.0, -1000.0) == pytest.approx(10000.0)
    limits = {"max_drawdown_kill": 0.15, "max_daily_loss": 0.03}
    ok, _, breach = risk.check_limits(risk.adjusted_equity(9000.0, -1000.0), limits)
    assert ok and not breach


def test_a_real_loss_still_breaches_the_drawdown_limit():
    limits = {"max_drawdown_kill": 0.15, "max_daily_loss": 0.03}
    risk.check_limits(10000.0, limits)
    ok, _, breach = risk.check_limits(risk.adjusted_equity(8000.0, 0.0), limits)
    assert breach


# ---------- C3 / H9: validation is bound to the exact settings and covers every enabled strategy ----------

def test_validation_gate_refuses_a_report_for_different_settings(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    report = tmp_path / "validation.json"
    monkeypatch.setattr(V, "REPORT", report)
    report.write_text(json.dumps({"ts": ops_state.now_utc().isoformat(), "passed": True,
                                  "reasons": ["ok"], "fingerprint": "not-the-current-one"}))
    ok, why = V.validation_gate()
    assert not ok and "settings changed" in why


def test_validation_gate_accepts_a_matching_passed_report(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    report = tmp_path / "validation.json"
    monkeypatch.setattr(V, "REPORT", report)
    report.write_text(json.dumps({"ts": ops_state.now_utc().isoformat(), "passed": True,
                                  "reasons": ["ok"], "fingerprint": V.config_fingerprint()}))
    assert V.validation_gate()[0] is True


def test_fingerprint_changes_when_an_enabled_strategy_changes(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    before = V.config_fingerprint()
    fw = json.loads((V.FRAMEWORK_CONFIG).read_text())
    fw["strategies"][0]["enabled"] = not fw["strategies"][0].get("enabled", True)
    monkeypatch.setattr(V, "FRAMEWORK_CONFIG", tmp_path / "framework.json")
    (tmp_path / "framework.json").write_text(json.dumps(fw))
    assert V.config_fingerprint() != before


def test_framework_replay_refuses_an_enabled_strategy_it_cannot_test(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    fw = {"enabled": True, "strategies": [{"type": "engulfing", "params": {}}]}
    path = tmp_path / "framework.json"
    path.write_text(json.dumps(fw))
    monkeypatch.setattr(V, "FRAMEWORK_CONFIG", path)
    res = V.framework_replay({})
    assert res["passed"] is False and "no replay validation" in res["reasons"][0]


def test_framework_replay_refuses_missing_real_data(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    fw = {"enabled": True, "strategies": [{"type": "ocl", "params": {}}]}
    path = tmp_path / "framework.json"
    path.write_text(json.dumps(fw))
    monkeypatch.setattr(V, "FRAMEWORK_CONFIG", path)
    monkeypatch.setattr(V, "ROOT", tmp_path)                  # no data folder at all
    monkeypatch.setattr(V, "load_profile_symbols", lambda: ["EURUSD"])
    res = V.framework_replay({})
    assert res["passed"] is False and "missing" in res["reasons"][0]


# ---------- H8: no unattended parameter change ----------

def test_candidate_is_not_applied_without_approval():
    from algolab.optimize.autotune import decide_apply
    assert decide_apply(better=True, exhausted=False, auto_apply=False, approve=False) is False


def test_candidate_applies_only_with_approval_or_explicit_auto_setting():
    from algolab.optimize.autotune import decide_apply
    assert decide_apply(True, False, False, True) is True
    assert decide_apply(True, False, True, False) is True
    assert decide_apply(True, True, True, True) is False      # holdout exhausted: never


def test_autotune_config_defaults_to_manual_approval():
    cfg = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "config", "autotune.json")).read())
    assert cfg["auto_apply"] is False


# ---------- H8: selection corrects for multiple comparisons and ignores unknown P&L ----------

def test_family_correction_gets_stricter_as_more_strategies_are_ranked():
    from algolab.optimize import selection as S
    assert S.family_z(1) < S.family_z(5) < S.family_z(20)


def test_selection_ignores_trades_without_broker_pnl(tmp_path, monkeypatch):
    from algolab.optimize import selection as S
    journal.record_open(source="framework", strategy="ocl", symbol="EURUSD", side="BUY", ticket=1,
                        lots=0.1, entry=1.0, sl=0.99, tp=None)
    journal.record_close({"strategy": "ocl", "symbol": "EURUSD", "reason": "SL", "price": 0.99,
                          "ticket": 1, "pnl": None})
    assert S._closes() == []


# ---------- C4: a fill is recorded at once, and an unrecorded fill is reported, never swallowed ----------

class TwoLots(Strategy):
    name = "two_lots"

    def signal(self, df):
        return {"side": "BUY", "lots": 0.1, "sl": 1.09, "tp": 1.13}


class BotAd:
    def __init__(self, partial_volume=None, fail_order=False):
        self.orders, self.partial_volume, self.fail_order = [], partial_volume, fail_order
        self.ticket = 100

    async def open_positions(self):
        return 0

    async def candles(self, symbol, tf_name, count):
        idx = pd.date_range("2024-01-01", periods=30, freq="5min")
        return pd.DataFrame({"open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1, "volume": 1}, index=idx)

    def market_order(self, symbol, side, lots, sl, tp, magic, deviation):
        if self.fail_order:
            raise A.AdapterError("order_send failed: retcode 10006")
        self.ticket += 1
        self.orders.append(lots)
        return {"ticket": self.ticket, "fill_price": 1.1001,
                "volume": self.partial_volume if self.partial_volume is not None else lots}


def test_fill_hook_runs_immediately_with_the_real_filled_volume():
    ad = BotAd(partial_volume=0.04)
    seen = []
    bot = Bot(adapter=ad, dry_run=False, max_open=5, on_fill=lambda r: seen.append(r))
    bot.add(TwoLots(ForexSymbol("EURUSD")))
    asyncio.run(bot.run_once())
    assert len(seen) == 1 and seen[0]["lots"] == pytest.approx(0.04)


def test_signal_key_is_persisted_before_the_order_is_sent():
    ad = BotAd()
    order_of_events = []
    orig = ad.market_order

    def logged(*a, **k):
        order_of_events.append("order")
        return orig(*a, **k)
    ad.market_order = logged
    bot = Bot(adapter=ad, dry_run=False, max_open=5,
              on_seen=lambda key: order_of_events.append("persist"))
    bot.add(TwoLots(ForexSymbol("EURUSD")))
    asyncio.run(bot.run_once())
    assert order_of_events[:2] == ["persist", "order"]


def test_unpersistable_signal_is_not_traded():
    ad = BotAd()

    def boom(key):
        raise OSError("disk full")
    bot = Bot(adapter=ad, dry_run=False, max_open=5, on_seen=boom)
    bot.add(TwoLots(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert ad.orders == [] and "not persisted" in res[0]["error"]


def test_a_recording_failure_is_reported_not_raised():
    ad = BotAd()

    def bad_fill(rec):
        raise OSError("state folder read-only")
    bot = Bot(adapter=ad, dry_run=False, max_open=5, on_fill=bad_fill)
    bot.add(TwoLots(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert res[0]["executed"] is True and "record_error" in res[0]


def test_one_failed_order_does_not_stop_the_other_signals():
    ad = BotAd(fail_order=True)
    bot = Bot(adapter=ad, dry_run=False, max_open=5)
    bot.add(TwoLots(ForexSymbol("EURUSD"))).add(TwoLots(ForexSymbol("GBPUSD")))
    res = asyncio.run(bot.run_once())
    assert len(res) == 2 and all("error" in r for r in res)


def test_signal_keys_survive_a_restart_through_the_persisted_store(isolated_state):
    bar = str(pd.date_range("2024-01-01", periods=30, freq="5min")[-2])   # the bar the bot keys on
    FR.write_json(FR.SEEN_STATE, {"keys": [f"two_lots|EURUSD|BUY|{bar}"]})
    seen = set(FR.read_json(FR.SEEN_STATE, {}).get("keys", []))
    bot = Bot(adapter=BotAd(), dry_run=False, max_open=5, seen=seen)
    bot.add(TwoLots(ForexSymbol("EURUSD")))
    asyncio.run(bot.run_once())                                 # same bar as before the restart
    assert bot.adapter.orders == []


# ---------- C4: orphaned broker positions are found and block new entries ----------

def test_reconcile_finds_broker_positions_nothing_tracks():
    class Ad:
        def position_tickets(self, magic):
            return {10, 11, 12}
    orphans = positions.reconcile(Ad(), {"10": {"ticket": 10}, "11": {"ticket": 11}}, magic=1)
    assert orphans == [12]


def test_a_block_reason_stops_all_entries():
    ad = BotAd()
    bot, _ = FR.build_bot(ad, {"dry_run": False}, [], 10000, FR.load_config() | {"enabled": True,
                                                                                     "strategies": []},
                          "USD", {}, block_reason="orphan position(s) [12]")
    bot.add(TwoLots(ForexSymbol("EURUSD")))
    res = asyncio.run(bot.run_once())
    assert ad.orders == [] and "orphan" in res[0]["why"]


def test_partial_close_keeps_the_position_and_reduces_its_volume(monkeypatch):
    import test_live_ocl as T
    ad = T.FakeAdapter()
    T.open_position(ad)
    ad.candles_by_tf = {"M5": T.ltf_closed_then_forming(T.ENTRY_BAR, (1.090, 1.091, 1.070, 1.077)),
                        "M15": T.htf_closed(1.09)}

    def partial(*a, **k):
        raise A.PartialClose("partial", remaining=0.2)
    ad.close_position = partial
    tracked = T.tracked_one()
    events = asyncio.run(positions.manage(ad, tracked, magic=20262002, deviation=20))
    assert events == []                                         # still open, nothing closed
    assert tracked["1"]["volume"] == pytest.approx(0.2)


# ---------- C5 / H5: the cycle lock and atomic writes ----------

def test_second_cycle_is_refused_while_the_first_holds_the_lock():
    with ops_state.cycle_lock("framework"):
        with pytest.raises(ops_state.LockBusy):
            with ops_state.cycle_lock("framework"):
                pass
    with ops_state.cycle_lock("framework"):                    # released afterwards
        pass


def test_a_stale_lock_from_a_crashed_run_is_removed(isolated_state):
    isolated_state.mkdir(parents=True, exist_ok=True)
    lock = isolated_state / "framework.lock"
    lock.write_text("999")
    old = ops_state.now_utc().timestamp() - 60 * 60
    os.utime(lock, (old, old))
    with ops_state.cycle_lock("framework"):
        pass


def test_atomic_write_leaves_no_temp_files(isolated_state):
    ops_state.atomic_write_text(isolated_state / "x.json", "{}")
    assert sorted(p.name for p in isolated_state.iterdir()) == ["x.json"]


# ---------- H2 / M6: alerts never crash on a missing package and only cool down after delivery ----------

def test_alerts_without_httpx_return_false_instead_of_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    assert alerts.send("hello", "WARNING", key="no-httpx", cooldown_min=60) is False


def test_cooldown_is_not_recorded_when_nothing_was_delivered(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    assert alerts.send("hello", "WARNING", key="undelivered", cooldown_min=60) is False
    assert "undelivered" not in ops_state.read_json("alerts.json", {})


# ---------- M1 / L: the framework and the optimizer write only validated, consistent settings ----------

def test_framework_and_backtest_share_one_risk_setting():
    fw = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "config", "framework.json")).read())
    bt = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "config", "backtest.json")).read())
    assert fw["risk_per_trade"] == bt["costs"]["risk_per_trade"]
    assert fw["max_lots"] == bt["costs"]["max_lots"]


def test_mcp_is_bound_to_loopback_on_the_vps_profile():
    p = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "config", "profiles", "vps.json")).read())
    assert p["mcp"]["host"] == "127.0.0.1"


# ---------- C3 / H9: the full validation path runs on real CSV files (no synthetic data) ----------

def _write_csvs(folder, symbols, bars=900, seed=3):
    folder.mkdir(parents=True, exist_ok=True)
    for k, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + k)
        close = 1.1 + np.cumsum(rng.normal(0, 0.0005, bars))
        open_ = np.r_[close[0], close[:-1]]
        idx = pd.date_range("2024-01-01", periods=bars, freq="5min")
        m5 = pd.DataFrame({"time": idx, "open": open_, "high": np.maximum(open_, close) + 0.0003,
                           "low": np.minimum(open_, close) - 0.0003, "close": close})
        m5.to_csv(folder / f"{sym}_M5.csv", index=False)
        m15 = m5.set_index("time").resample("15min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
        m15.to_csv(folder / f"{sym}_M15.csv", index=False)


def test_framework_replay_runs_on_real_csv_files_and_reports_trades(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    _write_csvs(tmp_path / "data", ["EURUSD"])
    fw = {"enabled": True, "strategies": [{"type": "scalp", "params": {"ema_fast": 8, "ema_slow": 21,
                                                                         "rsi_n": 14, "atr_n": 14,
                                                                         "sl_mult": 1.5, "tp_mult": 3.0,
                                                                         "rsi_low": 40, "rsi_high": 60}}]}
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "framework.json").write_text(json.dumps(fw))
    monkeypatch.setattr(V, "ROOT", tmp_path)
    monkeypatch.setattr(V, "FRAMEWORK_CONFIG", tmp_path / "config" / "framework.json")
    monkeypatch.setattr(V, "load_profile_symbols", lambda: ["EURUSD"])
    res = V.framework_replay({"framework_replay_bars": 900, "framework_min_trades": 10**6})
    assert str(res["data"]).startswith("CSV")         # real files were used, not synthetic data
    assert res["types"] == ["scalp"]
    assert res["passed"] is False                     # thresholds too strict on purpose: it must say why
    assert any("replay produced" in r for r in res["reasons"])


def test_run_validation_writes_a_report_bound_to_the_fingerprint(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    monkeypatch.setattr(V, "REPORT", tmp_path / "reports" / "validation.json")
    monkeypatch.setattr(V, "framework_replay", lambda cfg, keep_fraction=1.0: {"passed": True, "reasons": ["replay gates passed"],
                                                           "types": [], "trades": 0, "net": 0.0,
                                                           "profit_factor": None, "data": "CSV"})
    monkeypatch.setattr(V, "load_all", lambda htf: {"EURUSD": ohlc(random_walk(900, 21))})
    res = V.run_validation()
    saved = json.loads((tmp_path / "reports" / "validation.json").read_text())
    assert saved["fingerprint"] == V.config_fingerprint()
    assert res["passed"] == saved["passed"]


def test_validation_never_reads_the_sealed_tail(tmp_path, monkeypatch):
    from algolab.backtest import validate as V
    seen = {}
    monkeypatch.setattr(V, "REPORT", tmp_path / "validation.json")
    monkeypatch.setattr(V, "load_all", lambda htf: {"EURUSD": ohlc(random_walk(1000, 22))})

    def fake_replay(cfg, keep_fraction=1.0):
        seen["keep"] = keep_fraction
        return {"passed": True, "reasons": ["replay gates passed"], "types": [], "trades": 0,
                "net": 0.0, "profit_factor": None, "data": "CSV"}
    monkeypatch.setattr(V, "framework_replay", fake_replay)
    captured = {}
    orig_validate = V.validate

    def spy(dfs, p, c, cfg):
        captured["len"] = len(dfs["EURUSD"])
        return orig_validate(dfs, p, c, cfg)
    monkeypatch.setattr(V, "validate", spy)
    V.run_validation()
    keep = 1.0 - V.sealed_fraction()
    assert seen["keep"] == pytest.approx(keep)
    assert captured["len"] == int(1000 * keep)                  # the last 10% is not in the data
