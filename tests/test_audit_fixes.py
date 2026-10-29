"""Regression tests for the audit fixes: cost model in pips, currency-aware sizing, trade journal and
entry gate, shared risk limits, sealed split and holdout ledger, framework drift and sync."""
import json

import numpy as np
import pandas as pd
import pytest

from algolab.framework.protect import Protection, Trade
from algolab.live import sizing
from algolab.live.adapter import AdapterError
from algolab.ops import journal, risk
from algolab.optimize import autotune as AT


@pytest.fixture
def state(tmp_path, monkeypatch):
    import algolab.ops.state as st
    monkeypatch.setattr(st, "STATE_DIR", tmp_path)
    monkeypatch.setattr(journal, "STATE_DIR", tmp_path)
    monkeypatch.setattr(AT, "STATE_DIR", tmp_path)
    return tmp_path


# 1. cost model in pips
def test_breakeven_floor_uses_pips_not_percent():
    p = Protection(round_trip_cost_pips=1.5)
    t = Trade.open(entry=1.0800, trigger=1.0820, htf_sl=1.0700, broker_sl=1.0650, tp=None, p=p,
                   pip=0.0001)
    assert t.be_floor == pytest.approx(1.0820 + 0.00015)     # 1.5 pips, not 0.2% (~22 pips)


def test_pips_cost_requires_pip_size():
    with pytest.raises(ValueError):
        Trade.open(entry=1.08, trigger=1.082, htf_sl=1.07, broker_sl=1.06, tp=None,
                   p=Protection(round_trip_cost_pips=1.5))


def test_legacy_percent_cost_is_unchanged_without_pips():
    t = Trade.open(entry=1.0800, trigger=1.0820, htf_sl=1.0700, broker_sl=1.0650, tp=None, p=Protection())
    assert t.be_floor == pytest.approx(1.0820 * 1.002)


# 2. currency-aware sizing
class FakeAd:
    def __init__(self, ticks):
        self.ticks = ticks

    def tick(self, sym):
        if sym not in self.ticks:
            raise AdapterError("no such pair")
        return {"bid": self.ticks[sym], "ask": self.ticks[sym]}


def test_jpy_pair_converted_to_usd():
    rate = sizing.quote_to_account_rate(FakeAd({"USDJPY": 150.0}), "USDJPY", "USD")
    assert rate == pytest.approx(1 / 150.0)


def test_same_quote_currency_needs_no_conversion():
    assert sizing.quote_to_account_rate(FakeAd({}), "EURUSD", "USD") == 1.0


def test_missing_conversion_pair_returns_none():
    assert sizing.quote_to_account_rate(FakeAd({}), "EURGBP", "USD") is None


def test_lots_respect_risk_budget_for_jpy_pair():
    # $100 risk, 0.15 yen stop, 100k contract, USD/JPY 150 -> risk per lot = $100 -> 1.0 lot
    lots = sizing.lots_from_risk(100.0, 0.15, 100_000, 1 / 150.0, max_lots=5.0)
    assert lots == pytest.approx(1.0)


def test_lots_capped_and_zero_when_no_risk():
    assert sizing.lots_from_risk(1000.0, 0.0001, 100_000, 1.0, max_lots=2.0) == 2.0
    assert sizing.lots_from_risk(100.0, 0.0, 100_000, 1.0, max_lots=2.0) == 0.0


# 3. journal and entry gate
def test_journal_open_then_close_computes_signed_pnl(state):
    journal.record_open(source="framework", strategy="ocl", symbol="EURUSD", side="BUY", ticket=7,
                        lots=0.1, entry=1.10, sl=1.09, tp=None)
    ev = journal.record_close({"strategy": "ocl", "symbol": "EURUSD", "reason": "TRAIL",
                               "price": 1.12, "ticket": 7})
    assert ev["pnl"] == pytest.approx(0.02)
    assert ev["r"] == pytest.approx(2.0)                       # risk was 0.01


def test_close_without_open_is_ignored(state):
    assert journal.record_close({"strategy": "ocl", "symbol": "EURUSD", "reason": "SL",
                                 "price": 1.0, "ticket": 99}) is None


def test_gate_warm_up_then_blocks_then_checks_win_rate(state):
    cfg = {"enabled": True, "min_trades": 3, "warmup_entries": 2, "window": 10, "min_lower_bound": 0.25}
    ok, why = journal.entry_gate(cfg)
    assert ok and "warm-up" in why                              # no closes yet: bootstrap allowed
    for t in (1, 2):
        journal.record_open(source="framework", strategy="ocl", symbol="EURUSD", side="BUY", ticket=t,
                            lots=0.1, entry=1.0, sl=0.99, tp=None)
    ok, why = journal.entry_gate(cfg)
    assert not ok and "need 3" in why                           # warm-up used up, not enough closes
    for t in (1, 2):
        journal.record_close({"strategy": "ocl", "symbol": "EURUSD", "reason": "TRAIL",
                              "price": 1.02, "ticket": t})
    journal.record_open(source="framework", strategy="ocl", symbol="EURUSD", side="BUY", ticket=3,
                        lots=0.1, entry=1.0, sl=0.99, tp=None)
    journal.record_close({"strategy": "ocl", "symbol": "EURUSD", "reason": "TRAIL",
                          "price": 1.02, "ticket": 3})
    ok, why = journal.entry_gate(cfg)
    assert ok and "lower bound" in why                          # three wins clear the 0.25 bound


def test_gate_blocks_a_losing_record(state):
    cfg = {"enabled": True, "min_trades": 3, "warmup_entries": 0, "window": 10, "min_lower_bound": 0.25}
    for t in (1, 2, 3):
        journal.record_open(source="framework", strategy="ocl", symbol="EURUSD", side="BUY", ticket=t,
                            lots=0.1, entry=1.0, sl=0.99, tp=None)
        journal.record_close({"strategy": "ocl", "symbol": "EURUSD", "reason": "SL",
                              "price": 0.995, "ticket": t})
    ok, why = journal.entry_gate(cfg)
    assert not ok and "lower bound" in why


def test_gate_disabled_allows_everything(state):
    assert journal.entry_gate({"enabled": False})[0] is True
    assert journal.entry_gate(None)[0] is True


def test_torn_journal_line_does_not_break_reader(state):
    journal.record_open(source="legacy", strategy="x", symbol="EURUSD", side="BUY", ticket=1,
                        lots=0.1, entry=1.0, sl=0.9, tp=None)
    with (state / journal.FILE).open("a") as fh:
        fh.write('{"kind": "close", "trunc')
    assert len(journal.read()) == 1


# 4. shared risk limits
LIMITS = {"max_drawdown_kill": 0.10, "max_daily_loss": 0.03}


def test_drawdown_breach_is_reported(state):
    risk.check_limits(10_000, LIMITS)                           # sets the peak
    ok, why, breach = risk.check_limits(8_900, LIMITS)
    assert not ok and breach and "drawdown" in why


def test_daily_loss_pauses_without_kill(state):
    risk.check_limits(10_000, LIMITS)
    ok, why, breach = risk.check_limits(9_650, LIMITS)          # -3.5% today, drawdown 3.5% < 10%
    assert not ok and not breach and "daily loss" in why


def test_within_limits_is_ok(state):
    ok, why, breach = risk.check_limits(10_000, LIMITS)
    assert ok and not breach


# 5. sealed split and holdout ledger
def synth(n=1000):
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    close = 1.1 + np.cumsum(np.random.default_rng(1).normal(0, 0.001, n))
    return pd.DataFrame({"open": close, "high": close + 0.001, "low": close - 0.001,
                         "close": close}, index=idx)


def test_sealed_slice_is_disjoint_from_search_and_holdout():
    df = synth()
    search, hold, hold_start, sealed, sealed_start = AT._split({"EURUSD": df}, 0.7, 0.1)
    assert search["EURUSD"].index[-1] < hold_start["EURUSD"]
    assert sealed_start["EURUSD"] > hold_start["EURUSD"]
    assert sealed["EURUSD"].index[-1] == df.index[-1]
    assert sealed["EURUSD"].index[0] < sealed_start["EURUSD"]     # warm-up bars only, before the cut


def test_holdout_ledger_counts_per_window(state):
    key = "EURUSD=2024-06-01T00:00:00"
    assert AT._ledger_uses(key) == 0
    assert AT._ledger_bump(key) == 1
    assert AT._ledger_bump(key) == 2
    assert AT._ledger_uses(key) == 2


# 6. framework drift and sync (on a temp copy of framework.json)
def test_sync_writes_shared_keys_and_drift_clears(tmp_path, monkeypatch):
    fw = tmp_path / "framework.json"
    fw.write_text(json.dumps({"strategies": [
        {"type": "scalp", "params": {"rsi_n": 9, "atr_n": 9, "sl_mult": 1.0, "tp_mult": 1.5, "ema_fast": 8}},
        {"type": "ocl", "params": {"oversold": 30.0}}]}))
    monkeypatch.setattr(AT, "FW_CONFIG", fw)
    monkeypatch.setattr(AT.S, "load", lambda: {"rsi_n": 14, "atr_n": 14, "sl_mult": 1.5, "tp_mult": 3.0,
                                              "oversold": 30.0, "overbought": 70.0, "ml_threshold": 0.0})
    assert AT.framework_drift()                                  # differs before sync
    old = AT._sync_framework({"rsi_n": 14, "atr_n": 14, "sl_mult": 1.5, "tp_mult": 3.0})
    assert json.loads(fw.read_text())["strategies"][0]["params"]["sl_mult"] == 1.5
    assert json.loads(fw.read_text())["strategies"][0]["params"]["ema_fast"] == 8   # other keys kept
    assert b"rsi_n" in old
    assert AT.framework_drift() == []
