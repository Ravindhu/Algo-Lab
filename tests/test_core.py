"""Runs anywhere (no MT5 needed): engine, metrics, validation gate and token budget on synthetic data."""
import numpy as np
import pandas as pd

from algolab.backtest.engine import Costs, Params, run_backtest, trade_metrics
from algolab.backtest.validate import validate
from algolab.llm.budget import CachedAnalyst, TokenBudget


def synthetic(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0006, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0006, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_backtest_runs_and_metrics_are_consistent():
    df = synthetic()
    tr = run_backtest(df, Params(), Costs())
    m = trade_metrics(tr, Costs().initial_balance)
    assert m["trades"] == len(tr)
    if len(tr):
        assert set(tr["reason"]).issubset({"SL", "TP", "EOD"})


def test_validation_gate_rejects_too_few_trades():
    cfg = {"folds": 4, "min_trades_total": 10_000, "min_profit_factor": 1.2,
           "min_positive_fold_ratio": 0.6, "max_age_days": 30}
    res = validate({"TEST": synthetic()}, Params(), Costs(), cfg)
    assert res["passed"] is False
    assert any("trades" in r for r in res["reasons"])


class _FakeAnalyst:
    calls = 0

    def run(self, cand):
        _FakeAnalyst.calls += 1
        return {**cand, "decision": "TAKE", "confidence": 0.9, "reason": "ok"}


def test_token_cache_avoids_second_call(tmp_path, monkeypatch):
    import algolab.ops.state as st
    monkeypatch.setattr(st, "STATE_DIR", tmp_path)
    monkeypatch.setattr("algolab.llm.budget.read_json", lambda n, d=None: st.read_json(n, d))
    monkeypatch.setattr("algolab.llm.budget.write_json", lambda n, o: st.write_json(n, o))
    monkeypatch.setattr(st, "write_json", lambda n, o: (tmp_path.mkdir(exist_ok=True),
                        (tmp_path / n).write_text(__import__("json").dumps(o))))
    monkeypatch.setattr(st, "read_json", lambda n, d=None: (
        __import__("json").loads((tmp_path / n).read_text()) if (tmp_path / n).exists() else d))
    cand = {"symbol": "EURUSD", "side": "BUY", "time": "2026-10-28 08:00", "rsi": 28.1}
    ca = CachedAnalyst(_FakeAnalyst(), TokenBudget(100_000))
    first = ca.run(cand)
    second = ca.run(cand)
    assert first["source"] == "llm" and second["source"] == "cache"
    assert _FakeAnalyst.calls == 1
