"""Runs anywhere (no MT5, no Optuna needed): features, ML dataset, strategy file, tuner helpers."""
import json

import numpy as np
import pandas as pd
import pytest

from algolab import strategy as S
from algolab.backtest.engine import Costs, Params, run_backtest
from algolab.ml import features as F
from algolab.ml import model as M


def synthetic(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0006, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0006, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_features_have_no_lookahead():
    """Value at bar t must not change when later bars are added."""
    df = synthetic(600)
    full = F.build(df)
    t = 400
    prefix = F.build(df.iloc[:t + 1])
    a = full.iloc[t][F.FEATURES].to_numpy(dtype=float)
    b = prefix.iloc[t][F.FEATURES].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_trades_carry_signal_time_before_entry():
    tr = run_backtest(synthetic(), Params(), Costs())
    if not tr.empty:
        assert (tr["signal_time"] < tr["entry_time"]).all()


def test_dataset_aligns_features_to_trades():
    df = synthetic()
    tr = run_backtest(df, Params(), Costs()).assign(symbol="TEST")
    if tr.empty:
        pytest.skip("no trades on synthetic data")
    X, y = M.dataset(tr, {"TEST": F.build(df)})
    assert set(X.index).issubset(set(tr.index))
    assert list(X.columns) == F.FEATURES
    assert len(y) == len(X)


def test_strategy_write_and_restore(tmp_path, monkeypatch):
    cfg = tmp_path / "strategy.json"
    cfg.write_text(json.dumps({"rsi_n": 14, "oversold": 30.0, "overbought": 70.0, "atr_n": 14,
                               "sl_mult": 1.5, "tp_mult": 3.0, "ml_threshold": 0.0,
                               "version": 1, "updated": None, "reason": "test"}))
    monkeypatch.setattr(S, "FILE", cfg)
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    backup = S.write({"rsi_n": 12, "oversold": 28.0, "overbought": 72.0, "atr_n": 10,
                      "sl_mult": 1.2, "tp_mult": 2.5, "ml_threshold": 0.55}, reason="test")
    assert S.load()["version"] == 2
    assert S.params().rsi_n == 12
    S.restore(backup)
    assert S.load()["rsi_n"] == 14 and S.load()["version"] == 1


def test_better_rule(monkeypatch):
    from algolab.optimize import autotune as A
    cfg = {"min_improvement_pct": 10}
    assert A._better({"expectancy": 12.0}, {"expectancy": 10.0}, cfg)
    assert not A._better({"expectancy": 10.5}, {"expectancy": 10.0}, cfg)
    assert A._better({"expectancy": 1.0}, {"expectancy": -2.0}, cfg)


def test_gate_rejects_small_holdout():
    from algolab.optimize import autotune as A
    ok, reasons = A._gate({"trades": 5, "expectancy": 3.0, "profit_factor": 2.0},
                          {"min_holdout_trades": 60, "min_profit_factor": 1.2})
    assert not ok and any("holdout trades" in r for r in reasons)
