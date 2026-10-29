"""Plug-and-play strategy registry tests. No broker needed."""
import numpy as np
import pandas as pd
import pytest

from algolab.framework.strategies import registry
from algolab.framework.strategies.base import Strategy
from algolab.framework.symbol import ForexSymbol


def frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0003, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0003, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_builtin_strategies_are_loadable_by_type_name():
    for type_name in ("scalp_trend_pullback", "engulfing", "ocl_keylevel"):
        cls = registry.load(type_name)
        assert issubclass(cls, Strategy)
        assert cls.name == type_name


def test_example_plugin_loads_from_its_own_file():
    cls = registry.load("breakout_example")
    assert cls.name == "breakout_example"


def test_unknown_type_raises_with_available_names():
    with pytest.raises(KeyError):
        registry.load("does_not_exist")


def test_enabled_defaults_to_true_and_honours_false():
    assert registry.enabled({}) is True
    assert registry.enabled({"enabled": False}) is False


def test_example_strategy_contract_on_random_data():
    cls = registry.load("breakout_example")
    s = cls(ForexSymbol("EURUSD"))
    for seed in range(10):
        sig = s.signal(frame(seed=seed))
        if sig is None:
            continue
        assert sig["side"] in ("BUY", "SELL")
        assert sig["sl_dist"] > 0 and sig["tp_dist"] > sig["sl_dist"]
        assert "atr" in sig and "time" in sig


def test_example_strategy_signals_on_a_trend_frame():
    cls = registry.load("breakout_example")
    n = 60
    closes = np.r_[np.full(n - 1, 1.10), 1.12]     # last bar breaks the 20-bar high
    high = closes + 0.0005
    low = closes - 0.0005
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    df = pd.DataFrame({"open": closes, "high": high, "low": low, "close": closes}, index=idx)
    sig = cls(ForexSymbol("EURUSD")).signal(df)
    assert sig and sig["side"] == "BUY"
