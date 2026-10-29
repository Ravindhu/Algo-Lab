"""Agent memory, Monte Carlo simulation, win-rate gate and the Anthropic feasibility check."""
import numpy as np
import pytest

from algolab.framework.montecarlo import entry_allowed, simulate, win_lower_bound
from algolab.memory import store
from algolab.ops import state as ops_state


@pytest.fixture
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ops_state, "STATE_DIR", tmp_path)
    return tmp_path


# ---- memory -------------------------------------------------------------

def test_memory_round_trip_keeps_order_and_timestamps(state_dir):
    store.remember("mistake", {"what": "traded into news", "rule": "skip 30 min around high-impact news"})
    store.remember("mistake", {"what": "oversized after loss", "rule": "risk stays at 1% after a loss"})
    rows = store.recall("mistake")
    assert [r["what"] for r in rows] == ["traded into news", "oversized after loss"]
    assert all("ts" in r for r in rows)


def test_memory_recall_limit_returns_most_recent(state_dir):
    for i in range(5):
        store.remember("working", {"n": i})
    assert [r["n"] for r in store.recall("working", limit=2)] == [3, 4]


def test_memory_empty_when_nothing_written(state_dir):
    assert store.recall("calibration") == []


def test_memory_rejects_unknown_kind_and_empty_record(state_dir):
    with pytest.raises(store.MemoryStoreError):
        store.remember("opinions", {"x": 1})
    with pytest.raises(store.MemoryStoreError):
        store.remember("working", {})


def test_calibration_gap_shows_overconfidence(state_dir):
    rows = [{"predicted": 0.8, "won": False}, {"predicted": 0.8, "won": True},
            {"predicted": 0.8, "won": False}, {"predicted": 0.8, "won": False}]
    gap = store.calibration_gap(rows)
    assert gap["n"] == 4
    assert gap["gap"] == pytest.approx(0.8 - 0.25)
    assert store.calibration_gap([{"x": 1}]) is None


# ---- Monte Carlo --------------------------------------------------------

def test_simulate_runs_ten_thousand_and_is_reproducible():
    pnls = [50, -20, 30, -20, 40, -20, 60, -20]
    a = simulate(pnls, 10_000, runs=10_000, seed=1)
    b = simulate(pnls, 10_000, runs=10_000, seed=1)
    assert a["runs"] == 10_000 and a["trades"] == 8
    assert a == b


def test_positive_edge_gives_high_probability_of_profit():
    rng = np.random.default_rng(0)
    pnls = np.where(rng.random(300) < 0.6, 100.0, -100.0)     # 60% wins, 1:1
    rep = simulate(pnls, 10_000, runs=2000, seed=2)
    # 300 trades at a 60% win rate: about 95% of resamples end in profit. Assert a looser bound on purpose.
    assert rep["p_profit"] > 0.9
    assert rep["final_equity"]["p5"] <= rep["final_equity"]["median"] <= rep["final_equity"]["p95"]


def test_losing_edge_gives_low_probability_of_profit_and_some_ruin():
    rng = np.random.default_rng(0)
    pnls = np.where(rng.random(300) < 0.4, 100.0, -100.0)     # 40% wins, 1:1: negative expectancy
    rep = simulate(pnls, 1_000, runs=2000, seed=3, ruin_pct=0.3)
    assert rep["p_profit"] < 0.05
    assert rep["p_ruin"]["probability"] > 0


def test_drawdown_limit_probability_is_reported():
    rep = simulate([100, -100, -100, 100], 1000, runs=500, seed=4, dd_limit_pct=5)
    assert "p_drawdown_over_limit" in rep and 0 <= rep["p_drawdown_over_limit"]["probability"] <= 1


def test_simulate_rejects_bad_input():
    with pytest.raises(ValueError):
        simulate([], 1000)
    with pytest.raises(ValueError):
        simulate([1.0, float("nan")], 1000)


def test_win_lower_bound_is_below_point_estimate_and_zero_for_no_trades():
    assert win_lower_bound(0, 0) == 0.0
    lb = win_lower_bound(60, 100)
    assert 0 < lb < 0.6
    assert win_lower_bound(600, 1000) > win_lower_bound(60, 100)   # more evidence, tighter bound


def test_entry_gate_needs_enough_trades_and_a_strong_lower_bound():
    ok, why = entry_allowed([1, -1] * 10)                 # 20 trades < 30
    assert not ok and "need 30" in why
    ok, _ = entry_allowed([1] * 40)                       # 100% wins, bound well above 0.5
    assert ok
    ok, why = entry_allowed([1, -1] * 20)                 # 50% wins, bound below 0.5
    assert not ok and "lower bound" in why


# ---- feasibility check --------------------------------------------------

def test_anthropic_check_never_prints_the_key(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRET-VALUE-123")
    spec = importlib.util.spec_from_file_location(
        "check_anthropic", Path(__file__).resolve().parents[1] / "scripts" / "check_anthropic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check_key()
    out = capsys.readouterr().out
    assert "SECRET-VALUE" not in out and "length" in out
