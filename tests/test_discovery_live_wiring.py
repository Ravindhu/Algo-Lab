"""Discovery -> live wiring: the live `discovered` plugin must produce the SAME signal as the
discovery backtest for the same genome, and promotion must only write disabled, live-compatible entries."""
import json

import numpy as np
import pandas as pd
import pytest

from algolab import discovery_factory as df_mod
from algolab import discovery_promote as dp
from algolab.framework.strategies.discovered import DiscoveredStrategy


def _ohlc(n, seed):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0008, n))
    o = np.empty(n)
    o[0] = close[0]
    o[1:] = close[:-1] + rng.normal(0, 0.0003, n - 1)
    c = o + rng.normal(0, 0.0016, n) * rng.uniform(0.2, 1.8, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.0006, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.0006, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)


def _genome(fam="hm_cross", flt="none", dir_=1, sl=1.5, tp=3.0):
    rng = np.random.default_rng(3)
    g = df_mod.random_genome(rng, np.ones(len(df_mod.FAM_NAMES)) / len(df_mod.FAM_NAMES))
    g.update(fam=fam, p=df_mod.rand_params(df_mod.FAMILIES[fam]["space"], rng),
             flt=flt, fp=df_mod.rand_params(df_mod.FILTERS[flt]["space"], rng),
             sl=sl, tp=tp, hold=df_mod.SLTP_HOLD, dir=dir_)
    return df_mod.repair(g)


@pytest.fixture(scope="module")
def data():
    return _ohlc(1200, seed=5)


@pytest.mark.parametrize("fam", ["hm_cross", "ema_cross", "donchian", "rsi_reversion"])
def test_live_signal_equals_discovery_signal(data, fam):
    g = _genome(fam=fam)
    strat = DiscoveredStrategy("EURUSD", "M15", genome=g)
    fired = 0
    for k in range(350, 700):
        df = data.iloc[: k + 2]                       # framework df: last row is still forming
        res = strat.signal(df)
        want = int(df_mod.sig_and_fill_for_genome(df.iloc[:-1], g)[0][-1])
        got = {"BUY": 1, "SELL": -1}.get(res["side"], 0) if res else 0
        assert got == want, f"bar {k}"
        if res:
            fired += 1
            assert res["sl_dist"] == pytest.approx(g["sl"] * res["atr"])
            assert res["tp_dist"] == pytest.approx(g["tp"] * res["atr"])
    assert fired >= 1 or fam == "rsi_reversion"       # guard: most families must fire on this data


def test_only_symbol_gates_trading(data):
    g = _genome()
    other = DiscoveredStrategy("GBPUSD", "M15", genome=g, only_symbol="EURUSD")
    assert all(other.signal(data.iloc[: k + 2]) is None for k in range(350, 500))


def test_refuses_fib_family_and_wrong_hold():
    with pytest.raises(ValueError, match="not live-compatible"):
        DiscoveredStrategy("EURUSD", "M15", genome=_genome(fam="engulfing_ocl"))
    g = _genome()
    g["hold"] = 40
    with pytest.raises(ValueError, match="not live-compatible"):
        DiscoveredStrategy("EURUSD", "M15", genome=g)
    with pytest.raises(ValueError, match="genome is required"):
        DiscoveredStrategy("EURUSD", "M15")


def test_refuses_short_and_two_sided_genomes(tmp_path):
    """The replay simulator is BUY-only, so only long-only genomes may be promoted."""
    for d in (0, -1):
        ok, why = df_mod.live_compatible(_genome(dir_=d))
        assert not ok and "long-only" in why
        root = _write_root(tmp_path / f"d{d}", [_cand(_genome(dir_=d))])
        with pytest.raises(dp.PromoteError, match="cannot run live"):
            dp.promote(root, 1)
    assert df_mod.live_compatible(_genome(dir_=1))[0]


def test_replay_runs_discovered_entry_with_invariants():
    """The framework replay (the validate gate's engine) must trade a long-only discovered entry
    and keep every invariant. Synthetic data: this checks wiring, not performance."""
    import asyncio
    from algolab.framework.e2e import SPREADS, build_data, check_invariants
    from algolab.framework.replay import SimBroker, run_replay
    from algolab.profile import ROOT

    g = _genome("hm_cross", dir_=1)
    base = json.loads((ROOT / "config" / "framework.json").read_text())
    cfg = {**base, "enabled": True, "strategies": [
        {"type": "discovered", "enabled": True,
         "params": {"timeframe": "M15", "count": 400, "only_symbol": "EURUSD", "genome": g}}]}
    per, index, _ = build_data(["EURUSD"], 2500, 7, None, 1.0)
    sim = SimBroker(per, index, {"EURUSD": SPREADS.get("EURUSD", 0.0002)}, 10000.0)
    out = asyncio.run(run_replay(sim, ["EURUSD"], cfg, None, 130))
    assert len(sim.trades) > 0
    assert all(c["ok"] for c in check_invariants(sim, out["curve"], None, cfg))
    assert not [r for r in out["results"] if r.get("error")]


def test_hold_is_fixed_under_search():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = df_mod.random_genome(rng, np.ones(len(df_mod.FAM_NAMES)) / len(df_mod.FAM_NAMES))
        assert g["hold"] == df_mod.SLTP_HOLD
        m = df_mod.mutate(dict(g), rng, 1.0, np.ones(len(df_mod.FAM_NAMES)) / len(df_mod.FAM_NAMES))
        assert m["hold"] == df_mod.SLTP_HOLD


def _write_root(tmp_path, cands, applied=False):
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "discovery.json").write_text(json.dumps(
        {"generated_at": "2026-10-30T00:00:00+00:00", "applied": applied, "candidates": cands}))
    (tmp_path / "config" / "framework.json").write_text(json.dumps(
        {"enabled": False, "strategies": [{"type": "scalp", "params": {}}]}))
    return tmp_path


def _cand(g, sym="EURUSD", tf="M15"):
    return {"symbol": sym, "tf": tf, "score": 50.0, "trades": 120, "desc": "x", "genome": g,
            "test": {"pf": 1.4}}


def test_promote_writes_disabled_entry_and_backs_up(tmp_path):
    root = _write_root(tmp_path, [_cand(_genome())])
    entry = dp.promote(root, 1)
    fw = json.loads((root / "config" / "framework.json").read_text())
    assert fw["enabled"] is False                       # framework master switch untouched
    assert entry["enabled"] is False
    last = fw["strategies"][-1]
    assert last["type"] == "discovered" and last["enabled"] is False
    assert last["params"]["only_symbol"] == "EURUSD" and last["params"]["timeframe"] == "M15"
    assert (root / "config" / "framework.json.bak").exists()


def test_promote_refuses_duplicate_bad_rank_and_applied_report(tmp_path):
    g = _genome()
    root = _write_root(tmp_path, [_cand(g)])
    dp.promote(root, 1)
    with pytest.raises(dp.PromoteError, match="already"):
        dp.promote(root, 1)
    with pytest.raises(dp.PromoteError, match="between"):
        dp.promote(root, 5)
    root2 = _write_root(tmp_path / "b", [_cand(g)], applied=True)
    with pytest.raises(dp.PromoteError, match="applied"):
        dp.promote(root2, 1)


def test_promote_refuses_fib_candidate(tmp_path):
    root = _write_root(tmp_path, [_cand(_genome(fam="engulfing_ocl"))])
    with pytest.raises(dp.PromoteError, match="cannot run live"):
        dp.promote(root, 1)
