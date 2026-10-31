"""Discovery -> live wiring: the live `discovered` plugin must produce the SAME signal as the
discovery backtest for the same genome, and promotion must only write disabled, live-compatible entries."""
import json

import numpy as np
import pandas as pd
import pytest

from algolab import discovery_factory as df_mod
from algolab import discovery_promote as dp
from algolab.framework.strategies.discovered import DiscoveredStrategy
from algolab.live.adapter import AdapterError


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


def test_accepts_long_short_and_two_sided_genomes():
    """The replay trades both sides now, so every direction is promotable."""
    for d in (1, 0, -1):
        assert df_mod.live_compatible(_genome(dir_=d))[0]


@pytest.mark.parametrize("d", [1, -1, 0])
def test_replay_runs_discovered_entry_with_invariants(d, tmp_path, monkeypatch):
    """The framework replay (the validate gate's engine) must trade a discovered entry in each direction
    and keep every invariant. Synthetic data: this checks wiring, not performance. State goes to a
    temp folder, never the repo's state/ (the bot dedupes signals through it)."""
    import asyncio
    from algolab.framework.e2e import SPREADS, build_data, check_invariants
    from algolab.framework.replay import SimBroker, run_replay
    from algolab.ops import alerts, state as ops_state
    from algolab.profile import ROOT

    monkeypatch.setattr(ops_state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(alerts, "send", lambda *a, **k: True)
    g = _genome("hm_cross", dir_=d)
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
    if d == 1:
        assert all(t["side"] == 1 for t in sim.trades)
    if d == -1:
        assert all(t["side"] == -1 for t in sim.trades)
    # Tracked and classified: no "unmanaged position" block, and closes come back as framework events
    assert not any("unmanaged" in str(r.get("why", "")) for r in out["results"])
    reasons = {e["reason"] for e in out["events"]}
    assert reasons and reasons <= {"DISASTER", "TP", "BROKER_CLOSE"}



def test_simulator_short_fills_stops_targets_and_pnl():
    """Short side of the replay broker, same convention as the live adapter."""
    import asyncio
    from algolab.framework.replay import CONTRACT, SimBroker
    idx = pd.date_range("2024-01-01", periods=6, freq="5min")
    bars = pd.DataFrame({"open": [1.1000] * 6, "high": [1.1005] * 6, "low": [1.0995] * 6,
                         "close": [1.1000] * 6}, index=idx)
    sim = SimBroker({"EURUSD": {"M5": bars, "M15": bars}}, idx, {"EURUSD": 0.0002}, 10000.0, commission_per_lot=0.0)
    sim.set_index(1)
    bid = sim.tick("EURUSD")["bid"]
    res = sim.market_order("EURUSD", "SELL", 0.1, bid + 0.0050, bid - 0.0100, 1, 20)
    assert res["fill_price"] == pytest.approx(bid)
    assert sim.positions[res["ticket"]]["side"] == -1
    with pytest.raises(AdapterError):                     # SELL stop must sit above the fill
        sim.market_order("EURUSD", "SELL", 0.1, bid - 0.001, 0.0, 1, 20)
    # bar with high above the stop -> stop-out at the stop level, a loss
    stop_bars = bars.copy()
    stop_bars.iloc[2, stop_bars.columns.get_loc("high")] = bid + 0.0060
    sim.data["EURUSD"]["M5"] = stop_bars
    sim.intrabar(2)
    assert not sim.positions and sim.trades[-1]["reason"] == "DISASTER"
    assert sim.trades[-1]["pnl"] == pytest.approx(-0.0050 * 0.1 * CONTRACT)
    # target hit on a short -> a profit
    sim.set_index(3)
    res2 = sim.market_order("EURUSD", "SELL", 0.1, bid + 0.0050, bid - 0.0040, 1, 20)
    tgt_bars = bars.copy()
    tgt_bars.iloc[4, tgt_bars.columns.get_loc("low")] = bid - 0.0045
    sim.data["EURUSD"]["M5"] = tgt_bars
    sim.intrabar(4)
    assert sim.trades[-1]["reason"] == "TP" and sim.trades[-1]["pnl"] > 0
    assert res2["ticket"] not in sim.positions


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


def test_bank_rows_status_and_order(tmp_path):
    good, fib, other = _genome(), _genome(fam="engulfing_ocl"), _genome(fam="donchian")
    root = _write_root(tmp_path, [_cand(good), _cand(fib), _cand(other, sym="GBPUSD")])
    rows = dp.bank_rows(root)
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["status"] == "ready" and rows[0]["can_promote"]
    assert rows[1]["status"].startswith("not live") and not rows[1]["can_promote"]
    dp.promote(root, 1)
    rows = dp.bank_rows(root)
    assert rows[0]["status"] == "promoted (disabled)" and not rows[0]["can_promote"]
    assert rows[2]["symbol"] == "GBPUSD"
    assert dp.bank_rows(tmp_path / "nothing") == []
