"""End-to-end: the live framework code path over a simulated broker, with invariant checks.
Runs offline. Synthetic data: proves wiring and invariants, not edge."""
import asyncio
import json
from pathlib import Path

import pytest

from algolab.framework.e2e import run_e2e
from algolab.framework.prop import PROP_STATE, load_prop_config
from algolab.framework.replay import SimBroker, run_replay
from algolab.framework.e2e import build_data, SPREADS
from algolab.ops import alerts, state as ops_state


def test_e2e_run_passes_every_invariant(tmp_path):
    rep = run_e2e(["EURUSD", "USDJPY"], bars=1500, seed=7, out_dir=tmp_path)
    assert rep["passed"], [c for c in rep["checks"] if not c["ok"]]
    names = {c["name"] for c in rep["checks"]}
    assert "balance reconciles with trades" in names and "no position left open" in names
    assert (tmp_path / "e2e_report.json").exists()
    assert "SYNTHETIC" in rep["data"]


def test_e2e_without_prop_rules_still_passes(tmp_path):
    rep = run_e2e(["EURUSD"], bars=1200, seed=3, prop_enabled=False, out_dir=tmp_path)
    assert rep["passed"] and rep["profile"] is None and rep["prop_final"] is None


def test_injected_drawdown_fails_the_account_and_locks_out_new_orders(monkeypatch, tmp_path):
    monkeypatch.setattr(ops_state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(alerts, "send", lambda *a, **k: True)
    base = json.loads(Path("config/framework.json").read_text())
    cfg = {**base, "enabled": True, "strategies": [s for s in base["strategies"] if s["type"] == "ocl"]}
    per, idx, _ = build_data(["EURUSD", "USDJPY"], 3200, 7, None)
    sim = SimBroker(per, idx, {s: SPREADS[s] for s in ("EURUSD", "USDJPY")}, 10000)
    pc = {**load_prop_config(), "enabled": True}

    def outside_loss(sim, i):
        if i == 1000:
            sim.balance -= 2000          # equity falls to about 8000, below the 9000 floor

    out = asyncio.run(run_replay(sim, ["EURUSD", "USDJPY"], cfg, pc, 130, on_bar=outside_loss))
    failed_at = next((p["t"] for p in out["curve"] if p["status"] == "FAILED"), None)
    assert failed_at is not None, "the injected loss must fail the account"
    assert [o for o in sim.orders if o["time"] >= failed_at] == []
    assert out["curve"][-1]["status"] == "FAILED"
    assert not sim.positions                       # everything flattened or closed by the end
    saved = json.loads((tmp_path / "state" / PROP_STATE).read_text())
    assert saved["status"] == "FAILED" and saved["profile"] == "generic_2step_example"
