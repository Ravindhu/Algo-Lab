"""Start/stop control for the dashboard: it must use the same kill switch the bot obeys, and nothing else."""
import json

import pytest

from algolab.dashboard import control
from algolab.ops import state as st


@pytest.fixture
def sdir(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "STATE_DIR", tmp_path)
    return tmp_path


def test_stop_engages_kill_switch(sdir):
    assert control.algo_status() == {"trading_enabled": True, "kill": False}
    out = control.stop_algo("test")
    assert (sdir / "KILL").exists()
    assert out == {"trading_enabled": False, "kill": True}


def test_start_removes_kill_switch(sdir):
    control.stop_algo("test")
    out = control.start_algo()
    assert not (sdir / "KILL").exists()
    assert out == {"trading_enabled": True, "kill": False}


def test_start_when_not_stopped_is_harmless(sdir):
    assert control.start_algo() == {"trading_enabled": True, "kill": False}


def test_stop_then_start_round_trip_is_idempotent(sdir):
    control.stop_algo("a")
    control.stop_algo("b")                       # stopping twice is fine
    assert control.algo_status()["kill"] is True
    control.start_algo()
    control.start_algo()
    assert control.algo_status()["kill"] is False


def test_snapshot_reads_state_and_tolerates_missing_files(sdir):
    snap = control.snapshot()                    # nothing written yet: no errors, empty values
    assert snap["live"] == {} and snap["framework"] == {} and snap["trading_enabled"] is True
    (sdir / "live_last_cycle.json").write_text(json.dumps({"equity": 100000.0, "may_trade": True}))
    assert control.snapshot()["live"]["equity"] == 100000.0


def test_control_never_touches_dry_run_or_config(sdir, tmp_path):
    # Only the KILL file is created or removed: no config, no order path.
    control.stop_algo("x")
    control.start_algo()
    assert sorted(p.name for p in sdir.iterdir()) == []
