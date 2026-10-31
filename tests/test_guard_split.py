"""Fixes 1 and 7: a blocked guard stops NEW entries but never stops position management, and the drawdown
baseline keeps advancing while trading is blocked."""
import json
from datetime import datetime, timezone

import pytest

from algolab.framework.bot import Bot
from algolab.live import framework_runner, pipeline
from algolab.live.adapter import AccountState
from algolab.ops import state as st

RISK = {"max_drawdown_kill": 0.20, "max_daily_loss": 0.05}
NOW = datetime(2027, 1, 15, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "STATE_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(pipeline.alerts, "send", lambda *a, **k: sent.append(a))
    return tmp_path


class FakeAdapter:
    """Only the calls run_framework makes when there are no strategies."""

    def account_currency(self):
        return "USD"

    def positions_by_magic(self, magic):
        return []

    def position_tickets(self, magic):
        return set()


def _acct(equity, open_positions=0):
    return AccountState(equity=equity, balance=equity, open_positions=open_positions, net_deposits=0.0)


def test_blocked_entries_still_run_position_management(isolated, monkeypatch):
    calls = {"manage": 0, "run_once": 0}

    async def fake_manage(ad, tracked, **kw):
        calls["manage"] += 1
        return []

    async def fake_run_once(self):
        calls["run_once"] += 1
        return []

    monkeypatch.setattr(framework_runner.positions, "manage", fake_manage)
    monkeypatch.setattr(Bot, "run_once", fake_run_once)
    cfg = {"enabled": True, "magic": 1, "deviation_points": 20, "timeframe": "M5", "max_open": 3,
           "max_spread_atr": 0.25, "risk_per_trade": 0.005, "max_lots": 0.5, "strategies": []}

    import asyncio
    out = asyncio.run(framework_runner.run_framework(
        FakeAdapter(), {"dry_run": True}, ["EURUSD"], 100000.0, now=NOW, cfg=cfg,
        prop_cfg={"enabled": False}, allow_entries=False, entry_block_reason="kill switch active"))

    assert calls["manage"] == 1          # exits still managed during a halt
    assert calls["run_once"] == 0        # but no new entries
    assert out["entries_blocked"] == "kill switch active"
    assert out["results"] == []


def test_allowed_entries_run_the_bot(isolated, monkeypatch):
    calls = {"run_once": 0}

    async def fake_run_once(self):
        calls["run_once"] += 1
        return []

    monkeypatch.setattr(framework_runner.positions, "manage", lambda *a, **k: _async([]))
    monkeypatch.setattr(Bot, "run_once", fake_run_once)
    cfg = {"enabled": True, "magic": 1, "deviation_points": 20, "timeframe": "M5", "max_open": 3,
           "max_spread_atr": 0.25, "risk_per_trade": 0.005, "max_lots": 0.5, "strategies": []}

    import asyncio
    out = asyncio.run(framework_runner.run_framework(
        FakeAdapter(), {"dry_run": True}, ["EURUSD"], 100000.0, now=NOW, cfg=cfg,
        prop_cfg={"enabled": False}))
    assert calls["run_once"] == 1
    assert out["entries_blocked"] is None


async def _async(value):
    return value


def test_validation_failure_still_advances_drawdown_baseline(isolated, monkeypatch):
    """Before the fix, validation failed first and check_limits never saw the new peak."""
    monkeypatch.setattr(pipeline, "validation_gate", lambda: (False, "strategy not validated"))
    ops = {"risk": RISK}

    ok1, why1 = pipeline.guard(_acct(100000.0), ops)
    ok2, why2 = pipeline.guard(_acct(120000.0), ops)

    assert (ok1, why1) == (False, "strategy not validated")
    assert (ok2, why2) == (False, "strategy not validated")
    risk = json.loads((isolated / "risk.json").read_text())
    assert risk["peak"] == 120000.0                # advanced even while blocked
    assert not (isolated / "KILL").exists()        # a validation block is not a kill


def test_max_open_blocks_entries_without_engaging_kill(isolated, monkeypatch):
    monkeypatch.setattr(pipeline, "validation_gate", lambda: (True, "validated"))
    ok, why = pipeline.guard(_acct(100000.0, open_positions=pipeline.LIVE["max_open_trades"]), {"risk": RISK})
    assert (ok, why) == (False, "max open trades reached")
    assert not (isolated / "KILL").exists()


def test_drawdown_breach_engages_kill_once(isolated, monkeypatch):
    monkeypatch.setattr(pipeline, "validation_gate", lambda: (True, "validated"))
    (isolated / "risk.json").write_text(json.dumps({"peak": 100000.0, "day": "1900-01-01", "day_start": 100000.0}))
    ops = {"risk": RISK}

    ok, why = pipeline.guard(_acct(70000.0), ops)          # 30% below peak, limit 20%
    assert ok is False and "drawdown" in why
    assert (isolated / "KILL").exists()

    assert pipeline.guard(_acct(70000.0), ops) == (False, "kill switch active")  # no repeat breach alert
