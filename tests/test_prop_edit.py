"""Dashboard prop-rule editor: validated, backed up, audited, and never silently wrong."""
import json
import shutil
from pathlib import Path

import pytest

from algolab.dashboard import prop_edit as pe
from algolab.framework.prop import PROP_STATE, PropError
from algolab.ops import state as st

REPO_PROP = Path(__file__).resolve().parents[1] / "config" / "prop.json"


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copy(REPO_PROP, tmp_path / "config" / "prop.json")
    monkeypatch.setattr(st, "STATE_DIR", tmp_path / "state")
    return tmp_path


def _rules(root):
    return json.loads((root / "config" / "prop.json").read_text())["profiles"]["generic_2step_example"]["rules"]


def test_get_active_lists_every_editable_rule(root):
    view = pe.get_active(root)
    assert view["profile"] == "generic_2step_example"
    assert {f["name"] for f in view["fields"]} == set(pe.TYPES)
    assert view["enabled"] is False and view["saved_state"] is None


def test_valid_save_applies_backs_up_and_audits(root):
    res = pe.save_rules(root, {"max_daily_loss_pct": "4.5", "close_before_weekend": "false", "max_lots": 0.5})
    assert res["changes"]["max_daily_loss_pct"] == {"from": 5, "to": 4.5}
    assert res["changes"]["close_before_weekend"]["to"] is False
    r = _rules(root)
    assert r["max_daily_loss_pct"] == 4.5 and r["close_before_weekend"] is False and r["max_lots"] == 0.5
    assert (root / "config" / "prop.json.bak").exists()
    audit = (root / "state" / "prop_changes.jsonl").read_text().strip().splitlines()
    assert len(audit) == 1 and json.loads(audit[0])["profile"] == "generic_2step_example"


def test_no_change_writes_nothing(root):
    before = (root / "config" / "prop.json").read_text()
    assert pe.save_rules(root, {"max_lots": 1.0})["changes"] == {}
    assert (root / "config" / "prop.json").read_text() == before
    assert not (root / "state").exists()


@pytest.mark.parametrize("updates", [
    {"max_daily_loss_pct": 0},                       # must be > 0
    {"max_total_loss_pct": 150},                     # must be <= 100
    {"safety_buffer_pct": 6},                        # must be below both loss limits
    {"max_open_positions": 0},                       # must be >= 1
    {"weekend_close_hour_utc": 24},                  # must be 0..23
    {"min_trading_days": 1.5},                       # whole number only
    {"trailing_drawdown": "maybe"},                  # not a boolean
    {"max_daily_loss_pct": "abc"},                   # not a number
    {"max_daily_loss_pct": "nan"},                   # not finite
    {"enabled": True},                               # not editable here
    {"starting_balance": -1},                        # must be > 0
])
def test_invalid_values_refused_and_file_unchanged(root, updates):
    before = (root / "config" / "prop.json").read_text()
    with pytest.raises(PropError):
        pe.save_rules(root, updates)
    assert (root / "config" / "prop.json").read_text() == before
    assert not (root / "config" / "prop.json.bak").exists()


def test_starting_balance_locked_while_state_saved(root):
    st.write_json(PROP_STATE, {"profile": "generic_2step_example", "status": "ACTIVE"})
    with pytest.raises(PropError, match="starting_balance"):
        pe.save_rules(root, {"starting_balance": 20000})
    assert pe.save_rules(root, {"profit_target_pct": 8})["changes"]   # other rules still editable


def test_starting_balance_editable_with_no_state(root):
    assert pe.save_rules(root, {"starting_balance": 20000})["changes"]["starting_balance"]["to"] == 20000
