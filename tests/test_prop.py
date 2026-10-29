"""Prop-firm rules: limits, status transitions, entry gate, weekends, profiles, operator commands."""
import json
from pathlib import Path

import pytest

from algolab.framework import prop as P
from algolab.framework.prop import (ACTIVE, FAILED, PASSED, PropError, PropGuard, PropRules, PropState,
                                    guard_from_config, select_profile)

RULES = dict(starting_balance=10000, profit_target_pct=10, max_daily_loss_pct=5, max_total_loss_pct=10,
             trailing_drawdown=False, min_trading_days=0, max_risk_per_trade_pct=1.0, max_open_positions=1,
             max_lots=1.0, close_before_weekend=True, weekend_close_hour_utc=20, day_reset_hour_utc=0,
             safety_buffer_pct=0.5)
MON_10 = "2026-01-05 10:00"          # Monday
TUE_10 = "2026-01-06 10:00"


def guard(**kw):
    return PropGuard(PropRules(**{**RULES, **kw}), profile="test")


# ---------- rule validation ----------

def test_valid_rules_load():
    assert PropRules(**RULES).starting_balance == 10000


@pytest.mark.parametrize("bad", [
    dict(starting_balance=0), dict(max_daily_loss_pct=0), dict(max_total_loss_pct=150),
    dict(profit_target_pct=-1), dict(max_open_positions=0), dict(max_lots=-1),
    dict(weekend_close_hour_utc=24), dict(safety_buffer_pct=5),
])
def test_bad_rules_are_rejected(bad):
    with pytest.raises(PropError):
        PropRules(**{**RULES, **bad})


def test_unknown_rule_key_is_rejected():
    with pytest.raises(PropError, match="unknown prop rule"):
        PropRules.from_dict({**RULES, "max_dayly_loss_pct": 5})


# ---------- limits and room ----------

def test_static_floor_and_limits():
    g = guard()
    assert g.daily_limit() == pytest.approx(500)
    assert g.total_limit() == pytest.approx(1000)
    assert g.floor() == pytest.approx(9000)
    assert g.target_amount() == pytest.approx(1000)


def test_trailing_floor_follows_peak_equity():
    g = guard(trailing_drawdown=True)
    g.update(MON_10, 10500)
    assert g.floor() == pytest.approx(9500)


def test_room_is_positive_at_the_day_start():
    g = guard()
    g.update(MON_10, 10000)
    daily, total = g.room(10000)
    assert daily == pytest.approx(500)
    assert total == pytest.approx(1000)


def test_room_shrinks_with_a_loss_and_is_zero_at_the_limit():
    g = guard()
    g.update(MON_10, 10000)
    daily, _ = g.room(9500)
    assert daily == pytest.approx(0)


# ---------- status transitions ----------

def test_daily_loss_breach_fails_the_account():
    g = guard()
    g.update(MON_10, 10000)
    changes = g.update("2026-01-05 12:00", 9499)
    assert changes == [FAILED] and g.s.status == FAILED
    assert "daily loss" in g.s.fail_reason


def test_total_loss_breach_fails_the_account():
    g = guard(max_daily_loss_pct=9)
    g.update(MON_10, 10000)
    g.update("2026-01-05 11:00", 9200)
    assert g.update("2026-01-05 12:00", 8990) == [FAILED]
    assert "total loss" in g.s.fail_reason


def test_day_start_equity_is_the_last_equity_before_rollover():
    g = guard()
    g.update("2026-01-05 23:00", 9800)
    g.update(TUE_10, 9700)                     # new day starts from 9800, not from the old day's start
    assert g.s.day_start_equity == pytest.approx(9800)


def test_failed_is_sticky_until_a_person_resets_it():
    g = guard()
    g.update(MON_10, 10000)
    g.update("2026-01-05 12:00", 9000)
    assert g.s.status == FAILED
    g.update(TUE_10, 10400)
    assert g.s.status == FAILED
    g.reset(TUE_10)
    assert g.s.status == ACTIVE and g.s.fail_reason is None


def test_profit_target_passes_only_after_minimum_trading_days():
    g = guard(min_trading_days=2)
    g.update(MON_10, 10000)
    g.record_entry(MON_10)
    assert g.update("2026-01-05 15:00", 11000) == []        # target hit, 1 trading day only
    g.record_entry(TUE_10)
    assert g.update("2026-01-06 15:00", 11000) == [PASSED]


# ---------- entry gate and sizing ----------

def test_entry_allowed_when_room_and_risk_fit():
    g = guard()
    g.update(MON_10, 10000)
    assert g.check_entry(MON_10, 10000, 0, 50) == (True, "ok")


def test_entry_blocked_inside_the_daily_buffer():
    g = guard()
    g.update(MON_10, 10000)
    ok, why = g.check_entry(MON_10, 9450, 0, 10)            # 50 of room left, buffer 50
    assert not ok and "daily-loss safety buffer" in why


def test_entry_blocked_when_risk_exceeds_remaining_room():
    g = guard()
    g.update(MON_10, 10000)
    ok, why = g.check_entry(MON_10, 9700, 0, 400)           # room 300 minus buffer 50
    assert not ok and "exceeds" in why


def test_entry_blocked_at_max_open_positions():
    g = guard()
    g.update(MON_10, 10000)
    assert g.check_entry(MON_10, 10000, 1, 10) == (False, "prop: max open positions reached")


def test_entry_blocked_on_a_failed_or_passed_account():
    g = guard()
    g.update(MON_10, 10000)
    g.update("2026-01-05 12:00", 9000)
    ok, why = g.check_entry("2026-01-05 13:00", 9000, 0, 10)
    assert not ok and FAILED in why


def test_max_lots_for_respects_risk_percent_room_and_lot_cap():
    g = guard(max_lots=0.3)
    g.update(MON_10, 10000)
    # risk per lot 100: 1% of 10000 = 100 -> 1 lot by risk, capped at 0.3 by max_lots
    assert g.max_lots_for(10000, 100) == pytest.approx(0.3)
    g2 = guard(max_lots=0)
    g2.update(MON_10, 10000)
    # 1 lot by risk; room 500 - 50 buffer = 450 -> 4.5 lots. The risk cap wins.
    assert g2.max_lots_for(10000, 100) == pytest.approx(1.0)


def test_max_lots_for_zero_risk_is_zero():
    assert guard().max_lots_for(10000, 0) == 0.0


# ---------- weekends ----------

@pytest.mark.parametrize("ts,blocked", [
    ("2026-01-09 19:59", False),   # Friday before the cut-off
    ("2026-01-09 20:00", True),    # Friday at the cut-off
    ("2026-01-10 10:00", True),    # Saturday
    ("2026-01-05 10:00", False),   # Monday
])
def test_weekend_window(ts, blocked):
    assert guard().weekend_blocked(ts) is blocked


def test_must_flatten_on_friday_close_and_on_failure():
    g = guard()
    g.update("2026-01-09 20:00", 10000)
    assert g.must_flatten("2026-01-09 20:00") == "weekend close"
    g2 = guard()
    g2.update(MON_10, 10000)
    g2.update("2026-01-05 12:00", 9000)
    assert "FAILED" in g2.must_flatten(MON_10)


def test_weekend_rule_can_be_disabled():
    assert guard(close_before_weekend=False).weekend_blocked("2026-01-10 10:00") is False


# ---------- trading day reset ----------

def test_day_reset_hour_moves_the_day_boundary():
    g = guard(day_reset_hour_utc=5)
    assert g._day(pd_ts("2026-01-05 04:59")) == "2026-01-04"
    assert g._day(pd_ts("2026-01-05 05:00")) == "2026-01-05"


def pd_ts(s):
    import pandas as pd
    return pd.Timestamp(s)


def test_timezone_aware_timestamps_are_normalised_to_utc():
    g = guard()
    g.update("2026-01-05T15:30:00+05:30", 10000)            # 10:00 UTC
    assert g.s.last_ts.startswith("2026-01-05T10:00:00")


# ---------- profiles and saved state ----------

CFG = {"enabled": True, "active_profile": "a",
       "profiles": {"a": {"rules": RULES}, "b": {"rules": {**RULES, "starting_balance": 5000}}}}


def test_select_profile_and_unknown_profile():
    name, rules = select_profile(CFG)
    assert name == "a" and rules.starting_balance == 10000
    with pytest.raises(PropError):
        select_profile({**CFG, "active_profile": "missing"})


def test_profile_change_is_refused_while_saved_state_belongs_to_another():
    saved = {"profile": "a", "status": ACTIVE}
    with pytest.raises(PropError, match="belongs to profile"):
        guard_from_config({**CFG, "active_profile": "b"}, saved)


def test_saved_state_is_restored_for_the_same_profile():
    saved = {"profile": "a", "status": FAILED, "fail_reason": "x", "peak_equity": 10200.0}
    g = guard_from_config(CFG, saved)
    assert g.s.status == FAILED and g.s.peak_equity == 10200.0


def test_fresh_state_when_nothing_is_saved():
    g = guard_from_config(CFG, None)
    assert g.s.status == ACTIVE and g.s.profile == "a"


def test_example_profiles_in_config_are_valid():
    cfg = json.loads(Path(P.PROP_CONFIG).read_text())
    for name, prof in cfg["profiles"].items():
        PropRules.from_dict(prof["rules"])
    select_profile(cfg)


# ---------- operator CLI ----------

def test_cli_show_lists_profiles(capsys):
    assert P.main(["show"]) == 0
    out = capsys.readouterr().out
    assert "generic_2step_example" in out and "valid" in out


def test_cli_reset_needs_confirm(monkeypatch, tmp_path, capsys):
    from algolab.ops import state as ops_state
    monkeypatch.setattr(ops_state, "STATE_DIR", tmp_path)
    (tmp_path / P.PROP_STATE).write_text(json.dumps({"profile": "a", "status": FAILED, "fail_reason": "x"}))
    assert P.main(["reset"]) == 2
    assert json.loads((tmp_path / P.PROP_STATE).read_text())["status"] == FAILED
    assert P.main(["reset", "--confirm"]) == 0
    assert json.loads((tmp_path / P.PROP_STATE).read_text())["status"] == ACTIVE
