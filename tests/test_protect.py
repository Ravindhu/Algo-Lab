"""Protection rules (protect.py): breakeven, close-based stop and trail, intrabar disaster stop and target.
Expected values follow ocl_keylevel.py defaults: breakeven_rr 1.5, round_trip_cost_pct 0.20, atr_multiplier 2.0."""
import math

import pytest

from algolab.framework.protect import Protection, Trade

P = Protection()


def trade(entry=1.092, trigger=1.082, htf_sl=1.078, broker_sl=1.074, tp=None, p=P):
    return Trade.open(entry=entry, trigger=trigger, htf_sl=htf_sl, broker_sl=broker_sl, tp=tp, p=p)


def test_breakeven_levels_follow_the_original_formulas():
    t = trade()
    risk = 1.082 - 1.078
    assert t.risk == pytest.approx(risk)
    assert t.be_trigger == pytest.approx(1.082 + risk * 1.5)          # trigger + 1.5R
    assert t.be_floor == pytest.approx(1.082 * (1 + 0.20 / 100))      # trigger x (1 + cost)


def test_no_breakeven_when_risk_is_not_positive():
    t = trade(trigger=1.078, htf_sl=1.078)
    assert t.be_trigger == math.inf


def test_disaster_stop_is_checked_before_target_within_a_bar():
    t = trade(tp=1.120)
    assert t.intrabar_exit(high=1.130, low=1.070) == "DISASTER"
    assert t.intrabar_exit(high=1.121, low=1.080) == "TP"
    assert t.intrabar_exit(high=1.100, low=1.080) is None


def test_no_target_means_dynamic_and_no_tp_exit():
    t = trade(tp=None)
    assert t.intrabar_exit(high=9.9, low=1.09) is None


def test_ltf_stop_is_judged_on_the_close_not_the_wick():
    t = trade()
    assert t.on_closed_bar(high=1.093, close=1.079, atr=0.002, p=P) is None   # wick below 1.078 ignored
    assert t.on_closed_bar(high=1.080, close=1.077, atr=0.002, p=P) == "SL"   # close below the stop


def test_breakeven_arms_on_high_and_lifts_stop_to_the_floor():
    t = trade()
    t.on_closed_bar(high=t.be_trigger + 0.0001, close=t.be_trigger, atr=0.002, p=P)
    assert t.be_armed is True
    assert t.stop == pytest.approx(max(1.078, t.be_floor))


def test_stop_only_moves_up():
    t = trade()
    t.on_closed_bar(high=t.be_trigger + 0.001, close=1.10, atr=0.002, p=P)
    lifted = t.stop
    t.on_closed_bar(high=1.095, close=1.096, atr=0.002, p=P)
    assert t.stop == lifted >= 1.078


def test_trail_follows_highest_close_minus_two_atr_and_never_drops():
    t = trade()
    t.on_closed_bar(high=1.105, close=1.100, atr=0.002, p=P)
    assert t.trail == pytest.approx(1.100 - 2 * 0.002)
    t.on_closed_bar(high=1.112, close=1.110, atr=0.002, p=P)
    high_trail = t.trail
    assert high_trail == pytest.approx(1.110 - 2 * 0.002)
    t.on_closed_bar(high=1.104, close=1.103, atr=0.002, p=P)     # lower close: trail must not drop
    assert t.trail == high_trail


def test_close_below_the_trail_exits_with_trail():
    t = trade()
    t.on_closed_bar(high=1.105, close=1.100, atr=0.002, p=P)
    assert t.on_closed_bar(high=1.100, close=t.trail - 0.0001, atr=0.002, p=P) == "TRAIL"


def test_trail_disabled_leaves_no_trail():
    p = Protection(atr_trail_enabled=False)
    t = trade(p=p)
    t.on_closed_bar(high=1.105, close=1.100, atr=0.002, p=p)
    assert t.trail is None


def test_round_trip_serialisation_keeps_the_state():
    t = trade()
    t.on_closed_bar(high=1.105, close=1.100, atr=0.002, p=P)
    assert Trade.from_dict(t.to_dict()) == t


def test_protection_rejects_bad_values():
    with pytest.raises(ValueError):
        Protection(atr_multiplier=0)
    with pytest.raises(ValueError):
        Protection(breakeven_rr=-1)
