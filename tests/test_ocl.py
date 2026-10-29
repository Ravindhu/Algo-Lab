"""OCL strategy rules on hand-built candles. No broker, no network.
FILL bars never touch the 1.082 trigger level, so each LTF frame has its own decision bar."""
import pandas as pd
import pytest

from algolab.framework.strategies.multi import MultiTimeframeStrategy
from algolab.framework.strategies.ocl import (SOURCE_TF, OclKeylevel, OclParams, SetupState,
                                              hm_bullish, map_timeframe)

FILL = (1.090, 1.091, 1.089, 1.090)
FORMING = (1.0, 1.0, 1.0, 1.0)

# C1 red, C2 red (open 1.095), C3 green closing above C2 open. C4 is the newest closed HTF bar.
HTF_BREAK = [
    (1.100, 1.105, 1.090, 1.095),
    (1.095, 1.100, 1.080, 1.085),
    (1.082, 1.098, 1.078, 1.096),
    (1.096, 1.100, 1.094, 1.098),
]
# C3 green but closes BELOW C2 open; C4 closes above it and confirms.
HTF_C4 = [
    (1.100, 1.105, 1.090, 1.095),
    (1.095, 1.100, 1.080, 1.085),
    (1.082, 1.090, 1.079, 1.090),
    (1.090, 1.100, 1.088, 1.097),
]
# LTF: red, red, green. The last low 1.0795 touches the conservative level 1.082.
LTF_TOUCH = [
    (1.085, 1.086, 1.083, 1.083),
    (1.083, 1.084, 1.079, 1.080),
    (1.080, 1.093, 1.0795, 1.092),
]


def frame(rows, start="2026-01-05 00:00", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def htf_frame(rows):
    return frame(list(rows) + [FORMING], start="2026-01-04 23:15", freq="15min")


def ltf_frame(tail):
    return frame([FILL] * 20 + list(tail) + [FORMING])


def make(**kw):
    base = dict(hm_enabled=False, ltf_pattern_enabled=False)
    base.update(kw)
    return OclKeylevel("EURUSD", htf="15minute", ltf="5minute", **base)


def run(strategy, htf_rows, ltf_tail):
    return strategy.signal_frames({"M15": htf_frame(htf_rows), "M5": ltf_frame(ltf_tail)})


# ---------- contract ----------

def test_is_multi_timeframe_and_refuses_plain_signal():
    s = make()
    assert isinstance(s, MultiTimeframeStrategy)
    assert s.timeframes == ("M15", "M5")
    with pytest.raises(TypeError):
        s.signal(frame([FILL] * 30))


def test_signal_dict_has_the_contract_fields():
    sig = run(make(), HTF_BREAK, LTF_TOUCH)
    for key in ("side", "sl_price", "sl_dist", "tp_dist", "htf_sl", "trigger", "exit_mode", "atr",
                "atr_period", "rsi", "time", "protect", "max_spread_atr"):
        assert key in sig
    assert sig["side"] == "BUY" and sig["sl_dist"] > 0 and sig["atr"] > 0


# ---------- HTF blueprint ----------

def test_c3_break_levels_and_floor_from_c2_c3_and_confirmation():
    s = make()
    sig = run(s, HTF_BREAK, LTF_TOUCH)
    assert sig is not None
    assert s.st.aggressive == pytest.approx(1.095)     # C2 open on a C3 break
    assert s.st.conservative == pytest.approx(1.082)   # C3 open
    assert sig["htf_sl"] == pytest.approx(1.078)       # min(C2 low, C3 low, C3 low)


def test_c4_confirmation_uses_c3_close_as_aggressive_and_c4_low_in_floor():
    s = make()
    sig = run(s, HTF_C4, LTF_TOUCH)
    assert sig is not None and s.st.confirm_bar == "C4"
    assert s.st.aggressive == pytest.approx(1.090)     # C3 close
    assert sig["htf_sl"] == pytest.approx(1.079)       # min(C2 low 1.080, C3 low 1.079, C4 low 1.088)


def test_no_reclaim_by_c3_or_c4_is_rejected():
    bad = list(HTF_C4)
    bad[3] = (1.090, 1.100, 1.088, 1.094)
    assert run(make(), bad, LTF_TOUCH) is None


def test_c1_green_breaks_red_red_green():
    bad = list(HTF_BREAK)
    bad[0] = (1.090, 1.100, 1.089, 1.099)
    assert run(make(), bad, LTF_TOUCH) is None


def test_broker_disaster_stop_sits_beyond_the_floor_by_one_atr():
    sig = run(make(), HTF_BREAK, LTF_TOUCH)
    assert sig["sl_price"] == pytest.approx(sig["htf_sl"] - sig["atr"], rel=1e-9)


# ---------- LTF trigger and exit mode ----------

def test_fixed_exit_when_entry_is_near_the_aggressive_level():
    sig = run(make(), HTF_BREAK, LTF_TOUCH)
    assert sig["exit_mode"] == "FIXED"                 # 1.092 >= 1.095 * 0.995
    assert sig["tp_dist"] == pytest.approx(3.0 * (1.092 - 1.078))


def test_dynamic_exit_near_conservative_has_no_target():
    tail = list(LTF_TOUCH)
    tail[2] = (1.080, 1.085, 1.0795, 1.084)            # 1.084 <= 1.082 * 1.005
    sig = run(make(), HTF_BREAK, tail)
    assert sig["exit_mode"] == "DYNAMIC" and sig["tp_dist"] is None


def test_no_touch_no_signal():
    tail = list(LTF_TOUCH)
    tail[2] = (1.092, 1.095, 1.090, 1.094)
    assert run(make(), HTF_BREAK, tail) is None


def test_forming_ltf_bar_is_ignored_even_if_it_touches():
    tail = list(LTF_TOUCH)
    tail[2] = (1.092, 1.095, 1.090, 1.094)
    frames = {"M15": htf_frame(HTF_BREAK),
              "M5": frame([FILL] * 20 + tail + [(1.094, 1.094, 1.000, 1.000)])}
    assert make().signal_frames(frames) is None


def test_ltf_rrg_gate_blocks_a_non_rrg_last_bar():
    tail = list(LTF_TOUCH)
    tail[2] = (1.080, 1.093, 1.0795, 1.079)            # last bar red
    assert run(make(ltf_pattern_enabled=True), HTF_BREAK, tail) is None


def test_aggressive_mode_uses_the_aggressive_level():
    s = make(trigger_level_mode="AGGRESSIVE")
    assert run(s, HTF_BREAK, LTF_TOUCH) is not None
    assert s.st.level == pytest.approx(1.095)


def test_pip_buffer_uses_the_symbol_pip_size():
    tail = list(LTF_TOUCH)
    tail[2] = (1.080, 1.093, 1.0825, 1.092)            # misses 1.082 by 5 pips
    assert run(make(), HTF_BREAK, tail) is None
    assert run(make(touch_buffer_pips=5.0), HTF_BREAK, tail) is not None


def test_jpy_pair_pip_is_one_hundredth():
    s = OclKeylevel("USDJPY", htf="15minute", ltf="5minute", touch_buffer_pips=5.0)
    assert s.symbol.pip == 0.01 and s.p.touch_buffer_pips == 5.0


# ---------- Hilega-Milega gate ----------

def test_hilega_milega_gate_direct():
    p = OclParams()
    falling = [2.0 - 0.01 * i for i in range(30)]
    assert hm_bullish(falling, p, oversold_cap=True) is False
    assert hm_bullish(falling + [1.715], p, oversold_cap=True) is True
    assert hm_bullish(falling[:5], p, oversold_cap=False) is False


# ---------- retry ladder and lifecycle ----------

def test_stop_out_steps_trigger_down_and_retries():
    s = make(max_ltf_retries=3)
    run(s, HTF_BREAK, LTF_TOUCH)
    s.on_entry()
    s.on_exit("SL", 1.0795)
    assert s.st.retries == 1 and s.st.level == pytest.approx(1.0795) and s.st.armed
    sig = run(s, HTF_BREAK, LTF_TOUCH)
    assert sig is not None and sig["retries"] == 1


def test_retries_exhausted_abandons_pattern():
    s = make(max_ltf_retries=1)
    run(s, HTF_BREAK, LTF_TOUCH)
    s.on_entry(); s.on_exit("SL", 1.0795)
    run(s, HTF_BREAK, LTF_TOUCH)
    s.on_entry(); s.on_exit("SL", 1.0790)
    assert s.st.armed is False
    assert run(s, HTF_BREAK, LTF_TOUCH) is None


@pytest.mark.parametrize("reason", ["TP", "TRAIL", "HTF", "DISASTER"])
def test_non_stop_exits_consume_the_pattern(reason):
    s = make()
    run(s, HTF_BREAK, LTF_TOUCH)
    s.on_entry(); s.on_exit(reason, 1.1)
    assert run(s, HTF_BREAK, LTF_TOUCH) is None


def test_htf_close_below_floor_invalidates_an_armed_pattern():
    s = make()
    run(s, HTF_BREAK, LTF_TOUCH)
    broken = list(HTF_BREAK)
    broken[3] = (1.096, 1.100, 1.070, 1.075)           # HTF close 1.075 < floor 1.078
    assert run(s, broken, LTF_TOUCH) is None
    assert s.st.armed is False


def test_state_round_trips_through_plain_dict():
    s = make()
    run(s, HTF_BREAK, LTF_TOUCH)
    s.on_entry()
    other = make()
    other.restore(s.state())
    assert other.st == s.st
    assert isinstance(SetupState(**s.state()), SetupState)


def test_signal_never_reads_the_forming_bar():
    base = run(make(), HTF_BREAK, LTF_TOUCH)
    for junk in [(9.0, 9.0, 0.1, 9.0), (0.5, 0.6, 0.4, 0.55)]:
        htf = frame(list(HTF_BREAK) + [junk], start="2026-01-04 23:15", freq="15min")
        ltf = frame([FILL] * 20 + list(LTF_TOUCH) + [junk], start="2026-01-05 00:00", freq="5min")
        assert make().signal_frames({"M15": htf, "M5": ltf}) == base


def test_short_data_returns_none():
    assert make().signal_frames({"M15": htf_frame([]), "M5": frame([FILL])}) is None


# ---------- mapping and parameters ----------

def test_timeframe_mapping():
    assert map_timeframe("5minute") == "M5" and map_timeframe("15minute") == "M15"
    assert map_timeframe("M15") == "M15" and SOURCE_TF["60minute"] == "H1"
    with pytest.raises(ValueError):
        map_timeframe("1month")


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown OCL parameter"):
        make(target_rrr=2.0)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        make(trigger_level_mode="MIDDLE")
