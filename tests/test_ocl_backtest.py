"""OCL backtest: same strategy, protection and prop code as live. Synthetic candles, no look-ahead."""
import pandas as pd

from algolab.backtest.engine import Costs
from algolab.framework.strategies.ocl import OclKeylevel
from algolab.framework.ocl_backtest import run_ocl_backtest
from algolab.framework.prop import PropGuard, PropRules
from test_ocl import FILL, FORMING, HTF_BREAK, frame

T0 = "2026-01-05 00:00"
# The third bar touches the 1.082 level. Entry is at the open of the next bar (01:55).
BT_TOUCH = [
    (1.085, 1.086, 1.083, 1.083),
    (1.083, 1.084, 1.083, 1.080),
    (1.080, 1.093, 1.0795, 1.092),
]
# A closing bar after the scenario. Exits from the bar before it are applied at its open or intrabar.
TAIL = (1.080, 1.081, 1.079, 1.080)


def scenario(after, htf_rows=HTF_BREAK):
    htf = frame(list(htf_rows) + [FORMING], start="2026-01-04 23:15", freq="15min")
    ltf = frame([FILL] * 20 + BT_TOUCH + list(after) + [TAIL], start=T0, freq="5min")
    return htf, ltf


def strategy(**kw):
    return OclKeylevel("EURUSD", htf="15minute", ltf="5minute",
                       **{"hm_enabled": False, "ltf_pattern_enabled": False, **kw})


ENTRY = (1.092, 1.093, 1.091, 1.092)


def test_entry_fills_at_the_next_open_and_hits_target():
    htf, ltf = scenario([ENTRY, (1.092, 1.145, 1.091, 1.140)])
    tr = run_ocl_backtest(strategy(), htf, ltf, Costs())
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "TP"
    t = tr.iloc[0]
    assert t["entry_time"] == pd.Timestamp(T0) + pd.Timedelta(minutes=5 * 23)
    assert t["signal_time"] < t["entry_time"]          # the signal came from a closed bar
    assert t["pnl"] > 0


def test_no_pattern_no_trades():
    flat = frame([FILL] * 6 + [FORMING], start="2026-01-04 23:15", freq="15min")
    _, ltf = scenario([ENTRY])
    assert run_ocl_backtest(strategy(), flat, ltf, Costs()).empty


def test_disaster_stop_is_checked_before_target_inside_one_bar():
    htf, ltf = scenario([ENTRY, (1.092, 1.145, 1.060, 1.100)])
    tr = run_ocl_backtest(strategy(), htf, ltf, Costs())
    assert tr.iloc[0]["reason"] == "DISASTER"


def test_ltf_stop_is_a_close_rule_not_a_wick_rule():
    # The bar after entry wicks below the 1.078 floor but closes above it. The LTF stop must not fire.
    htf, ltf = scenario([ENTRY, (1.092, 1.093, 1.077, 1.090)])
    tr = run_ocl_backtest(strategy(), htf, ltf, Costs())
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "EOD"


def test_spread_filter_blocks_wide_spread():
    htf, ltf = scenario([ENTRY, (1.092, 1.145, 1.091, 1.140)])
    assert run_ocl_backtest(strategy(), htf, ltf, Costs(spread=0.05)).empty


def test_breakeven_lifts_the_stop_so_a_pullback_exits_above_the_original_floor():
    # Bar 24 runs past the breakeven trigger (1.088), so the stop moves to the cost floor (about 1.0842).
    # Bar 25 closes at 1.080, below that lifted stop, so the exit happens at the next open (1.080).
    htf, ltf = scenario([ENTRY, (1.092, 1.125, 1.091, 1.124), (1.100, 1.101, 1.079, 1.080)])
    tr = run_ocl_backtest(strategy(target_rr=10.0), htf, ltf, Costs())
    t = tr.iloc[0]
    assert t["reason"] == "SL"
    assert t["exit"] > 1.078                           # above the original structural floor
    # The stop-out steps the ladder down, so the pattern retries (a second trade). That is correct.
    assert len(tr) >= 1


def test_prop_risk_cap_shrinks_the_position():
    htf, ltf = scenario([ENTRY, (1.092, 1.145, 1.091, 1.140)])
    free = run_ocl_backtest(strategy(), htf, ltf, Costs())
    rules = PropRules(starting_balance=10000, max_risk_per_trade_pct=0.3)    # 30 currency at risk
    capped = run_ocl_backtest(strategy(), htf, ltf, Costs(), prop=PropGuard(rules, profile="t"))
    assert len(free) == 1 and len(capped) == 1
    assert 0 < capped.iloc[0]["lots"] < free.iloc[0]["lots"]


def test_prop_daily_room_limits_the_size():
    htf, ltf = scenario([ENTRY, (1.092, 1.145, 1.091, 1.140)])
    rules = PropRules(starting_balance=10000, max_daily_loss_pct=0.5, safety_buffer_pct=0.4)
    g = PropGuard(rules, profile="t")
    g.update(pd.Timestamp("2026-01-04 23:00"), 10000)        # previous day closes at 10000
    tr = run_ocl_backtest(strategy(), htf, ltf, Costs(initial_balance=9965), prop=g)
    assert tr.empty or tr["lots"].max() <= 0.1               # the 50 limit leaves little room
