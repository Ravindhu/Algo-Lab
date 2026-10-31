"""Discovered-candidate time exit (parity with the discovery backtest), config-driven timeframes, and H4 replay.

Offline. The position manager is driven by a fake adapter, so no MT5 terminal is needed."""
import asyncio

import numpy as np
import pandas as pd
import pytest

from algolab.framework.e2e import build_data, required_timeframes, run_e2e
from algolab.framework.timeframe import normalize_tf
from algolab.live import positions

MAGIC, DEV = 20262002, 20


def _bars_to(last_closed: str, n: int = 12) -> pd.DataFrame:
    """M15 bars ending with `last_closed` (closed), plus one forming row, as the adapter returns them."""
    closed_idx = pd.date_range(end=pd.Timestamp(last_closed), periods=n, freq="15min")
    idx = closed_idx.append(pd.DatetimeIndex([closed_idx[-1] + pd.Timedelta(minutes=15)]))
    return pd.DataFrame({"open": 1.10, "high": 1.1005, "low": 1.0995, "close": 1.10}, index=idx)


class FakeAdapter:
    """Only the calls the plain-record path makes."""

    def __init__(self, bars: pd.DataFrame, open_: bool = True):
        self.bars, self.open = bars, open_
        self.closes: list[tuple] = []

    async def candles(self, symbol, tf, count):
        return self.bars

    def positions_by_magic(self, magic):
        return [{"ticket": 1, "symbol": "EURUSD", "side": 1, "price_open": 1.10,
                 "sl": 1.09, "tp": 1.12, "volume": 0.1}] if self.open else []

    def close_position(self, ticket, symbol, side, volume, magic, deviation):
        self.closes.append((ticket, side, volume))
        self.open = False
        return {"ticket": ticket, "fill_price": 1.1001, "volume": volume}

    def realized_pnl(self, ticket):
        return 12.5


def _record(side: str, hold: int | None) -> dict:
    buy = side == "BUY"
    rec = {"ticket": 1, "symbol": "EURUSD", "side": side, "lots": 0.1, "fill_price": 1.10,
           "sl": 1.09 if buy else 1.11, "tp": 1.12 if buy else 1.08, "bar": "2026-01-05 10:00"}
    return positions.new_plain_record(rec, strategy="discovered", tf="M15", hold_bars=hold)


def _cycle(ad, tracked):
    return asyncio.run(positions.manage(ad, tracked, magic=MAGIC, deviation=DEV))


def test_exit_after_matches_the_backtest_bar_count():
    # Backtest: signal on bar i, entry at the open of i+1, exit at the close of bar i+1+hold.
    rec = _record("BUY", hold=3)
    assert rec["exit_after"] == "2026-01-05 11:00:00"   # 10:00 + (1 + 3) x 15 min


def test_no_hold_means_no_time_exit():
    assert _record("BUY", hold=None)["exit_after"] is None


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_time_exit_closes_once_the_last_hold_bar_has_closed(side):
    tracked = {"1": _record(side, hold=3)}
    ad = FakeAdapter(_bars_to("2026-01-05 10:45"))        # bar 10:45 closed; 11:00 still forming
    assert _cycle(ad, tracked) == [] and "1" in tracked
    assert ad.closes == []

    ad.bars = _bars_to("2026-01-05 11:00")                # bar 11:00 has closed: exit now
    events = _cycle(ad, tracked)
    assert len(events) == 1 and events[0]["reason"] == "TIME"
    assert events[0]["pnl"] == 12.5 and events[0]["price"] == pytest.approx(1.1001)
    assert "1" not in tracked
    expected_side = -1 if side == "SELL" else 1
    assert ad.closes == [(1, expected_side, 0.1)]         # closed on the broker side, not the mirrored one


def test_scalp_style_plain_record_is_never_closed_by_time():
    tracked = {"1": _record("BUY", hold=None)}
    ad = FakeAdapter(_bars_to("2026-01-05 12:00"))
    assert _cycle(ad, tracked) == [] and "1" in tracked and ad.closes == []


def test_broker_stop_before_the_hold_is_classified_not_timed():
    tracked = {"1": _record("BUY", hold=3)}
    ad = FakeAdapter(_bars_to("2026-01-05 10:30"), open_=False)   # broker already closed it
    events = _cycle(ad, tracked)
    assert len(events) == 1 and events[0]["reason"] == "BROKER_CLOSE"
    assert ad.closes == []                                          # nothing sent to the broker


# ---- config-driven timeframes ---------------------------------------------------------------------

def test_normalize_accepts_every_spelling_a_config_may_use():
    assert normalize_tf("H4") == "H4" and normalize_tf("h4") == "H4"
    assert normalize_tf("15minute") == "M15" and normalize_tf("5minute") == "M5"
    assert normalize_tf("day") == "D1"
    with pytest.raises(ValueError):
        normalize_tf("H3")


def test_required_timeframes_come_from_each_entry_finest_first():
    engulfing = {"type": "engulfing_ocl", "params": {"htf": "H4", "ltf": "M15"}}
    ocl = {"type": "ocl", "params": {"htf": "15minute", "ltf": "5minute"}}
    scalp = {"type": "scalp", "params": {"ema_fast": 8}}          # names no timeframe: framework default
    assert required_timeframes([engulfing]) == ["M15", "H4"]
    assert required_timeframes([ocl]) == ["M5", "M15"]
    assert required_timeframes([scalp], default_tf="M5") == ["M5"]
    assert required_timeframes([engulfing, ocl, scalp], "M5") == ["M5", "M15", "H4"]


def _write_csv(path, index, seed):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, len(index)))
    df = pd.DataFrame({"open": close, "high": close + 0.0004, "low": close - 0.0004, "close": close},
                      index=index)
    df.index.name = "time"
    df.reset_index().to_csv(path, index=False)


def test_real_h4_data_is_aligned_to_the_clock_and_never_read_past_it(tmp_path):
    m15 = pd.date_range("2026-01-05 00:00", periods=400, freq="15min")
    h4 = pd.date_range("2026-01-01 00:00", periods=80, freq="4h")
    _write_csv(tmp_path / "EURUSD_M15.csv", m15, 1)
    _write_csv(tmp_path / "EURUSD_H4.csv", h4, 2)
    per, index, source = build_data(["EURUSD"], 0, 0, tmp_path, keep_fraction=0.5, tfs=("M15", "H4"))
    assert source.startswith("CSV")
    assert list(per["EURUSD"]) == ["M15", "H4"]
    assert per["EURUSD"]["M15"].index.equals(index)              # the clock is M15, cut at the kept fraction
    assert per["EURUSD"]["H4"].index.max() <= index[-1]          # no H4 bar opens after the replay's last bar


def test_missing_h4_file_falls_back_to_synthetic_not_a_half_real_run(tmp_path):
    _write_csv(tmp_path / "EURUSD_M15.csv", pd.date_range("2026-01-05", periods=300, freq="15min"), 1)
    _, _, source = build_data(["EURUSD"], 800, 7, tmp_path, tfs=("M15", "H4"))
    assert source.startswith("SYNTHETIC")


def test_engulfing_ocl_replays_on_h4_with_synthetic_data(tmp_path):
    rep = run_e2e(["EURUSD"], bars=4000, seed=7, prop_enabled=False, out_dir=tmp_path,
                  strategy_types=("engulfing_ocl",))
    assert rep["timeframes"] == ["M15", "H4"]
    assert rep["passed"], [c for c in rep["checks"] if not c["ok"]]
