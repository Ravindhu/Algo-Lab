"""Tests for the audit fixes that are not covered elsewhere:
  2. default-deny strategy switch, and every framework.json entry says "enabled" explicitly
  3. broker server-clock bar times are converted to UTC, and fail closed without a server time
  4. dashboard password and write gate
  6. lot sizing does not lose a whole volume step to float error
Fixes 1 and 7 live in test_guard_split.py. Fix 5 is a compose setting, checked here by reading the file."""
import asyncio
import json
import types
from pathlib import Path

import pandas as pd
import pytest

from algolab.dashboard import security
from algolab.framework.strategies import registry
from algolab.live import adapter as A
from algolab.live.sizing import lots_from_risk

ROOT = Path(__file__).resolve().parents[1]


# --- fix 2: default-deny -------------------------------------------------------------------------

def test_every_framework_entry_states_enabled_explicitly():
    fw = json.loads((ROOT / "config" / "framework.json").read_text())
    missing = [e["type"] for e in fw["strategies"] if "enabled" not in e]
    assert missing == [], f"entries without an explicit enabled flag: {missing}"


def test_only_scalp_is_enabled_in_framework_config():
    # Behaviour preserved for scalp (it was on by default before); every other entry stays off until validated.
    fw = json.loads((ROOT / "config" / "framework.json").read_text())
    on = sorted(e["type"] for e in fw["strategies"] if registry.enabled(e))
    assert on == ["scalp"]


# --- fix 3: broker server time -> UTC -------------------------------------------------------------

def test_server_utc_offset_rounds_to_half_hours():
    base = 1_700_000_000.0
    assert A.server_utc_offset(base + 3 * 3600, base) == 3 * 3600          # UTC+3 broker
    assert A.server_utc_offset(base - 5 * 3600, base) == -5 * 3600         # UTC-5 broker
    assert A.server_utc_offset(base + 5.5 * 3600, base) == int(5.5 * 3600)  # half-hour zone
    assert A.server_utc_offset(base + 3 * 3600 + 400, base) == 3 * 3600    # ~7 min clock skew absorbed


class _Clock:
    def __init__(self, t):
        self.t = t

    def time(self):
        return self.t


def _candle_adapter(tick_time, bar_epoch):
    tick = types.SimpleNamespace(bid=1.1, ask=1.1001, time=tick_time)
    rates = [{"time": bar_epoch, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1}]
    mt5 = types.SimpleNamespace(
        symbol_select=lambda *a, **k: True,
        symbol_info_tick=lambda *a, **k: tick,
        copy_rates_from_pos=lambda *a, **k: rates,
        last_error=lambda: (0, "ok"),
        **{f"TIMEFRAME_{t}": 1 for t in A.TIMEFRAMES},
    )
    ad = A.Mt5Adapter()
    ad.mt5 = mt5
    return ad


def test_candles_are_stamped_in_utc_not_broker_time(monkeypatch):
    utc_now = 1_700_000_000
    monkeypatch.setattr(A, "time", _Clock(utc_now))
    broker_offset = 3 * 3600
    # The broker stamps the bar 3 hours ahead of UTC.
    ad = _candle_adapter(tick_time=utc_now + broker_offset, bar_epoch=utc_now + broker_offset)
    df = asyncio.run(ad.candles("EURUSD", "M5", 1))
    assert df.index[0] == pd.Timestamp(utc_now, unit="s")


def test_candles_fail_closed_without_server_time(monkeypatch):
    monkeypatch.setattr(A, "time", _Clock(1_700_000_000))
    ad = _candle_adapter(tick_time=0, bar_epoch=1_700_000_000)
    with pytest.raises(A.AdapterError, match="no server time"):
        asyncio.run(ad.candles("EURUSD", "M5", 1))


# --- fix 4: dashboard password and write gate ----------------------------------------------------

def _basic(user="ops", pw="s3cret"):
    import base64
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


def test_no_password_configured_refuses_everything(monkeypatch):
    monkeypatch.delenv("DASH_PASSWORD", raising=False)
    assert security.is_authorized(_basic()) is False
    with pytest.raises(security.DashboardSecurityError):
        security.require_password_for_start()


def test_correct_password_is_accepted_and_username_ignored(monkeypatch):
    monkeypatch.setenv("DASH_PASSWORD", "s3cret")
    assert security.is_authorized(_basic(user="anyone")) is True
    assert security.is_authorized(_basic(pw="wrong")) is False


@pytest.mark.parametrize("header", [None, "", "Bearer s3cret", "Basic", "Basic !!notbase64", "Basic " + "eA=="])
def test_malformed_or_missing_auth_fails_closed(monkeypatch, header):
    monkeypatch.setenv("DASH_PASSWORD", "s3cret")
    assert security.is_authorized(header) is False


def test_writes_are_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("DASH_ALLOW_WRITES", raising=False)
    assert security.writes_enabled() is False
    monkeypatch.setenv("DASH_ALLOW_WRITES", "TRUE")
    assert security.writes_enabled() is True
    monkeypatch.setenv("DASH_ALLOW_WRITES", "yes")      # only "true" counts
    assert security.writes_enabled() is False


def test_api_refuses_requests_without_login(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from algolab.dashboard import api
    monkeypatch.setenv("DASH_PASSWORD", "s3cret")
    client = TestClient(api.app)
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": _basic(pw="nope")}).status_code == 401


def test_api_start_is_blocked_without_write_permission(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from algolab.dashboard import api
    monkeypatch.setenv("DASH_PASSWORD", "s3cret")
    monkeypatch.delenv("DASH_ALLOW_WRITES", raising=False)
    client = TestClient(api.app)
    res = client.post("/api/algo/start", headers={"Authorization": _basic()})
    assert res.status_code == 403


# --- fix 5: compose wiring -----------------------------------------------------------------------

def test_compose_dashboard_state_mount_is_writable():
    text = (ROOT / "docker-compose.yml").read_text()
    assert "./state:/app/state" in text
    assert "./state:/app/state:ro" not in text    # Stop must be able to write state/KILL


# --- fix 6: lot sizing float edge ----------------------------------------------------------------

def test_lots_do_not_lose_a_step_to_float_error():
    # 0.3 / 0.1 / 0.1 is 29.999999999999993 in floating point. A plain int() gave 2.9 lots here.
    assert lots_from_risk(0.3, 0.1, 1.0, 1.0, max_lots=10.0, volume_step=0.1) == 3.0


def test_lots_still_respect_max_and_zero_inputs():
    assert lots_from_risk(1000.0, 0.0001, 100_000, 1.0, max_lots=2.0) == 2.0
    assert lots_from_risk(100.0, 0.0, 100_000, 1.0, max_lots=2.0) == 0.0
