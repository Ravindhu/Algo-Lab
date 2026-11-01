"""Monitor panels for the FastAPI dashboard: halts, stale validation, open positions, risk, equity, validate job.
Each test runs on the isolated state folder from conftest and a temporary config tree."""
import json
import sys

import pytest

from algolab.dashboard import panels
from algolab.ops import journal
from algolab.ops import state as st


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def root(tmp_path):
    _write(tmp_path / "config" / "ops.json", {"risk": {"max_daily_loss": 0.03, "max_drawdown_kill": 0.15}})
    _write(tmp_path / "config" / "llm.json", {"daily_token_budget": 1000})
    return tmp_path


# ---------- status and halts ----------

def test_status_lists_every_reason_trading_is_stopped(root):
    st.engage_kill("drawdown breached")
    s = panels.status(root)
    assert s["trading_enabled"] is False and s["kill"] is True
    assert s["kill_reason"] == "drawdown breached"
    assert s["can_trade"] is False
    assert any("kill switch on: drawdown breached" in h for h in s["halts"])
    assert "validation not_run" in s["halts"]


def test_status_with_no_halts_when_validated_and_no_kill(root, monkeypatch):
    from algolab.backtest import validate as v
    monkeypatch.setattr(v, "config_fingerprint", lambda: "abc")
    _write(root / "reports" / "validation.json", {"passed": True, "reasons": [], "fingerprint": "abc"})
    s = panels.status(root)
    assert s["halts"] == [] and s["trading_enabled"] is True and s["validation"]["state"] == "passed"
    assert s["can_trade"] is True


def test_stale_validation_blocks_and_says_why(root, monkeypatch):
    from algolab.backtest import validate as v
    monkeypatch.setattr(v, "config_fingerprint", lambda: "new")
    _write(root / "reports" / "validation.json", {"passed": True, "reasons": [], "fingerprint": "old"})
    val = panels.validation(root)
    assert val["state"] == "stale" and val["passed"] is False
    assert "settings changed" in val["reasons"][0]


def test_failed_validation_is_not_passed(root, monkeypatch):
    from algolab.backtest import validate as v
    monkeypatch.setattr(v, "config_fingerprint", lambda: "abc")
    _write(root / "reports" / "validation.json",
           {"passed": False, "reasons": ["profit factor 0.9 below 1.2"], "fingerprint": "abc"})
    assert panels.validation(root)["state"] == "failed"


def test_framework_block_and_cycle_error_are_halts(root):
    st.write_json("framework_last_cycle.json", {"ts": "t", "may_trade": False, "reason": "max open trades reached"})
    assert "max open trades reached" in panels.status(root)["halts"]
    st.write_json("framework_last_cycle.json", {"ts": "t", "error": "MT5 contract failed"})
    assert any("MT5 contract failed" in h for h in panels.status(root)["halts"])


# ---------- trades and open positions ----------

def test_open_positions_are_opens_without_a_close(root):
    journal.record_open(source="framework", strategy="scalp", symbol="EURUSD", side="BUY", ticket=1,
                        lots=0.1, entry=1.10, sl=1.09, tp=1.12)
    journal.record_open(source="framework", strategy="scalp", symbol="GBPUSD", side="SELL", ticket=2,
                        lots=0.1, entry=1.30, sl=1.31, tp=None)
    journal.record_close({"strategy": "scalp", "symbol": "GBPUSD", "reason": "TP", "price": 1.29,
                          "ticket": 2, "pnl": 12.5})
    opens = panels.open_positions()
    assert [p["ticket"] for p in opens] == [1]


def test_open_without_ticket_is_not_shown_as_open(root):
    journal.record_open(source="legacy", strategy="engulfing", symbol="EURUSD", side="BUY", ticket=None,
                        lots=0.1, entry=1.1, sl=1.09, tp=None)
    assert panels.open_positions() == []


def test_unknown_pnl_is_shown_as_unknown_not_zero(root):
    journal.record_open(source="framework", strategy="scalp", symbol="EURUSD", side="BUY", ticket=7,
                        lots=0.1, entry=1.1, sl=1.09, tp=None)
    journal.record_close({"strategy": "scalp", "symbol": "EURUSD", "reason": "SL", "price": 1.09,
                          "ticket": 7, "pnl": None})
    row = panels.trades()[0]
    assert row["kind"] == "close" and row["pnl"] == "unknown"


# ---------- risk ----------

def test_risk_ratios_and_status(root):
    st.write_json("risk.json", {"peak": 10000.0, "day": panels._today(), "day_start": 10000.0})
    st.write_json("framework_last_cycle.json", {"ts": "t", "equity": 9750.0})   # 2.5% down today
    r = panels.risk(root)
    daily = next(i for i in r["items"] if i["name"] == "Daily loss (ops)")
    assert daily["used_pct"] == 2.5 and daily["limit_pct"] == 3.0
    assert daily["status"] == "warn"                                             # 83% of the limit
    dd = next(i for i in r["items"] if i["name"].startswith("Drawdown"))
    assert dd["used_pct"] == 2.5 and dd["status"] == "ok"


def test_risk_breach_at_the_limit(root):
    st.write_json("risk.json", {"peak": 10000.0, "day": panels._today(), "day_start": 10000.0})
    st.write_json("framework_last_cycle.json", {"ts": "t", "equity": 9700.0})   # exactly 3% down
    daily = next(i for i in panels.risk(root)["items"] if i["name"] == "Daily loss (ops)")
    assert daily["status"] == "breach"


def test_risk_with_no_cycle_says_so(root):
    assert panels.risk(root) == {"equity": None, "items": [], "note": "no cycle has run yet"}


# ---------- equity history ----------

def test_equity_history_skips_torn_lines(root):
    st.append_equity(10000.0)
    st.append_equity(10050.5)
    path = st.STATE_DIR / "equity_history.jsonl"
    path.write_text(path.read_text() + "{not json\n", encoding="utf-8")
    rows = panels.equity()
    assert [r["equity"] for r in rows] == [10000.0, 10050.5]


def test_equity_history_empty_when_missing(root):
    assert panels.equity() == []


# ---------- autotune and tokens ----------

def test_autotune_best_is_the_highest_score_and_is_read_only(root):
    _write(root / "reports" / "autotune_latest.json", {
        "applied": False, "trials": [{"score": 1.0, "trades": 40, "rsi_n": 9, "junk": "x"},
                                     {"score": 2.5, "trades": 55, "rsi_n": 14, "sl_mult": 1.5}]})
    a = panels.autotune(root)
    assert a["trials"] == 2 and a["best"]["score"] == 2.5 and a["best"]["rsi_n"] == 14
    assert "junk" not in a["best"] and a["applied"] is False


def test_tokens_reset_on_a_new_day(root):
    st.write_json("tokens.json", {"day": "1999-01-01", "used": 900, "calls": 5, "cache": {"a": 1}})
    t = panels.tokens(root)
    assert t["used"] == 0 and t["calls"] == 0 and t["limit"] == 1000


def test_tokens_today(root):
    st.write_json("tokens.json", {"day": panels._today(), "used": 250, "calls": 3, "cache": {"a": 1, "b": 2}})
    t = panels.tokens(root)
    assert t["used"] == 250 and t["ratio"] == 0.25 and t["cached"] == 2


# ---------- validate job ----------

def test_validate_job_runs_and_records_success(root):
    job = panels.start_validate(root, background=False,
                                command=[sys.executable, "-c", "print('passed')"])
    assert job["state"] == "done" and job["returncode"] == 0 and "passed" in job["tail"]


def test_validate_job_records_failure(root):
    job = panels.start_validate(root, background=False,
                                command=[sys.executable, "-c", "import sys; print('no data'); sys.exit(2)"])
    assert job["state"] == "failed" and job["returncode"] == 2


def test_validate_job_refuses_a_second_run_while_one_is_running(root):
    st.write_json(panels.JOB_FILE, {"state": "running", "started": panels._now()})
    with pytest.raises(panels.JobBusy):
        panels.start_validate(root, background=False, command=[sys.executable, "-c", "pass"])


def test_validate_job_running_too_long_is_treated_as_left_behind(root):
    old = "2000-01-01T00:00:00+00:00"
    st.write_json(panels.JOB_FILE, {"state": "running", "started": old})
    job = panels.start_validate(root, background=False, command=[sys.executable, "-c", "print('ok')"])
    assert job["state"] == "done"


def test_validate_job_idle_by_default(root):
    assert panels.validate_job() == {"state": "idle"}
