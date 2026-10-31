"""Trade setup (HTF, LTF, risk, symbols): validation, the config map, and the write gate. Runs on a temp config tree."""
import json

import pytest

from algolab.dashboard import trade_setup as ts


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("ALGOLAB_PROFILE", raising=False)
    cfg = tmp_path / "config"
    _write(cfg / "framework.json", {
        "enabled": False, "risk_per_trade": 0.005,
        "strategies": [
            {"type": "scalp", "enabled": True, "params": {"rsi_n": 14}},
            {"type": "engulfing_ocl", "enabled": False,
             "params": {"htf": "H4", "ltf": "M15", "fib_pct": 0.886}},
        ]})
    _write(cfg / "live.json", {"risk_per_trade": 0.005, "max_lots": 0.5})
    _write(cfg / "ops.json", {"symbols": ["EURUSD", "GBPUSD"], "risk": {"max_daily_loss": 0.03}})
    _write(cfg / "profiles" / "local.json", {"name": "local", "scanner": {"htf": "H4", "rsi_n": 9}})
    return tmp_path


def _setup(**over):
    base = {"htf": "H4", "ltf": "M15", "risk_pct": 0.5, "symbols": ["EURUSD", "GBPUSD"]}
    base.update(over)
    return base


# ---------- symbols ----------

def test_add_symbol_normalises_and_appends():
    out, err = ts.add_symbol(["EURUSD"], " usdjpy ")
    assert err is None and out == ["EURUSD", "USDJPY"]


def test_add_symbol_rejects_empty_short_duplicate_and_full():
    assert ts.add_symbol([], "  ")[1] == "Enter a symbol, e.g. EURUSD"
    assert ts.add_symbol([], "EU")[1].startswith("Use 3 to 6")
    assert ts.add_symbol(["EURUSD"], "eurusd")[1] == "EURUSD is already added"
    full = [f"S{i:03d}" for i in range(ts.MAX_SYMBOLS)]
    assert ts.add_symbol(full, "EURUSD")[1] == f"Maximum {ts.MAX_SYMBOLS} symbols"


def test_add_symbol_error_leaves_list_unchanged():
    base = ["EURUSD"]
    out, err = ts.add_symbol(base, "EURUSD")
    assert err and out == base and out is not base


# ---------- validation ----------

def test_valid_setup_has_no_errors(root):
    assert ts.validate_setup(_setup(), root) == []


def test_htf_must_be_higher_than_ltf(root):
    errs = ts.validate_setup(_setup(htf="M15", ltf="M15"), root)
    assert "HTF must be higher than LTF" in errs


def test_unknown_timeframes_are_rejected(root):
    errs = ts.validate_setup(_setup(htf="W1", ltf=None), root)
    assert "Pick an HTF: H1, H4 or D1" in errs and "Pick an LTF: M5, M15 or M30" in errs


def test_risk_cap_comes_from_daily_loss_limit(root):
    assert ts.risk_cap_pct({"risk": {"max_daily_loss": 0.01}}) == 1.0
    assert ts.risk_cap_pct({"risk": {"max_daily_loss": 0.03}}) == 3.0
    assert ts.risk_cap_pct({}) == ts.RISK_ABS_MAX_PCT
    assert ts.validate_setup(_setup(risk_pct=3.5), root) == ["Risk must be between 0.1% and 3.0%"]


def test_risk_below_minimum_and_non_numeric_are_rejected(root):
    assert ts.validate_setup(_setup(risk_pct=0.05), root)
    assert ts.validate_setup(_setup(risk_pct=None), root) == ["Risk must be a number"]
    assert ts.validate_setup(_setup(risk_pct=True), root) == ["Risk must be a number"]
    assert ts.validate_setup(_setup(risk_pct=float("nan")), root) == ["Risk must be a number"]


def test_symbol_problems_are_reported(root):
    assert ts.validate_setup(_setup(symbols=[]), root) == ["Add at least one symbol"]
    assert any("Invalid symbol" in e for e in ts.validate_setup(_setup(symbols=["EUR-USD"]), root))
    assert ts.validate_setup(_setup(symbols=["EURUSD", "EURUSD"]), root) == ["Duplicate symbols"]


# ---------- read and write ----------

def test_read_setup_maps_config_keys(root):
    s = ts.read_setup(root)
    assert s == {"htf": "H4", "ltf": "M15", "risk_pct": 0.5,
                 "symbols": ["EURUSD", "GBPUSD"], "risk_cap_pct": 3.0}


def test_save_is_refused_without_write_permission_and_touches_nothing(root):
    before = (root / "config" / "framework.json").read_text()
    with pytest.raises(ts.SetupError, match="DASH_ALLOW_WRITES"):
        ts.save_setup(_setup(htf="D1"), root, allow_writes=False)
    assert (root / "config" / "framework.json").read_text() == before
    assert not list((root / "config").rglob("*.bak"))


def test_invalid_setup_writes_nothing(root):
    snapshot = {p: p.read_text() for p in (root / "config").rglob("*.json")}
    with pytest.raises(ts.SetupError):
        ts.save_setup(_setup(htf="M5", ltf="M15"), root, allow_writes=True)
    assert {p: p.read_text() for p in (root / "config").rglob("*.json")} == snapshot


def test_save_writes_each_key_and_keeps_the_rest(root):
    out = ts.save_setup(_setup(htf="D1", ltf="M30", risk_pct=1.0, symbols=["usdjpy", "AUDUSD"]),
                        root, allow_writes=True)
    cfg = root / "config"
    fw = json.loads((cfg / "framework.json").read_text())
    eng = next(e for e in fw["strategies"] if e["type"] == "engulfing_ocl")
    assert eng["params"] == {"htf": "D1", "ltf": "M30", "fib_pct": 0.886}   # other params kept
    assert fw["risk_per_trade"] == 0.01
    assert next(e for e in fw["strategies"] if e["type"] == "scalp")["params"] == {"rsi_n": 14}
    assert json.loads((cfg / "live.json").read_text())["risk_per_trade"] == 0.01
    assert json.loads((cfg / "live.json").read_text())["max_lots"] == 0.5
    assert json.loads((cfg / "ops.json").read_text())["symbols"] == ["USDJPY", "AUDUSD"]
    assert json.loads((cfg / "ops.json").read_text())["risk"] == {"max_daily_loss": 0.03}
    assert json.loads((cfg / "profiles" / "local.json").read_text())["scanner"]["htf"] == "D1"
    assert out["needs_validate"] is True
    assert "config/framework.json" in out["written"] and "config/ops.json" in out["written"]


def test_save_makes_backups_before_replacing(root):
    cfg = root / "config"
    original = (cfg / "framework.json").read_text()
    ts.save_setup(_setup(htf="D1"), root, allow_writes=True)
    assert (cfg / "framework.json.bak").read_text() == original
    assert not (cfg / "framework.json.tmp").exists()


def test_needs_validate_is_false_when_only_symbols_change(root):
    out = ts.save_setup(_setup(symbols=["EURUSD"]), root, allow_writes=True)
    assert out["needs_validate"] is False            # symbols are not in the validation fingerprint


def test_save_round_trips_through_read(root):
    ts.save_setup(_setup(htf="H1", ltf="M5", risk_pct=2.0, symbols=["EURUSD"]), root, allow_writes=True)
    s = ts.read_setup(root)
    assert (s["htf"], s["ltf"], s["risk_pct"], s["symbols"]) == ("H1", "M5", 2.0, ["EURUSD"])


def test_save_fails_cleanly_when_framework_has_no_engulfing_entry(root):
    fw_path = root / "config" / "framework.json"
    fw = json.loads(fw_path.read_text())
    fw["strategies"] = [s for s in fw["strategies"] if s["type"] != "engulfing_ocl"]
    fw_path.write_text(json.dumps(fw))
    before = {p: p.read_text() for p in (root / "config").rglob("*.json")}
    with pytest.raises(ts.SetupError, match="no engulfing_ocl entry"):
        ts.save_setup(_setup(), root, allow_writes=True)
    assert {p: p.read_text() for p in (root / "config").rglob("*.json")} == before
