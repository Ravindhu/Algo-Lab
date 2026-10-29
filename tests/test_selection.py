"""Selection: automatic high-performance strategy selection for the Bot. No broker needed."""
import json
import numpy as np
import pandas as pd
import pytest

from algolab.optimize import selection as Sel


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Root, state dir and framework config all point into a temp dir.
    Nothing in this suite reads or writes the repo's real config."""
    from algolab.ops import journal
    import algolab.optimize.selection as _s
    monkeypatch.setattr(journal, "STATE_DIR", tmp_path)
    monkeypatch.setattr(_s, "REPORTS", tmp_path)
    monkeypatch.setattr(_s, "ROOT", tmp_path)
    monkeypatch.setattr(_s, "load_framework_config",
                        lambda: json.loads(json.dumps({
                            "enabled": True, "max_open": 3,
                            "strategies": [
                                {"type": "engulfing", "enabled": True, "params": {}},
                                {"type": "ocl", "enabled": True, "params": {}},
                                {"type": "scalp", "enabled": True, "params": {}}]})))
    monkeypatch.setattr(_s, "_closes", lambda: [])
    return tmp_path


def _set_journal(rows):
    Sel._closes = lambda: rows


def _closes_by(strategy: str, r: float, n: int) -> list[dict]:
    return [{"kind": "close", "strategy": strategy, "r": r} for _ in range(n)]


def test_score_needs_enough_evidence_before_it_ranks():
    m = Sel._score([0.5] * 5, {"min_trades": 20, "min_profit_factor": 1.0,
                               "min_lower_bound": 0.35})
    assert m["passes"] is None and "unranked" in m["why"]


def test_score_passes_positive_expectancy_and_fails_negative():
    ok = Sel._score([1.0] * 30, {"min_trades": 20, "min_profit_factor": 1.0,
                                 "min_lower_bound": 0.35})
    bad = Sel._score([-1.0] * 30, {"min_trades": 20, "min_profit_factor": 1.0,
                                   "min_lower_bound": 0.35})
    assert ok["passes"] is True and bad["passes"] is False


def test_ranks_by_expectancy_x_sqrt_n():
    monkey_rows = (_closes_by("ocl", 2.0, 40) + _closes_by("engulfing", 0.2, 40)
                   + _closes_by("scalp", -0.4, 40))
    monkey = dict(Sel._config()) | {"min_trades": 20, "min_profit_factor": 1.0,
                                    "min_lower_bound": 0.35}
    ranks = {k: Sel._score(v, monkey) for k, v in Sel._r_per_strategy(monkey_rows).items()}
    assert ranks["ocl"]["score"] > ranks["engulfing"]["score"]
    assert ranks["scalp"]["passes"] is False


def test_apply_enables_top_n_and_demotes_the_rest():
    rows = _closes_by("ocl", 2.0, 40) + _closes_by("scalp", 0.3, 40)
    Sel._closes = lambda: rows  # not used (patched in fixture) - for clarity
    Sel.apply_selection(cfg=Sel._config() | {"top_n": 1, "min_trades": 20,
                                             "min_profit_factor": 1.0,
                                             "min_lower_bound": 0.35}, dry_run=True)


def test_dry_run_reports_but_does_not_write(tmp_path):
    Sel._closes = lambda: _closes_by("ocl", 2.0, 40) + _closes_by("scalp", -0.3, 40)
    monkey = Sel._config() | {"top_n": 1, "min_trades": 20, "min_profit_factor": 1.0,
                              "min_lower_bound": 0.35}
    r = Sel.apply_selection(cfg=monkey, dry_run=True)
    assert any(c["strategy"] == "scalp" for c in r["changes"])
    assert not (tmp_path / "framework.json").exists()


def test_apply_writes_framework_json(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "framework.json").write_text(json.dumps({
        "enabled": True, "strategies": [
            {"type": "engulfing", "enabled": True},
            {"type": "ocl", "enabled": True},
            {"type": "scalp", "enabled": True}]}))
    Sel._closes = lambda: _closes_by("ocl", 2.0, 40) + _closes_by("scalp", -0.3, 40)
    monkey = Sel._config() | {"top_n": 1, "min_trades": 20, "min_profit_factor": 1.0,
                              "min_lower_bound": 0.35}
    r = Sel.apply_selection(cfg=monkey, dry_run=False)
    assert any(c["strategy"] == "scalp" and c["would"] is False for c in r["changes"])
    written = json.loads((cfg_dir / "framework.json").read_text())
    assert next(e for e in written["strategies"] if e["type"] == "scalp")["enabled"] is False
