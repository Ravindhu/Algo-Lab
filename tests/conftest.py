"""Shared test fixtures.

Every test runs with its own state folder. Traded-signal keys, journal, positions and kill switch are
persisted by design, so without this they leak between tests and between runs (and into the real state/)."""
import pytest

from algolab.ops import journal
from algolab.ops import state as ops_state


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ops_state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(journal, "STATE_DIR", None)
    return tmp_path / "state"
