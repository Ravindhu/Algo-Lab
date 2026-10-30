"""Start/stop control and read-only status for the FastAPI dashboard. No web framework here, so the logic
is testable on its own.

Stop engages the kill switch (state/KILL), which the bot already checks every cycle. Start removes it.
Neither changes DRY_RUN, config, or places orders."""
from __future__ import annotations

from algolab.ops import state as st


def trading_enabled() -> bool:
    return not st.kill_switch_on()


def algo_status() -> dict:
    kill = st.kill_switch_on()
    return {"trading_enabled": not kill, "kill": kill}


def stop_algo(reason: str = "stopped from dashboard") -> dict:
    st.engage_kill(reason)
    return algo_status()


def start_algo() -> dict:
    f = st.STATE_DIR / "KILL"
    if f.exists():
        f.unlink()
    return algo_status()


def snapshot() -> dict:
    """Everything the live page needs in one read. Missing files give empty values, never errors."""
    return {**algo_status(),
            "live": st.read_json("live_last_cycle.json", {}) or {},
            "framework": st.read_json("framework_last_cycle.json", {}) or {}}
