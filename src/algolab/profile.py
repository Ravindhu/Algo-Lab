"""Loads a runtime profile (local or vps). Secrets and endpoints come from env vars."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load .env once, for every entry point (Task Scheduler, systemd, Docker, CLI).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def load_profile(name: str | None = None) -> dict:
    name = name or os.getenv("ALGOLAB_PROFILE", "local")
    p = json.loads((ROOT / "config" / "profiles" / f"{name}.json").read_text())
    o = p["ollama"]
    o["url"] = os.getenv("OLLAMA_URL", o.get("url", "")).rstrip("/")   # remote Ollama machine
    o["model"] = os.getenv("OLLAMA_MODEL", o["model"])
    o["api_key"] = os.getenv("OLLAMA_API_KEY", "")
    p["dry_run"] = os.getenv("DRY_RUN", str(p["dry_run"])).lower() == "true"
    # Strategy parameters live in config/strategy.json, the single source of truth.
    strat = json.loads((ROOT / "config" / "strategy.json").read_text())
    p["scanner"].update({k: strat[k] for k in ("rsi_n", "oversold", "overbought")})
    p["strategy"] = strat
    return p
