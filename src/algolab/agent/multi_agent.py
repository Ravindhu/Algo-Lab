"""Frontline multi-agent pipeline: Scanner -> Risk gate -> Analyst (Ollama).
Agents only recommend. They never place orders. Order execution stays in the bridge."""
import json
import logging

from algolab.llm.ollama_client import OllamaClient
from algolab.profile import load_profile
from algolab.signals.engulfing import scan_all

log = logging.getLogger("algolab.agents")

ANALYST_PROMPT = (
    "You are a forex trade analyst. Input is one candidate signal as JSON. "
    'Reply ONLY with JSON: {"decision":"TAKE" or "SKIP","confidence":0-1,"reason":"max 20 words"}. '
    "Skip when the pattern is weak or the context is unclear."
)


class ScannerAgent:
    def run(self, htf: str) -> list[dict]:
        return scan_all(htf)


class RiskAgent:
    def __init__(self, max_open: int):
        self.max_open = max_open

    def run(self, candidates: list[dict], open_positions: int) -> list[dict]:
        room = max(self.max_open - open_positions, 0)
        return candidates[:room]


class AnalystAgent:
    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def run(self, cand: dict) -> dict:
        raw = self.llm.chat(
            [{"role": "system", "content": ANALYST_PROMPT},
             {"role": "user", "content": json.dumps(cand, separators=(",", ":"))}],
            json_mode=True,
        )
        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            verdict = {"decision": "SKIP", "confidence": 0, "reason": "unparseable LLM output"}
        return {**cand, **verdict}


class Orchestrator:
    def __init__(self, profile: dict | None = None):
        self.p = profile or load_profile()
        o = self.p["ollama"]
        self.scanner = ScannerAgent()
        self.risk = RiskAgent(self.p["risk"]["max_open_trades"])
        self.analyst = AnalystAgent(OllamaClient(o["url"], o["api_key"], o["model"]))

    def run(self, open_positions: int = 0) -> dict:
        candidates = self.risk.run(self.scanner.run(self.p["scanner"]["htf"]), open_positions)
        reviewed = [self.analyst.run(c) for c in candidates]
        min_conf = float(self.p["analyst"]["min_confidence"])
        take = [d for d in reviewed
                if d.get("decision") == "TAKE" and float(d.get("confidence", 0)) >= min_conf]
        for d in take:
            log.info("trade candidate", extra={"symbol": d["symbol"], "side": d["side"]})
        return {"dry_run": self.p["dry_run"], "reviewed": reviewed, "take": take}
