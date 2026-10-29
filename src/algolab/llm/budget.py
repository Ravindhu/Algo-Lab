"""Token control for the LLM analyst.

  1. Compact payloads: only the candidate fields are sent, never candles or logs.
  2. Cache: a candidate that was already reviewed is never sent again.
  3. Daily budget: when it is spent, the analyst SKIPs (fail closed) until the next UTC day.
Usage is estimated at about 4 characters per token and stored in state/tokens.json."""
from __future__ import annotations

import hashlib
import json

from algolab.agent.multi_agent import ANALYST_PROMPT
from algolab.ops.state import now_utc, read_json, write_json
from algolab.profile import ROOT


class TokenBudget:
    def __init__(self, daily_limit: int, cache_entries: int = 500):
        self.limit = int(daily_limit)
        self.cache_entries = cache_entries

    def load(self) -> dict:
        today = now_utc().date().isoformat()
        s = read_json("tokens.json", {})
        if s.get("day") != today:
            s = {"day": today, "used": 0, "calls": 0, "cache": {}}
        return s

    def status(self) -> dict:
        s = self.load()
        return {"day": s["day"], "used": s["used"], "limit": self.limit,
                "calls": s["calls"], "cached": len(s["cache"])}


class CachedAnalyst:
    def __init__(self, analyst, budget: TokenBudget):
        self.analyst = analyst
        self.budget = budget

    @staticmethod
    def key(cand: dict) -> str:
        # The identity of a signal: the same closed bar must never be sent to the LLM twice.
        fields = {k: cand.get(k) for k in ("symbol", "side", "time")}
        return hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]

    def run(self, cand: dict) -> dict:
        s = self.budget.load()
        k = self.key(cand)
        if k in s["cache"]:
            return {**cand, **s["cache"][k], "source": "cache"}
        if s["used"] >= self.budget.limit:
            return {**cand, "decision": "SKIP", "confidence": 0,
                    "reason": "daily token budget used up", "source": "budget"}

        verdict = self.analyst.run(cand)
        v = {f: verdict.get(f) for f in ("decision", "confidence", "reason")}
        sent = len(ANALYST_PROMPT) + len(json.dumps(cand, separators=(",", ":")))
        got = len(json.dumps(v))
        s["used"] += (sent + got) // 4 + 10
        s["calls"] += 1
        s["cache"][k] = v
        if len(s["cache"]) > self.budget.cache_entries:
            s["cache"] = dict(list(s["cache"].items())[-self.budget.cache_entries:])
        write_json("tokens.json", s)
        return {**cand, **v, "source": "llm"}


def budget_from_config() -> TokenBudget:
    cfg = json.loads((ROOT / "config" / "llm.json").read_text())
    return TokenBudget(cfg["daily_token_budget"], cfg["cache_entries"])
