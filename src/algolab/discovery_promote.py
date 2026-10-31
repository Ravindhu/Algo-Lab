"""Promote a discovery candidate into config/framework.json as a DISABLED "discovered" entry.

Used by `algolab promote` (CLI) and the dashboard button. Both are human actions.

What promotion does:
  - Reads reports/discovery.json (written by discovery_factory). Never runs the search.
  - Refuses candidates the live bot cannot run as written (see discovery_factory.live_compatible).
  - Backs up framework.json, then appends one entry with "enabled": false.
  - Refuses a genome that is already in framework.json.

What it does NOT do:
  - It never sets "enabled": true. The strategy trades only after the validation gate passes
    (`algolab validate`) and a person turns it on.
  - It never touches the framework's top-level "enabled" flag or any other strategy.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from algolab import discovery_factory as df_mod

COUNT_BARS = 400          # history the live strategy sees each cycle (>= MIN_BARS in discovered.py)


class PromoteError(Exception):
    """Promotion refused. The message is safe to show to the person."""


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise PromoteError(f"missing file: {path}") from e
    except json.JSONDecodeError as e:
        raise PromoteError(f"invalid JSON in {path}: {e}") from e


def list_candidates(root: Path) -> list[dict]:
    rep = _load_json(root / "reports" / "discovery.json")
    return list(rep.get("candidates") or [])


def promote(root: Path, rank: int) -> dict:
    """Add candidate #rank (1-based, as shown in the report) to framework.json, disabled.
    Returns the entry that was written."""
    root = Path(root)
    report_path = root / "reports" / "discovery.json"
    fw_path = root / "config" / "framework.json"

    rep = _load_json(report_path)
    if rep.get("applied"):
        raise PromoteError("report is marked applied; re-run discovery before promoting")
    cands = rep.get("candidates") or []
    if not cands:
        raise PromoteError("the last discovery run produced no candidates")
    if not (1 <= int(rank) <= len(cands)):
        raise PromoteError(f"rank must be between 1 and {len(cands)}")
    cand = cands[int(rank) - 1]
    genome = cand.get("genome") or {}

    ok, why = df_mod.live_compatible(genome)
    if not ok:
        raise PromoteError(f"candidate #{rank} cannot run live: {why}")

    fw = _load_json(fw_path)
    strategies = fw.setdefault("strategies", [])
    for e in strategies:
        if e.get("type") == "discovered" and (e.get("params") or {}).get("genome") == genome:
            raise PromoteError(f"candidate #{rank} is already in framework.json")

    test = cand.get("test") or {}
    entry = {
        "type": "discovered",
        "enabled": False,
        "reason": (f"promoted from discovery run {rep.get('generated_at', '?')[:19]} UTC, "
                   f"rank {rank}. Disabled until validate passes and a person enables it."),
        "params": {
            "timeframe": cand["tf"],
            "count": COUNT_BARS,
            "only_symbol": cand["symbol"],
            "genome": genome,
            "candidate": {"score": cand.get("score"), "trades": cand.get("trades"),
                          "holdout_pf": test.get("pf"), "rule": cand.get("desc", "")},
        },
    }
    strategies.append(entry)

    # Back up, then replace atomically so a crash never leaves a half-written framework.json.
    shutil.copy2(fw_path, fw_path.with_suffix(".json.bak"))
    tmp = fw_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fw, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, fw_path)
    return entry
