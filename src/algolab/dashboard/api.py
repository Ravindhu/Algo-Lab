"""FastAPI server for the AlgoLab dashboard: the live page, the discovery page, and a small JSON API.

Run:  uvicorn algolab.dashboard.api:app --host 127.0.0.1 --port 8050
Writes are limited to two actions: start/stop, which only toggle the kill switch (see control.py),
and promote, which adds a DISABLED entry to framework.json. Nothing here enables a strategy."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from algolab import discovery_promote as dp
from algolab.dashboard import control
from algolab.profile import ROOT

TEMPLATES = Path(__file__).parent / "templates"
REPORTS = ROOT / "reports"

app = FastAPI(title="AlgoLab dashboard")


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((TEMPLATES / name).read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
@app.get("/live.html", response_class=HTMLResponse)
def live_page():
    return _page("live.html")


@app.get("/discovery", response_class=HTMLResponse)
@app.get("/discovery.html", response_class=HTMLResponse)
def discovery_page():
    return _page("discovery.html")


@app.get("/api/snapshot")
def snapshot():
    return control.snapshot()


@app.post("/api/algo/stop")
def stop_algo():
    return control.stop_algo()


@app.post("/api/algo/start")
def start_algo():
    return control.start_algo()


@app.get("/api/discovery")
def discovery_report():
    try:
        return json.loads((REPORTS / "discovery.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@app.get("/api/bank")
def bank():
    """Candidate bank: every candidate from the last discovery run, in rank order, with its live status."""
    return {"rows": dp.bank_rows(ROOT)}


@app.post("/api/discovery/promote/{rank}")
def promote(rank: int):
    """Add one candidate to framework.json as a DISABLED entry. Refused with 409 when it cannot run live."""
    try:
        entry = dp.promote(ROOT, rank)
    except dp.PromoteError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "rank": rank, "enabled": entry["enabled"],
            "symbol": entry["params"]["only_symbol"], "timeframe": entry["params"]["timeframe"]}
