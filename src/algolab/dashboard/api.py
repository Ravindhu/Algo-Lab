"""FastAPI server for the AlgoLab dashboard: the live page, the discovery page, and a small JSON API.

Run:  uvicorn algolab.dashboard.api:app --host 127.0.0.1 --port 8050
Read-only except start/stop, which only toggle the kill switch (see control.py)."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from algolab.dashboard import control
from algolab.profile import ROOT

TEMPLATES = Path(__file__).parent / "templates"
REPORTS = ROOT / "reports"

app = FastAPI(title="AlgoLab dashboard")


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((TEMPLATES / name).read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def live_page():
    return _page("live.html")


@app.get("/discovery", response_class=HTMLResponse)
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
