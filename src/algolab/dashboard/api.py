"""FastAPI server for the AlgoLab dashboard: the live page, the discovery page, and a small JSON API.

Run:  DASH_PASSWORD=... uvicorn algolab.dashboard.api:app --host 127.0.0.1 --port 8050
Every route needs the DASH_PASSWORD (HTTP Basic). Without it, every request is refused (see security.py).
Writes are limited to three actions: start (needs DASH_ALLOW_WRITES=true), stop (always allowed), promote, which
adds a DISABLED entry to framework.json, and prop, which edits the active prop-firm rules (validated and audited;
applies from the next cycle). Start, promote and prop need DASH_ALLOW_WRITES=true. Nothing here enables a
strategy or the prop switch."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from algolab import discovery_promote as dp
from algolab.dashboard import control, prop_edit, security
from algolab.framework.prop import PropError
from algolab.profile import ROOT

TEMPLATES = Path(__file__).parent / "templates"
REPORTS = ROOT / "reports"

app = FastAPI(title="AlgoLab dashboard")


@app.middleware("http")
async def require_login(request: Request, call_next):
    """Every route, including the pages, needs the dashboard password. Refuses everything when none is set."""
    if not security.configured_password():
        return JSONResponse({"detail": "dashboard locked: set DASH_PASSWORD"}, status_code=503)
    if not security.is_authorized(request.headers.get("authorization")):
        return JSONResponse({"detail": "login required"}, status_code=401,
                            headers={"WWW-Authenticate": security.REALM})
    return await call_next(request)


def _require_writes() -> None:
    if not security.writes_enabled():
        raise HTTPException(status_code=403, detail="writes are disabled: set DASH_ALLOW_WRITES=true")


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
    _require_writes()                       # starting trading again is a write; stopping is not
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
    _require_writes()
    try:
        entry = dp.promote(ROOT, rank)
    except dp.PromoteError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "rank": rank, "enabled": entry["enabled"],
            "symbol": entry["params"]["only_symbol"], "timeframe": entry["params"]["timeframe"]}


@app.get("/api/prop")
def prop_get():
    """Active prop-firm profile and its editable rules."""
    return prop_edit.get_active(ROOT)


@app.put("/api/prop")
def prop_put(body: dict = Body(...)):
    """Save changes to the active profile's rules. A refused value returns 409 and saves nothing."""
    _require_writes()
    try:
        return prop_edit.save_rules(ROOT, body.get("rules") or {})
    except PropError as e:
        raise HTTPException(status_code=409, detail=str(e))
