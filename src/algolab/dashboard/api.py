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
from algolab.dashboard import control, panels, prop_edit, security
from algolab.dashboard import trade_setup as ts
from algolab.dashboard.panels import JobBusy
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


@app.get("/api/setup")
def setup_get():
    """Trade setup: HTF, LTF, risk per trade and symbols, read from config/."""
    try:
        return ts.read_setup(ROOT)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=500, detail=f"could not read config: {e}")


@app.put("/api/setup")
def setup_put(body: dict = Body(...)):
    """Validate and save HTF, LTF, risk and symbols. An invalid value returns 409 and writes nothing."""
    _require_writes()
    try:
        return ts.save_setup(body, ROOT, allow_writes=True)
    except ts.SetupError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ---------- monitor panels (read-only, except the validate job) ----------

@app.get("/api/status")
def status_panel():
    return panels.status(ROOT)


@app.get("/api/trades")
def trades_panel(limit: int = 100):
    return {"rows": panels.trades(max(1, min(limit, 500)))}


@app.get("/api/positions")
def positions_panel():
    return {"rows": panels.open_positions()}


@app.get("/api/risk")
def risk_panel():
    return panels.risk(ROOT)


@app.get("/api/signals")
def signals_panel(limit: int = 50):
    return {"rows": panels.signals(max(1, min(limit, 200)), ROOT)}


@app.get("/api/equity")
def equity_panel(limit: int = 500):
    return {"rows": panels.equity(max(1, min(limit, 5000)))}


@app.get("/api/alerts")
def alerts_panel():
    return {"rows": panels.alerts()}


@app.get("/api/autotune")
def autotune_panel():
    return panels.autotune(ROOT)


@app.get("/api/tokens")
def tokens_panel():
    return panels.tokens(ROOT)


@app.get("/api/validate/job")
def validate_job_panel():
    return panels.validate_job()


@app.post("/api/validate")
def validate_start():
    """Runs `algolab validate` in the background. Needs DASH_ALLOW_WRITES=true. One run at a time."""
    _require_writes()
    try:
        return panels.start_validate(ROOT, background=True)
    except JobBusy as e:
        raise HTTPException(status_code=409, detail=str(e))


def main() -> None:
    """Starts the dashboard (used by `algolab dashboard` and the Docker service). Refuses to start without
    DASH_PASSWORD. Host and port come from DASH_HOST (default 127.0.0.1) and DASH_PORT (default 8050)."""
    import os
    import uvicorn
    from algolab.dashboard import security as sec
    sec.require_password_for_start()
    uvicorn.run(app, host=os.getenv("DASH_HOST", "127.0.0.1"), port=int(os.getenv("DASH_PORT", "8050")))


if __name__ == "__main__":
    main()
