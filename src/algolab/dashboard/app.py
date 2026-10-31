"""Dashboard. It reads state/ and reports/ and never talks to the broker or places orders. The writes are the
Promote button and the Trade Setup popup (HTF, LTF, risk, symbols into config/). Both need DASH_ALLOW_WRITES=true.

Run:  algolab dashboard        (then open http://127.0.0.1:8050)
Env:  DASH_PASSWORD (required: every request needs it), DASH_HOST (default 127.0.0.1), DASH_PORT (default 8050),
      DASH_ALLOW_WRITES (default false)"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from flask import Response, request

from algolab.dashboard import security
from algolab.dashboard import trade_setup as ts

from algolab.memory import store as memory
from algolab.ops import journal
from algolab.ops.state import STATE_DIR
from algolab.profile import ROOT

REPORTS = ROOT / "reports"
CARD = {"display": "inline-block", "minWidth": "190px", "margin": "8px", "padding": "14px",
        "borderRadius": "10px", "background": "#161b22", "border": "1px solid #30363d",
        "verticalAlign": "top"}
PAGE = {"background": "#0d1117", "color": "#e6edf3", "fontFamily": "system-ui, sans-serif",
        "minHeight": "100vh", "padding": "18px"}
TH = {"textAlign": "left", "padding": "6px 10px", "borderBottom": "1px solid #30363d"}


def _j(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def snapshot() -> dict:
    live = _j(STATE_DIR / "live_last_cycle.json", {})
    health = _j(STATE_DIR / "health.json", {})
    tokens = _j(STATE_DIR / "tokens.json", {})
    validation = _j(REPORTS / "validation.json", {})
    autotune = _j(REPORTS / "autotune_latest.json", {})
    strategy = _j(ROOT / "config" / "strategy.json", {})
    llm_cfg = _j(ROOT / "config" / "llm.json", {"daily_token_budget": 0})
    recent = []
    for f in sorted(glob.glob(str(REPORTS / "cycle_*.json")))[-20:]:
        for hit in _j(Path(f), {}).get("take", []):
            recent.append({"time": hit.get("time"), "symbol": hit.get("symbol"),
                           "side": hit.get("side"), "rsi": hit.get("rsi"),
                           "confidence": hit.get("confidence"), "reason": hit.get("reason")})
    # Live-trading results (legacy engulfing path) appear beside the alert-only scans.
    for r in (live.get("results") or [])[::-1]:
        recent.insert(0, {"time": live.get("ts"), "symbol": r.get("symbol"), "side": r.get("side"),
                          "rsi": "", "confidence": "", "reason": f'{"EXECUTED" if r.get("executed") else "dry run"}: {r.get("reason", "")}'})
    return {"live": live, "health": health, "tokens": tokens, "validation": validation,
            "autotune": autotune, "strategy": strategy,
            "token_limit": llm_cfg.get("daily_token_budget", 0),
            "kill": (STATE_DIR / "KILL").exists(), "recent": recent[::-1][:50],
            "framework": _j(STATE_DIR / "framework_last_cycle.json", {}),
            "journal": journal.read(), "sealed": _j(STATE_DIR / "sealed_result.json", {}),
            "calibration": memory.recall("calibration", 200),
            "mistakes": memory.recall("mistake", 20), "working": memory.recall("working", 5),
            "discovery": _j(REPORTS / "discovery.json", {})}


def _card(title: str, value, note: str = "") -> html.Div:
    return html.Div([html.Div(title, style={"opacity": 0.7, "fontSize": "12px"}),
                     html.Div(str(value), style={"fontSize": "22px", "fontWeight": 600}),
                     html.Div(note, style={"opacity": 0.6, "fontSize": "12px"})], style=CARD)


def _table(rows: list[dict], cols: list[str]):
    if not rows:
        return html.Div("no data yet", style={"opacity": 0.6})
    head = html.Tr([html.Th(c, style=TH) for c in cols])
    body = [html.Tr([html.Td(str(r.get(c, "")), style={"padding": "6px 10px"}) for c in cols])
            for r in rows]
    return html.Table([html.Thead(head), html.Tbody(body)], style={"width": "100%"})


def overview(s: dict) -> list:
    live, val, tok, at = s["live"], s["validation"], s["tokens"], s["autotune"]
    may = live.get("may_trade")
    used = tok.get("used", 0) if tok.get("day") else 0
    st = s["strategy"]
    return [
        _card("Trading", "HALTED" if s["kill"] or may is False else ("ON" if may else "NO DATA"),
              live.get("reason", "no cycle yet")),
        _card("Kill switch", "ENGAGED" if s["kill"] else "off",
              "delete state/KILL to resume" if s["kill"] else ""),
        _card("Equity", f'{live["equity"]:,.2f}' if "equity" in live else "-",
              f'open positions: {live.get("open_positions", "-")}'),
        _card("Strategy validation", "PASSED" if val.get("passed") else "NOT PASSED",
              "; ".join(val.get("reasons", [])) or "run: algolab validate"),
        _card("Live parameters", f"v{st.get('version', '-')}",
              f"RSI {st.get('rsi_n')} EMA {st.get('ema_n')} "
              f"SL {st.get('sl_mult')}xATR TP {st.get('tp_mult')}xATR ML {st.get('ml_threshold')}"),
        _card("Last autotune", "APPLIED" if at.get("applied") else
              ("ROLLED BACK" if at.get("rolled_back") else ("kept" if at else "never")),
              (at.get("reasons") or ["-"])[0]),
        _card("LLM tokens today", f"{used:,} / {s['token_limit']:,}",
              f'calls {tok.get("calls", 0)}, cached {tok.get("cached", 0)}'),
        _card("Mode", "DRY RUN" if live.get("dry_run", True) else "LIVE",
              f'last cycle {live.get("ts", "-")}'),
    ]


def health_table(s: dict) -> list:
    checks = s["health"].get("checks", {})
    rows = [{"check": k, "status": "OK" if v["ok"] else "FAIL", "detail": v["detail"]}
            for k, v in checks.items()]
    return [html.H4("Health (VPS checks)"), _table(rows, ["check", "status", "detail"])]


def signals_table(s: dict) -> list:
    return [html.H4("Recent signals"),
            _table(s["recent"], ["time", "symbol", "side", "rsi", "confidence", "reason"])]


def trades_panel(s: dict) -> list:
    rows = s["journal"][-60:][::-1]
    table = _table([{"time": r.get("ts", ""), "kind": r.get("kind", ""),
                     "source": r.get("source", r.get("reason", "")), "strategy": r.get("strategy", ""),
                     "symbol": r.get("symbol", ""), "side": r.get("side", ""),
                     "entry": r.get("entry", r.get("price", "")), "sl": r.get("sl", ""),
                     "pnl": round(float(r["pnl"]), 2) if r.get("pnl") is not None
                     else ("unknown" if r.get("kind") == "close" else ""),
                     "R": round(float(r["r"]), 2) if r.get("r") is not None else ""}
                    for r in rows],
                   ["time", "kind", "source", "strategy", "symbol", "side", "entry", "sl", "pnl", "R"])
    # Only trades with a known broker P&L are charted; unknown ones are shown in the table as "unknown".
    closes = [r for r in s["journal"] if r.get("kind") == "close"]
    known = [r for r in closes if r.get("r") is not None]
    fig = go.Figure()
    if known:
        cum, total = [], 0.0
        for r in known:
            total += float(r["r"])
            cum.append(total)
        fig.add_trace(go.Scatter(y=cum, mode="lines+markers", name="cumulative R"))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Closed trades: {len(closes)} ({len(known)} with known P&L, cumulative R)")
    return [html.H4("Trades (journal)"), dcc.Graph(figure=fig, config={"displayModeBar": False}), table]


def framework_panel(s: dict) -> list:
    fw = s["framework"]
    if not fw:
        return [html.Div("framework has not run yet (enable it in config/framework.json)", style={"opacity": 0.6})]
    res = fw.get("framework", {}) if isinstance(fw.get("framework"), dict) else {}
    results = res.get("results", [])
    events = res.get("events", [])
    prop = res.get("prop")
    rows = [{"kind": "result", "detail": f'{r.get("strategy")} {r.get("symbol")} {r.get("side", "")} '
             f'{"EXECUTED" if r.get("executed") else r.get("why", "not placed")}'} for r in results]
    rows += [{"kind": "closed", "detail": f'{e.get("strategy")} {e.get("symbol")} {e.get("reason")} at {e.get("price")}'}
             for e in events]
    fw_cfg = _j(ROOT / "config" / "framework.json", {})
    ok_gate, gate_why = journal.entry_gate(fw_cfg.get("entry_gate"))
    gate = f'entry gate: {"open" if ok_gate else "BLOCKING"} - {gate_why}'
    return [html.H4("Framework (last cycle)"),
            html.Div(f"may trade: {fw.get('may_trade')} | {fw.get('reason', '')} | equity {fw.get('equity', '-')}"),
            html.Div(f"prop: {prop}" if prop else "prop rules: off", style={"opacity": 0.7}),
            _table(rows, ["kind", "detail"]),
            html.Div(gate, style={"opacity": 0.6})]


def agent_panel(s: dict) -> list:
    gap = memory.calibration_gap(s["calibration"])
    sealed = s["sealed"]
    cards = [_card("Confidence calibration",
                   "n/a" if not gap else f'{gap["mean_predicted"]:.2f} vs {gap["realised_win_rate"]:.2f}',
                   "no calibration records yet" if not gap else f'{gap["n"]} trades, gap {gap["gap"]:+.2f}'),
             _card("Sealed test", "not run" if not sealed else ("PASSED" if sealed.get("passed") else "FAILED"),
                   "; ".join(sealed.get("reasons", [])) if sealed else "run: algolab sealed"),
             _card("Holdout reuse", "-", (s["autotune"].get("reasons") or ["-"])[0])]
    mistakes = [{"ts": m.get("ts", ""), "mistake": m.get("what", m.get("summary", str(m))),
                 "rule": m.get("rule", "")} for m in s["mistakes"]][::-1]
    working = [{"ts": w.get("ts", ""), "note": w.get("hypothesis", str(w))} for w in s["working"]][::-1]
    return [html.Div(cards, style={"display": "flex", "flexWrap": "wrap"}),
            html.H4("Recent mistakes and rules"), _table(mistakes, ["ts", "mistake", "rule"]),
            html.H4("Working notes"), _table(working, ["ts", "note"])]


def tokens_panel(s: dict) -> list:
    tok = s["tokens"]
    used = tok.get("used", 0) if tok.get("day") else 0
    limit = s["token_limit"] or 1
    fig = go.Figure(go.Bar(x=["used", "left"], y=[used, max(limit - used, 0)],
                           marker_color=["#f78166", "#3fb950"]))
    fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=30, b=10),
                      title="LLM token budget today")
    return [dcc.Graph(figure=fig, config={"displayModeBar": False})]


def opt_figure(metric: str) -> go.Figure:
    rep = _j(REPORTS / "autotune_latest.json", {})
    trials = rep.get("trials") or []
    if not trials:
        return go.Figure().update_layout(template="plotly_dark", height=620,
                                         title="No tuning run yet: algolab optimize")
    df = pd.DataFrame(trials)
    text = [f"score {r.score}<br>trades {r.trades}<br>RSI {r.rsi_n} EMA {r.ema_n} ATR {r.atr_n}"
            f"<br>ML {r.ml_threshold}" for r in df.itertuples()]
    fig = go.Figure(go.Scatter3d(
        x=df["tp_mult"], y=df["sl_mult"], z=df[metric], mode="markers", text=text,
        hoverinfo="text",
        marker=dict(size=5, color=df[metric], colorscale="Viridis", showscale=True)))
    fig.update_layout(template="plotly_dark", height=620, margin=dict(l=0, r=0, t=40, b=0),
                      title=f"Autotune trials: {len(df)} completed",
                      scene=dict(xaxis_title="TP mult (x ATR)", yaxis_title="SL mult (x ATR)",
                                 zaxis_title=metric))
    return fig


def _num(x, nd: int = 2):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return ""


GATE_LABELS = [("min_pf", "Validation PF >="), ("min_beat_random", "Beat random >="),
               ("min_nudge", "Nudge ok >="), ("max_dd", "Monte Carlo DD95 <= %"),
               ("min_train_trades", "Min trades: train"), ("min_val_trades", "Min trades: validation"),
               ("min_test_trades", "Min trades: hold-out")]


def bank_section() -> list:
    """Candidate bank: every candidate from the last run, in rank order, with a one-click Promote button."""
    from algolab import discovery_promote as dp
    rows = dp.bank_rows(ROOT)
    if not rows:
        return [html.H4("Candidate bank"), html.Div("No candidates to promote.", style={"opacity": 0.6})]
    td = {"padding": "6px 10px"}
    cols = ["rank", "symbol", "tf", "family", "score", "hold-out PF", "trades", "status", ""]
    head = html.Tr([html.Th(c, style=TH) for c in cols])
    body = []
    for r in rows:
        btn = html.Button("Promote", id={"type": "promote-row", "rank": r["rank"]}, n_clicks=0,
                          disabled=not r["can_promote"], style=NAV_BTN)
        body.append(html.Tr([
            html.Td(str(r["rank"]), style=td), html.Td(r["symbol"] or "", style=td),
            html.Td(r["tf"] or "", style=td), html.Td(r["family"] or "", style=td),
            html.Td(_num(r["score"], 1), style=td), html.Td(_num(r["holdout_pf"]), style=td),
            html.Td(str(r["trades"] if r["trades"] is not None else ""), style=td),
            html.Td(r["status"], style=td), html.Td(btn, style=td),
        ]))
    return [html.H4("Candidate bank (one click adds a DISABLED entry)"),
            html.Table([html.Thead(head), html.Tbody(body)], style={"width": "100%"}),
            html.Div("Enabling stays manual: run algolab validate, then set enabled to true in framework.json.",
                     style={"opacity": 0.6, "marginTop": "6px"})]


def discovery_panel(s: dict) -> list:
    """Shows the last discovery run (reports/discovery.json, written by algolab.discovery_factory).
    Read-only. Gate values come from that run's saved settings, so nothing is hard-coded here."""
    rep = s.get("discovery") or {}
    if not rep:
        return [html.Div("No discovery run yet. Run: python -m algolab.discovery_factory "
                         "(add --demo to try it on synthetic data).", style={"opacity": 0.7})]
    cfg = rep.get("settings", {})
    cands = rep.get("candidates") or []
    cards = [
        _card("Candidates passing every gate", len(cands), rep.get("note", "")),
        _card("Last run", (rep.get("generated_at") or "-")[:16].replace("T", " "), "UTC"),
        _card("Applied to live", "NO" if not rep.get("applied") else "YES",
              "report only: a person promotes a candidate by hand"),
    ]
    funnel_rows = [{k: f.get(k, "") for k in ("dataset", "candidates", "val", "rand", "mc", "nudge", "holdout")}
                   for f in (rep.get("funnel") or [])]
    gate_rows = [{"gate": label, "value": cfg.get(key, "-")} for key, label in GATE_LABELS]
    cand_rows = []
    for i, r in enumerate(cands, 1):
        t = r.get("test") or {}
        cand_rows.append({"rank": i, "symbol": r.get("symbol"), "tf": r.get("tf"),
                          "family": (r.get("genome") or {}).get("fam"), "score": _num(r.get("score"), 1),
                          "trades": r.get("trades"), "hold-out PF": _num(t.get("pf")),
                          "exp R": _num(t.get("exp"), 3), "win %": _num(t.get("wr"), 1),
                          "MC DD95 %": _num(r.get("mc_dd95"), 1), "nudge": r.get("nudge"),
                          "rule": r.get("desc", "")})
    return [html.Div(cards, style={"display": "flex", "flexWrap": "wrap"}),
            *bank_section(),
            html.H4("Gate funnel (genomes reaching each stage)"),
            _table(funnel_rows, ["dataset", "candidates", "val", "rand", "mc", "nudge", "holdout"]),
            html.H4("Gate rules (from the last run's settings)"),
            _table(gate_rows, ["gate", "value"]),
            html.H4("Top candidates"),
            _table(cand_rows, ["rank", "symbol", "tf", "family", "score", "trades", "hold-out PF",
                               "exp R", "win %", "MC DD95 %", "nudge", "rule"]),
            html.Div("An empty result is a valid result. Never loosen gates to force a winner.",
                     style={"opacity": 0.6, "marginTop": "8px"})]


NAV_BTN = {"background": "transparent", "color": "#3fd0ff", "border": "1px solid #30363d",
           "borderRadius": "8px", "padding": "6px 14px", "cursor": "pointer", "font": "inherit"}

PILL = {"fontSize": "12px", "padding": "4px 10px", "borderRadius": "999px", "background": "#161b22",
        "border": "1px solid #30363d", "color": "#e6edf3"}
LABEL = {"fontSize": "12px", "opacity": 0.7, "margin": "12px 0 4px"}
INPUT = {"background": "#0d1117", "color": "#e6edf3", "border": "1px solid #30363d", "borderRadius": "8px",
         "padding": "8px 10px", "width": "100%", "boxSizing": "border-box"}
SAVE_BTN = {**NAV_BTN, "background": "#1f6feb", "color": "#ffffff", "border": "0"}
X_BTN = {"background": "none", "border": "0", "color": "#8b98a5", "cursor": "pointer", "padding": "0 2px"}
MODAL_BASE = {"position": "fixed", "inset": "0", "background": "rgba(0,0,0,.6)", "zIndex": 50,
              "alignItems": "center", "justifyContent": "center", "padding": "12px"}
MODAL_SHOW = {**MODAL_BASE, "display": "flex"}
MODAL_HIDE = {**MODAL_BASE, "display": "none"}


def _setup_or_none() -> dict | None:
    """The current trade setup, or None when a config file cannot be read (the header says so)."""
    try:
        return ts.read_setup()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def setup_chips(s: dict | None) -> list:
    if not s:
        return [html.Span("setup unreadable", style={**PILL, "borderColor": "#f59e0b", "color": "#f59e0b"})]
    return [html.Span(t, style=PILL) for t in (f"HTF {s['htf']}", f"LTF {s['ltf']}",
                                               f"Risk {s['risk_pct']}%", f"{len(s['symbols'])} symbols")]


def trade_setup_modal() -> html.Div:
    """Trade setup popup: HTF, LTF, risk and symbols only. Saved values go to config/ on Save."""
    return html.Div(id="setup-modal", style=MODAL_HIDE, children=[
        html.Div(style={"width": "100%", "maxWidth": "520px", "maxHeight": "92vh", "overflow": "auto",
                        "background": "#161b22", "border": "1px solid #30363d", "borderRadius": "14px",
                        "padding": "16px", "color": "#e6edf3"}, children=[
            html.Div([html.H3("Trade Setup", style={"margin": "0", "flex": "1"}),
                      html.Button("×", id="setup-x", n_clicks=0, style=X_BTN)],
                     style={"display": "flex", "alignItems": "center"}),
            html.Div("HTF (signal timeframe)", style=LABEL),
            dcc.RadioItems(id="setup-htf", options=list(ts.HTF_CHOICES), inline=True,
                           inputStyle={"marginRight": "4px", "marginLeft": "12px"}),
            html.Div("LTF (entry timeframe)", style=LABEL),
            dcc.RadioItems(id="setup-ltf", options=list(ts.LTF_CHOICES), inline=True,
                           inputStyle={"marginRight": "4px", "marginLeft": "12px"}),
            html.Div(f"Risk per trade, % of equity (max as set by the daily loss limit, up to "
                     f"{ts.RISK_ABS_MAX_PCT}%)", style=LABEL),
            dcc.Input(id="setup-risk", type="number", min=ts.RISK_MIN_PCT, max=ts.RISK_ABS_MAX_PCT,
                      step=0.1, style=INPUT),
            html.Div("Symbols", style=LABEL),
            html.Div([dcc.Input(id="setup-sym-input", type="text", placeholder="e.g. EURUSD",
                                style={**INPUT, "flex": "1"}),
                      html.Button("Add", id="setup-sym-add", n_clicks=0, style=NAV_BTN)],
                     style={"display": "flex", "gap": "8px"}),
            html.Div(id="setup-sym-error", style={"color": "#ef4444", "fontSize": "12px", "minHeight": "14px"}),
            html.Div(id="setup-sym-list", style={"display": "flex", "flexWrap": "wrap", "gap": "6px"}),
            dcc.Store(id="setup-symbols", data=[]),
            html.Div(id="setup-status", style={"marginTop": "12px", "fontSize": "13px"}),
            html.Div([html.Button("Cancel", id="setup-cancel", n_clicks=0, style=NAV_BTN),
                      html.Button("Save", id="setup-save", n_clicks=0, style=SAVE_BTN)],
                     style={"display": "flex", "justifyContent": "flex-end", "gap": "8px", "marginTop": "14px"}),
        ])])


app = Dash(__name__, title="AlgoLab")


@app.server.before_request
def require_login():
    """Every request, including the Dash callbacks, needs the dashboard password."""
    if not security.configured_password():
        return Response("dashboard locked: set DASH_PASSWORD", 503)
    if not security.is_authorized(request.headers.get("Authorization")):
        return Response("login required", 401, {"WWW-Authenticate": security.REALM})
    return None


app.layout = html.Div(style=PAGE, children=[
    html.Div([html.H2("AlgoLab", style={"margin": "0"}),
              html.Div(id="setup-chips", style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "flex": "1"}),
              html.Button("Trade Setup", id="nav-setup", n_clicks=0, style=NAV_BTN),
              html.Button("Discovery", id="nav-discovery", n_clicks=0, style=NAV_BTN)],
             style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "12px"}),
    trade_setup_modal(),
    dcc.Interval(id="tick", interval=15_000),
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview"),
        dcc.Tab(label="Signals", value="signals"),
        dcc.Tab(label="Health", value="health"),
        dcc.Tab(label="Tokens", value="tokens"),
        dcc.Tab(label="Trades", value="trades"),
        dcc.Tab(label="Framework", value="framework"),
        dcc.Tab(label="Agent", value="agent"),
        dcc.Tab(label="Discovery", value="discovery"),
    ]),
    html.Div(id="promote-status", style={"marginTop": "8px", "opacity": 0.85}),
    html.Div(id="tab-body", style={"marginTop": "12px"}),
    html.H3("Autotune (3D)"),
    dcc.Dropdown(id="metric", value="score", options=["score", "trades"], clearable=False,
                 style={"color": "#000", "width": "280px"}),
    dcc.Graph(id="opt-3d"),
])


@app.callback(Output("setup-modal", "style"), Output("setup-htf", "value"), Output("setup-ltf", "value"),
              Output("setup-risk", "value"), Output("setup-symbols", "data"),
              Output("setup-status", "children", allow_duplicate=True),
              Output("setup-sym-error", "children", allow_duplicate=True),
              Input("nav-setup", "n_clicks"), Input("setup-x", "n_clicks"), Input("setup-cancel", "n_clicks"),
              prevent_initial_call=True)
def open_setup(_open, _x, _cancel):
    """Opens the popup filled from config/, or closes it. Opening never writes anything."""
    if ctx.triggered_id != "nav-setup":
        return MODAL_HIDE, no_update, no_update, no_update, no_update, "", ""
    s = _setup_or_none()
    if s is None:
        return MODAL_SHOW, no_update, no_update, no_update, [], "Could not read config/ files.", ""
    return MODAL_SHOW, s["htf"], s["ltf"], s["risk_pct"], s["symbols"], "", ""


@app.callback(Output("setup-symbols", "data", allow_duplicate=True),
              Output("setup-sym-input", "value"),
              Output("setup-sym-error", "children", allow_duplicate=True),
              Input("setup-sym-add", "n_clicks"),
              Input({"type": "setup-sym-remove", "sym": ALL}, "n_clicks"),
              State("setup-sym-input", "value"), State("setup-symbols", "data"),
              prevent_initial_call=True)
def edit_symbols(_add, _remove, raw, syms):
    """Adds the typed symbol or removes a chip. Nothing is written to disk here; Save does that."""
    syms = list(syms or [])
    tid = ctx.triggered_id
    if isinstance(tid, dict) and tid.get("type") == "setup-sym-remove":
        if not ctx.triggered[0].get("value"):      # chips re-rendering, not a click
            return no_update, no_update, no_update
        return [x for x in syms if x != tid["sym"]], no_update, ""
    if tid == "setup-sym-add":
        out, err = ts.add_symbol(syms, raw or "")
        if err:
            return no_update, no_update, err
        return out, "", ""
    return no_update, no_update, no_update


@app.callback(Output("setup-sym-list", "children"), Input("setup-symbols", "data"))
def render_setup_symbols(syms):
    if not syms:
        return html.Span("No symbols yet", style={"opacity": 0.6, "fontSize": "13px"})
    return [html.Span([s, html.Button("×", id={"type": "setup-sym-remove", "sym": s}, n_clicks=0,
                                      style=X_BTN)],
                      style={**PILL, "display": "inline-flex", "alignItems": "center", "gap": "6px"})
            for s in syms]


@app.callback(Output("setup-status", "children"),
              Input("setup-save", "n_clicks"),
              State("setup-htf", "value"), State("setup-ltf", "value"),
              State("setup-risk", "value"), State("setup-symbols", "data"),
              prevent_initial_call=True)
def save_setup_click(_n, htf, ltf, risk, syms):
    """Writes the four config files. Refused unless DASH_ALLOW_WRITES=true."""
    try:
        res = ts.save_setup({"htf": htf, "ltf": ltf, "risk_pct": risk, "symbols": syms or []},
                            allow_writes=security.writes_enabled())
    except ts.SetupError as e:
        return f"Not saved: {e}"
    msg = f"Saved: {', '.join(res['written'])}."
    if res["needs_validate"]:
        msg += " HTF, LTF or risk changed, so the last validation is stale. Run algolab validate before trading resumes."
    return msg


@app.callback(Output("setup-chips", "children"), Input("tick", "n_intervals"))
def render_setup_chips(_):
    return setup_chips(_setup_or_none())


@app.callback(Output("promote-status", "children"),
              Input({"type": "promote-row", "rank": ALL}, "n_clicks"), prevent_initial_call=True)
def promote_row(_clicks):
    """One click on a candidate row adds it to framework.json as a DISABLED entry. Never enables it.
    Disabled unless DASH_ALLOW_WRITES=true."""
    from algolab import discovery_promote as dp
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return no_update                        # a re-render of the table, not a click
    if not security.writes_enabled():
        return "Promote is disabled. Set DASH_ALLOW_WRITES=true to allow it."
    rank = ctx.triggered_id["rank"]
    try:
        dp.promote(ROOT, int(rank))
    except dp.PromoteError as e:
        return f"Candidate {rank} refused: {e}"
    return (f"Candidate {rank} added to framework.json as DISABLED. Run algolab validate, "
            "then enable it by hand.")


@app.callback(Output("tabs", "value"), Input("nav-discovery", "n_clicks"), prevent_initial_call=True)
def open_discovery(_):
    return "discovery"


@app.callback(Output("tab-body", "children"), Input("tabs", "value"), Input("tick", "n_intervals"))
def render_tab(tab: str, _):
    s = snapshot()
    if tab == "discovery":
        return discovery_panel(s)
    if tab == "signals":
        return signals_table(s)
    if tab == "health":
        return health_table(s)
    if tab == "tokens":
        return tokens_panel(s)
    if tab == "trades":
        return trades_panel(s)
    if tab == "framework":
        return framework_panel(s)
    if tab == "agent":
        return agent_panel(s)
    return [html.Div(overview(s), style={"display": "flex", "flexWrap": "wrap"})]


@app.callback(Output("opt-3d", "figure"), Input("metric", "value"), Input("tick", "n_intervals"))
def render_opt(metric: str, _):
    return opt_figure(metric)


def main() -> None:
    security.require_password_for_start()   # refuse to start unlocked, even on localhost
    app.run(host=os.getenv("DASH_HOST", "127.0.0.1"), port=int(os.getenv("DASH_PORT", "8050")))


if __name__ == "__main__":
    main()
