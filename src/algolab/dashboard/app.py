"""Dashboard. Read-only: it reads state/ and reports/ and never talks to the broker or places orders.

Run:  algolab dashboard        (then open http://127.0.0.1:8050)
Env:  DASH_HOST (default 127.0.0.1), DASH_PORT (default 8050)"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

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
            "mistakes": memory.recall("mistake", 20), "working": memory.recall("working", 5)}


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
              f"RSI {st.get('rsi_n')} OS {st.get('oversold')} OB {st.get('overbought')} "
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
                     "pnl": round(r["pnl"], 5) if "pnl" in r else "", "R": round(r["r"], 2) if "r" in r else ""}
                    for r in rows],
                   ["time", "kind", "source", "strategy", "symbol", "side", "entry", "sl", "pnl", "R"])
    closes = [r for r in s["journal"] if r.get("kind") == "close"]
    fig = go.Figure()
    if closes:
        cum, total = [], 0.0
        for r in closes:
            total += float(r["r"])
            cum.append(total)
        fig.add_trace(go.Scatter(y=cum, mode="lines+markers", name="cumulative R"))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Closed trades: {len(closes)} (cumulative R)")
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
    text = [f"score {r.score}<br>trades {r.trades}<br>RSI {r.rsi_n} ATR {r.atr_n}"
            f"<br>OS {round(r.oversold, 1)} OB {round(r.overbought, 1)}"
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


app = Dash(__name__, title="AlgoLab")
app.layout = html.Div(style=PAGE, children=[
    html.H2("AlgoLab"),
    dcc.Interval(id="tick", interval=15_000),
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview"),
        dcc.Tab(label="Signals", value="signals"),
        dcc.Tab(label="Health", value="health"),
        dcc.Tab(label="Tokens", value="tokens"),
        dcc.Tab(label="Trades", value="trades"),
        dcc.Tab(label="Framework", value="framework"),
        dcc.Tab(label="Agent", value="agent"),
    ]),
    html.Div(id="tab-body", style={"marginTop": "12px"}),
    html.H3("Autotune (3D)"),
    dcc.Dropdown(id="metric", value="score", options=["score", "trades"], clearable=False,
                 style={"color": "#000", "width": "280px"}),
    dcc.Graph(id="opt-3d"),
])


@app.callback(Output("tab-body", "children"), Input("tabs", "value"), Input("tick", "n_intervals"))
def render_tab(tab: str, _):
    s = snapshot()
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
    app.run(host=os.getenv("DASH_HOST", "127.0.0.1"), port=int(os.getenv("DASH_PORT", "8050")))


if __name__ == "__main__":
    main()
