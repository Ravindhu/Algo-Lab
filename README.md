# AlgoLab

Forex trading system for MetaTrader 5. It scans higher-timeframe engulfing signals with an RSI filter, checks them with a local LLM, sizes trades from ATR stops, and runs unattended. It runs on Windows, on an Ubuntu VPS, or on both.

Dependencies come from PyPI packages only. The broker connection uses the official `MetaTrader5` package directly. No third-party wrapper, and no GitHub repository is needed.

## Install

**Windows (this PC)**
```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1 -Mode Standalone
```
Use `-Mode Vps` to also sync data and account state to a VPS.

**Ubuntu VPS**
```bash
sudo bash install_server.sh <YOUR_TRUSTED_IP>
sudo bash deploy/install_ops.sh
```

Then, on either machine:
1. Edit `.env` (MT5 login, Ollama URL, Telegram token or webhook). Keep `DRY_RUN=true`.
2. `algolab doctor` (every line should say OK, except possibly the DRY_RUN line)
3. `algolab validate` (the strategy must pass before the live cycle trades)
4. `algolab dashboard`, then open http://127.0.0.1:8050

## Schedule

| Job | Command | Every |
|---|---|---|
| Framework cycle (scalp, engulfing, OCL, prop rules) | `python -m algolab.live.pipeline --framework` | 5 min |
| Alert-only scan (VPS) | `python -m algolab.ops.pipeline` | 15 min |
| Autotune | `algolab optimize` | on demand (never automatic; a person approves with `--apply`) |

Both live jobs run only after the same guards pass: kill switch, validation, fresh account state, and loss limits.

## Daily commands

| Command | What it does |
|---|---|
| `algolab status` | Kill switch, last cycle, health, token use |
| `algolab select` | Strategy ranking (report only). `algolab select --apply` enables or disables strategies after your review |
| `algolab validate` | Out-of-sample gate on real history (legacy rule + every enabled framework strategy, replayed). Required before live trading. Re-run after any settings change |
| `algolab optimize` | Parameter search. Report only; `algolab optimize --apply` applies a candidate after you have reviewed it |
| `algolab backtest` | Run the live rule on history |
| `algolab kill` | Stop all trading now |
| `algolab resume` | Allow trading again after you have reviewed the cause |
| `algolab dashboard` | Web dashboard (read-only): overview, signals, trades, framework, agent |
| `algolab sealed` | One-time test on the sealed final 10% of history. The optimizer never sees it |
| `algolab doctor` | Setup checks. Includes a drift check between strategy.json and framework.json |

## How the safety works

- **Fail closed.** The live cycle trades only if the kill switch is off, the strategy validation passed and is recent, the account state is fresh, the market data is fresh, and the loss limits are not hit.
- **Validation gate.** `algolab validate` tests the live parameters on several folds of history, without fitting. It requires enough trades, a profit factor above the threshold, positive expectancy, and most folds profitable.
- **Kill switch.** `algolab kill` or create `state/KILL`. Drawdown beyond the limit engages it automatically.
- **Entry gate.** Framework entries need the lower bound of the recent win rate (closed trades in `state/journal.jsonl`) to clear the breakeven level for the target RR. A bounded warm-up lets the first trades happen before any closes exist.
- **Holdout discipline.** Each optimizer selection on the same holdout window is counted. After `max_holdout_uses`, the candidate is not applied until the sealed test is run.
- **LLM token control.** A daily token budget, a cache per candidate, and compact payloads. When the budget is spent, the analyst skips trades. Usage is on the dashboard.
- **Settings are bound to validation.** `validate` records a fingerprint of every setting that affects trading. A changed strategy, parameter, cost or enabled set makes the report stale, and trading stops until you validate again.
- **Every enabled framework strategy must be replayed.** The replay runs the live code path on real M5/M15 data (never synthetic data). A strategy the replay cannot test fails the gate, so it cannot trade.
- **Broker P&L only.** Closed trades are booked at the broker's realised profit (commission and swap included). Unknown P&L is recorded as unknown and excluded from the entry gate and from strategy selection. It is never estimated.
- **Fills are recorded at once.** Each fill is tracked and journaled the moment it happens. A position the broker holds that nothing tracks blocks new entries and raises a CRITICAL alert.
- **One cycle at a time.** A lock file stops overlapping runs from trading together. Traded signal keys persist, so a restart cannot trade the same bar twice.
- **Margin and volume.** Lots are snapped to the broker's volume step, and an order is skipped if its margin exceeds `max_margin_use` of free margin.
- **Deposits are not losses.** Money moved in or out is removed before the drawdown and daily-loss limits are measured.
- **No unattended changes.** The optimizer and the selector never change live settings on their own. Approval is explicit (`--apply`).
- **DRY_RUN.** Alerts only, no orders, until you set `DRY_RUN=false`.

## What you need

- Real history CSVs in `data/`: `{SYMBOL}_H4.csv` (legacy rule) and `{SYMBOL}_M5.csv` plus `{SYMBOL}_M15.csv` (framework replay). Export them with the Windows sync or from your broker. Without them, `validate` fails on purpose.
- A Windows machine running MT5 and a **demo** account for the live adapter. Check the broker's filling modes and volume steps in the terminal first.
- `config/symbols.json` (export with `scripts/export_symbols.py`), or the symbol list in `config/ops.json`.
- Optional: Ollama for the LLM analyst, a Telegram token or webhook for alerts, and `ANTHROPIC_API_KEY` for the advisor.

## Accuracy, honestly

No system guarantees accuracy or profit. The validation gate reduces the chance of trading parameters that only worked on past data. It does not predict the future. Run on a demo account for several weeks, read the dashboard daily, and only then consider `DRY_RUN=false`.

## Layout

```
src/algolab/
  cli.py                 `algolab` command
  profile.py             loads config/profiles and .env
  strategy.py            live strategy parameters (config/strategy.json)
  hilega_milega.py       legacy HM math (kept for live/positions.py); strategies use indicators/hm
  signals/               back-compat shim only (scanner + engulfing names re-exported)
  framework/scanner.py   HTF engulfing + Hilega-Milega scanner (scan_frame / scan_all)
  framework/strategies/indicators/  the ONE Hilega-Milega + ATR (hm.py), used by all strategies
  framework/strategies/engulfing_strategy.py  Engulfing strategy + engulfing()/fib_entry_level
  backtest/engine.py     bar simulation with spread, commission, ATR stops
  backtest/validate.py   out-of-sample gate -> reports/validation.json
  optimize/autotune.py   Optuna + ML search, sealed split, holdout ledger, framework sync
  framework/             Bot, prop rules, protect, Monte Carlo, e2e replay
  framework/strategies/  all strategies: scalp, Engulfing, OCL + their base classes
  ml/                    features and model for the optional ML filter
  memory/store.py        agent memory (working, calibration, mistake)
  llm/budget.py          token budget and cache
  llm/ollama_client.py   remote Ollama client (URL + key)
  agent/multi_agent.py   signal scan -> risk -> analyst pipeline (alert-only path)
  live/adapter.py        the only module that talks to MetaTrader5 (official package)
  live/pipeline.py       live framework cycle (guards, Bot, orders behind DRY_RUN)
  live/sizing.py         shared account-currency conversion and lot sizing
  live/framework_runner.py  connects the framework bot to the live cycle
  bridge/windows_sync.py Windows -> VPS data and account sync
  mcp/                   MCP server (runs in Docker, reached through an SSH tunnel)
  ops/                   state, alerts, health, notifications, risk limits, trade journal
  dashboard/app.py       Dash dashboard (read-only)
config/                  backtest, validation, live, llm, ops, profiles
deploy/                  systemd units, installer for timers
scripts/                 installers helpers, backup, WinSCP upload
tests/                   runs anywhere (test_core.py); contract test on Windows
```
