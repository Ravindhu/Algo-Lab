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

## Daily commands

| Command | What it does |
|---|---|
| `algolab status` | Kill switch, last cycle, health, token use |
| `algolab validate` | Out-of-sample gate. Required before live trading |
| `algolab optimize` | Parameter search, report only (never changes settings) |
| `algolab backtest` | Run the live rule on history |
| `algolab kill` | Stop all trading now |
| `algolab resume` | Allow trading again after you have reviewed the cause |
| `algolab dashboard` | Web dashboard (read-only) |

## How the safety works

- **Fail closed.** The live cycle trades only if the kill switch is off, the strategy validation passed and is recent, the account state is fresh, the market data is fresh, and the loss limits are not hit.
- **Validation gate.** `algolab validate` tests the live parameters on several folds of history, without fitting. It requires enough trades, a profit factor above the threshold, positive expectancy, and most folds profitable.
- **Kill switch.** `algolab kill` or create `state/KILL`. Drawdown beyond the limit engages it automatically.
- **LLM token control.** A daily token budget, a cache per candidate, and compact payloads. When the budget is spent, the analyst skips trades. Usage is on the dashboard.
- **DRY_RUN.** Alerts only, no orders, until you set `DRY_RUN=false`.

## Accuracy, honestly

No system guarantees accuracy or profit. The validation gate reduces the chance of trading parameters that only worked on past data. It does not predict the future. Run on a demo account for several weeks, read the dashboard daily, and only then consider `DRY_RUN=false`.

## Layout

```
src/algolab/
  cli.py                 `algolab` command
  profile.py             loads config/profiles and .env
  scanner/scanner.py     HTF engulfing + RSI signal (shared by live and backtest)
  backtest/engine.py     bar simulation with spread, commission, ATR stops
  backtest/validate.py   out-of-sample gate -> reports/validation.json
  optimize/autotune.py   Optuna + ML parameter search with holdout gate and rollback
  llm/budget.py          token budget and cache
  llm/ollama_client.py   remote Ollama client (URL + key)
  agent/multi_agent.py   scanner -> risk -> analyst pipeline
  live/adapter.py        the only module that talks to MetaTrader5 (official package)
  live/pipeline.py       live cycle (guards, sizing, orders behind DRY_RUN)
  bridge/windows_sync.py Windows -> VPS data and account sync
  ops/                   state, alerts, health checks, notifications
  dashboard/app.py       Dash dashboard
config/                  backtest, validation, live, llm, ops, profiles
deploy/                  systemd units, installer for timers
scripts/                 installers helpers, backup, WinSCP upload
tests/                   runs anywhere (test_core.py); contract test on Windows
```
