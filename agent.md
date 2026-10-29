# agent.md: working rules for AI agents in this repo

Read this before changing anything. It states what the repo does, how to check your work, and the limits you must not cross.

## What this repo is

AlgoLab is a forex trading system on MetaTrader 5. It has four layers:

1. **Signals and strategies** (pure rules, no broker calls): `src/algolab/signals/`, `src/algolab/framework/strategies/`
2. **Backtest and validation** (same rules as live): `src/algolab/backtest/`, `src/algolab/framework/ocl_backtest.py`
3. **Live execution** (orders behind `DRY_RUN`): `src/algolab/framework/bot.py`, `src/algolab/live/`
4. **Advisory agents and tools** (recommend only): `src/algolab/agent/`, `src/algolab/mcp/`, `skills/multi-agent-frontline/`

## Hard rules

1. **Agents recommend. They never place orders.** Orders go only through `live/pipeline.py` and `live/adapter.py`, and only when `DRY_RUN=false`. Do not add an order tool to the MCP server.
2. **Keep `DRY_RUN=true` by default.** Never change it in code, config or tests.
3. **The framework is OFF until validated.** `config/framework.json` has `"enabled": false`. Turn a strategy on only after its backtest and replay pass, and say so in the change.
4. **Signals use closed bars only.** The last row of any frame is still forming. Use `closed()` or `iloc[-2]`. Never read a forming bar.
5. **Signal functions are pure.** No orders, no network, no file writes inside `signal()` or `signal_frames()`. State goes through `state()` and `restore()`.
6. **Thresholds belong in config.** Put new numbers in `config/*.json` or `config/profiles/*.json`, not in code.
7. **Do not weaken fail-closed behaviour.** Unknown open positions, unknown margin, and unpersisted signal keys all block the trade. Keep it that way.
8. **MCP tools stay read-only and advisory:** `scan_forex`, `run_agents`, `ask_llm`.

## Strategy model

Every strategy is a class with a `name`, exported in a module as `STRATEGY`. It is loaded by name from `config/framework.json`.

- `Strategy` (single timeframe): implement `signal(df)`.
- `MultiTimeframeStrategy` (HTF plus LTF): implement `signal_frames(frames)`. Also `state()`, `restore()`, `on_entry()`, `on_exit(reason, price)`.
- A signal dict must have `side` ("BUY" or "SELL"). A live order also needs `lots`, `sl`, `tp`, which the sizer fills in.

### Current strategies

| Config `type` | Class | Side | Status |
|---|---|---|---|
| `scalp` | `ScalpTrader` | both | enabled in config, framework off |
| `engulfing_ocl` | `EngulfingOcl` | BUY | HTF engulfing arms the 88.6% fib level; OCL LTF entry model. Enabled in config, framework off |
| `engulfing_ocl_inv` | `EngulfingOclInv` | SELL | inverse of `engulfing_ocl`, disabled |
| (legacy alias) | `Engulfing` (`engulfing`; alias `EngulfingHmStrategy`) | both | single-timeframe engulfing with a fib touch window; no longer in config |
| `ocl` | `OclKeylevel` | BUY | disabled by default |
| `ocl_sell_fail_top` | `OclSellFailTop` | BUY | disabled by default |
| `ocl_sell_fail_bottom` | `OclSellFailBottom` | BUY | disabled by default |
| `ocl_inv` | `OclInv` | SELL | inverse of `ocl`, disabled |
| `ocl_sell_fail_top_inv` | `OclSellFailTopInv` | SELL | inverse, disabled |
| `ocl_sell_fail_bottom_inv` | `OclSellFailBottomInv` | SELL | inverse, disabled |

### Engulfing rules (`framework/scanner.py`, `framework/strategies/engulfing_strategy.py`, `engulfing_ocl.py`)

- **One indicator home:** the Hilega-Milega math lives in `framework/strategies/indicators/hm.py`; OCL, engulfing, the scanner and the backtest all import it from there. `algolab/hilega_milega.py` stays only for `live/positions.py`.

- **Body filter:** both candles of the pattern must have a body above `min_body_pct` (default 0.5) of their high-low range. `engulfing()` applies it, so the scanner, the backtest and the strategies all agree.
- **HTF setup:** a bullish engulfing on the newest closed HTF bar, with Hilega-Milega confirmation (RSI above EMA(RSI)) on that bar. It arms one level, the `fib_pct` (default 0.886) point inside the engulfing candle, and its floor is the candle's low. SELL is the mirror.
- **LTF entry:** the OCL model, unchanged. Touch of the level, red-red-green on the last three LTF bars, the LTF Hilega-Milega gate with the oversold cap, the retry ladder, and the protections.
- **Scanner:** `framework/scanner.py` — `scan_frame` reports `status: ARMED`, `entry_level` and `floor`. The alert text shows them. The scanner has one timeframe, so it does not check the LTF touch. `signals/engulfing.py` re-exports it for back-compat.
- **Legacy:** `Engulfing` (class `name = "engulfing"`) still exists. It fills at the level on a touch within a window and is not in config.

### Inverse (SELL) blueprints (`framework/strategies/ocl_inverse.py`)

Each inverse blueprint runs its BUY twin on mirrored prices: high and low swap, and every price is negated. Red becomes green, RSI becomes 100 minus RSI, and levels flip sign. The strategy code is unchanged. The wrapper flips frames on the way in and prices on the way out.

Rules that keep this correct:

- Never use a multiplicative tolerance on a price (`x * 0.995`). It breaks for negative prices. Use `x ± 0.005 * abs(x)`.
- Never clamp a price with `max(x, 0)` unless it is known to be positive.
- Sell-failure backtests use `run_sell_fail_pair()`. TOP and BOTTOM share an in-memory `MemoryLog`. Each exhaustion is stamped with the simulated bar time, and BOTTOM only trusts stamps strictly before its own clock. This avoids look-ahead and never writes `framework_handoff.json`.
- SELL handoff state lives in its own namespace (`"SELL:"` prefix in `framework_handoff.json`), so it never collides with the BUY blueprints.
- In the live manager, a SELL position is tracked in mirrored space (`positions.new_record`, `_manage_one`). Closes use the real broker side (`_broker_side`).
- The SELL backtest is `run_inverse_backtest()` in `framework/ocl_backtest.py`.

## How to check your work

```bash
cd algo-lab
PYTHONPATH=src python -m pytest -q          # full suite, currently 280 passed, 1 skipped
PYTHONPATH=src python -m pytest -q tests/test_inverse_and_engulfing.py
```

`tests/conftest.py` gives each test its own state folder. Do not write to the real `state/` from tests.

Before reporting a strategy change as done:

- The full test suite passes.
- New logic has a test that would fail without the change.
- For a new or changed strategy, the backtest and the e2e replay (`framework/e2e.py`) run with the same settings.
- Config changes keep `"enabled": false` for anything not yet validated.

## Working conventions

- Make small, reviewable changes. One concern per commit.
- Keep the BUY behaviour unchanged unless the task asks otherwise. Run the existing OCL and protection tests after any change to `ocl.py`, `ocl_sell_fail.py`, `protect.py` or `positions.py`.
- Keep prompts and MCP output compact. Send candidates as JSON, not raw candles.
- Do not add dependencies without a reason. `requirements.txt` is the core set. `requirements-live.txt` holds the Windows MT5 packages.

## Ops pointers

- Framework cycle: `python -m algolab.live.pipeline --framework` (every 5 minutes)
- Alert-only scan: `python -m algolab.ops.pipeline` (every 15 minutes)
- Operator guide: `docs/OPERATOR_MANUAL.md`
- Multi-agent pipeline details: `skills/multi-agent-frontline/SKILL.md`

## Limits to state honestly

No strategy here guarantees accuracy or profit. The validation gate reduces the chance of trading parameters that only fit past data. It does not predict the future. Any strategy change should be run on a demo account for several weeks before `DRY_RUN=false`.
