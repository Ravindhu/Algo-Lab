# Operator manual

Keep this file current. `tests/test_operator_manual.py` fails if a command written here no longer exists.
Last updated on branch `fix/audit-gaps-feedback-loop`.

## 1. What this system is

An MT5 forex framework. It has a live pipeline (5-minute cycle), an OCL key-level strategy, prop-firm
rule enforcement, a simulated broker for offline replay, a Monte Carlo check, and agent memory.

**Safety defaults.** The framework is off (`config/framework.json`, `"enabled": false`). The profile
runs dry (`DRY_RUN=true`) until you change it. Prop enforcement is off (`config/prop.json`,
`"enabled": false`). Agents recommend; they never place orders on their own.

## 2. Configuration files

| File | What it controls |
|---|---|
| `config/framework.json` | master enable, strategies, symbols, risk per trade, max lots, spread filter, magic number |
| `config/prop.json` | prop-firm profiles. The shipped profiles are EXAMPLES. Replace them with your firm's rules. |
| `config/profiles/` | run profiles: dry-run flag and schedule. Broker login comes from the environment. |

Never commit credentials. Use environment variables or the ops secrets location.

## 3. Daily operation

Check status:
```
python -m algolab.framework.prop status        # prop account state (ACTIVE / PASSED / FAILED)
python -m algolab.framework.prop show          # the active profile's rules
```

Reset a prop account after a deliberate decision (this clears a FAILED state; the reset is logged):
```
python -m algolab.framework.prop reset --confirm
```

Run one framework cycle by hand (the scheduler normally does this every 5 minutes):
```
python -m algolab.live.pipeline --framework
```
It writes `framework_last_cycle.json` in the state folder.

## 4. Offline checks before any change goes live

Run the end-to-end replay on synthetic data (proves wiring and invariants, not edge):
```
python -m algolab.framework.e2e --bars 1500 --seed 7
```
Report: `reports/e2e/e2e_report.json`. Every invariant must show OK.

Monte Carlo of a realised trade log (10,000 bootstrap runs):
```
python -m algolab.framework.montecarlo trades.csv --balance 10000 --runs 10000 --dd-limit-pct 10
```
Read `p_profit`, `p_ruin` and the drawdown percentiles together with the trade count. Few trades
means a wide, unreliable picture.

Win-rate gate: an entry is allowed only when the lower bound of the recent win rate passes the threshold
(`framework/montecarlo.py: entry_allowed`). Thresholds are a judgement call. Set them on out-of-sample data.

## 5. Agent memory

Three stores under the ops state folder, `memory/`:

| Kind | Holds |
|---|---|
| `working` | current hypothesis, open questions, last sync |
| `calibration` | predicted probability vs outcome; check the gap regularly |
| `mistake` | what went wrong, the cause, and the rule adopted |

Python access (`algolab.memory`): `remember(kind, record)`, `recall(kind, limit)`,
`calibration_gap(records)`. Records are append-only. A new rule is added as a new record, not by editing old ones.

## 6. Agent models

The analyst agents run on Ollama (`OLLAMA_URL`, `OLLAMA_MODEL`). The Anthropic check below is an optional
check only. Nothing in the trading path calls the Anthropic API.

## 6a. Anthropic API check (optional)

Check feasibility without spending credits:
```
python scripts/check_anthropic.py
```
It reports whether `ANTHROPIC_API_KEY` is set (length and prefix only, never the value), whether the
`anthropic` SDK is installed, and whether `api.anthropic.com` is reachable.

Add `--ping` for one tiny paid call that proves the key works end to end.

To enable: set `ANTHROPIC_API_KEY` in the environment of the scheduled task (not in any file in the repo),
install `anthropic`, then run the check. Agents may read the code and the memory stores. Agents must not
place orders or edit configuration without a human approving the change.

## 7. Troubleshooting

| Symptom | Likely cause and action |
|---|---|
| Every entry blocked with "prop" | Account is FAILED or the daily room is exhausted. Run `prop status`. |
| "spread too wide for ATR" | Normal during news or the rollover. The filter is doing its job. |
| "size below broker minimum" | Stop too wide for the risk budget. Raise the account risk or skip the trade. |
| Push blocked with 403 | The git proxy has no access to the repo. The Claude GitHub App must be installed on that repo. |
| Tests fail with "No module named algolab" | Run with `PYTHONPATH=src python -m pytest`. |

## 7a. Sealed test and holdout discipline

The optimizer splits history three ways: search (first 70%), holdout (next 20%), sealed (last 10%).
The sealed slice is never used for selection. Run it once per parameter set:
```
algolab sealed
```
Result: `reports/sealed.json`. It refuses to run again for the same parameters unless `--force` is given.
Each optimizer run that selects on the holdout is counted in `state/holdout_ledger.json`. After
`max_holdout_uses` runs on the same window, no candidate is applied.

## 7b. Trade journal and entry gate

Every executed order and every framework closure is written to `state/journal.jsonl`. The dashboard's
Trades tab reads it. The framework's entry gate uses the closed trades: after `warmup_entries` open trades
and before `min_trades` closes, entries are allowed. After that, entries need the lower bound of the recent
win rate to clear `min_lower_bound` (0.25 is the breakeven win rate for a 3R target, before costs).

Legacy engulfing trades are recorded as opens only. Their broker-side exits are not yet reported back.

## 8. Known limits

- Backtest and e2e results come from synthetic or unvalidated data until a real-history run is recorded here.
- The optimizer tunes the legacy path. It writes its shared parameters (rsi_n, atr_n, sl_mult, tp_mult) into the framework's scalp strategy. OCL parameters are not optimized.
- `validate` still uses the whole history, including the sealed slice. Only `algolab sealed` is out-of-sample for the optimizer's choices.
- OCL parameters were ported from a stock-market setup. Touch buffer and round-trip cost now use pips for forex (`round_trip_cost_pips`). Re-derive them from your broker's spread and commission.
- Monte Carlo assumes independent trades. Real trades cluster, so treat the drawdown numbers as optimistic.
- No entry can be guaranteed profitable. The honest target is a positive lower bound on expectancy after costs, out of sample.
