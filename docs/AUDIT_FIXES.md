# Algo-Lab audit: prioritized fix list

Audited default branch at commit `1d27b6d`.

Priority:
- **Critical:** can cause wrong or duplicate broker orders, or leave real positions unprotected.
- **High:** breaks a safety promise, or silently weakens risk limits or research integrity.
- **Medium:** correctness or design gap with a narrower trigger.
- **Low:** operability, docs, or cleanup.

Items marked **(verify)** depend on files not yet read: `framework/bot.py`, `framework/prop.py`, `ops/journal.py`, the ML model, and the dashboard.

---

## Critical

### C1. Stale tick corrupts bar timestamps used for stop management
- **Where:** `live/adapter.py`, `candles()` and `server_utc_offset()`
- **Problem:** The MT5 server-time offset is computed from the latest tick. A stale tick (quiet symbol, or weekend) shifts every bar by hours or days. The 30-minute rounding absorbs only about 15 minutes of error. `live/positions.py` then re-processes old bars (re-triggering trail logic) or skips new ones, and `last_bar` drifts.
- **Fix:** Reject ticks older than a few seconds and fail closed. Sanity-check that the offset is a whole number of 30-minute steps within ±14 hours. Add a test with a 2-hour-old tick.

### C2. A stale lock lets two trading cycles run at once
- **Where:** `ops/state.py`, `cycle_lock()`; `live/pipeline.py`
- **Problem:** A lock older than 10 minutes is removed as if its process crashed. The lock is never refreshed, so a slow but live cycle (for example, a hung MT5 call) is treated as dead, and the next timer run trades concurrently. The PID written to the lock is never checked. No MT5 call has a timeout.
- **Fix:** Check whether the PID in the lock is alive before stealing it. Refresh the lock during the cycle. Add a timeout to every MT5 call. Never steal a lock whose owner is still running.

### C3. Prop FAILED flatten leaves untracked positions open
- **Where:** `live/framework_runner.py`, prop-flatten block in `run_framework()`; `live/positions.py`, `flatten()`
- **Problem:** `flatten()` closes only positions in `tracked`. A broker position with this magic number that isn't tracked (an orphan) is reported and blocks entries, but it stays open through a prop breach.
- **Fix:** On FAILED or weekend flatten, close every broker position with the project's magic number, including orphans. Alert on any that can't be closed.

---

## High

### H1. DRY_RUN does not stop exits
- **Where:** `live/positions.py`, `manage()` and `flatten()`, called from `live/framework_runner.py`
- **Problem:** The docstrings of `live/pipeline.py` and `framework_runner.py` say DRY_RUN=true means "alert only" and "nothing reaches the broker." But `manage()` and `flatten()` receive no `dry_run` flag. Any existing position with this magic number is closed for real in dry-run mode.
- **Fix:** Pass `dry_run` through. In dry-run, log and alert the would-be exit instead of sending it. Decide the intended behavior, then make docs and code agree.

### H2. One bad position record stops management for all positions
- **Where:** `live/positions.py`, `manage()`
- **Problem:** `manage()` catches only `AdapterError`. Any other exception (malformed `last_bar`, pandas error, missing key) escapes the loop. No other position gets exits that cycle, and `framework_runner` never saves state.
- **Fix:** Catch exceptions per position, alert with the ticket number, and continue. Keep the record tracked.

### H3. Corrupt state silently resets risk and tracking
- **Where:** `ops/state.py`, `read_json()`; used by `ops/risk.py` and `live/framework_runner.py`
- **Problem:** A JSON decode error returns the default.
  - If `risk.json` is unreadable, the drawdown peak resets to current equity. A prior drawdown is forgotten, and the kill switch may never fire.
  - If `framework_positions.json` is unreadable, `tracked` becomes `{}`, and the next write overwrites the file with `{}`. Every open position loses its trailing stop and HTF exit.
- **Fix:** Fail closed on decode errors for `risk.json`, positions, and strategy state. Keep a `.bak` copy before each write. Alert on any decode failure.

### H4. Holdout and sealed windows move as data grows
- **Where:** `optimize/autotune.py`, `_split()`, `_window_key()`, `run_sealed_test()`
- **Problem:** Splits are fractions of the current series length. Each new bar moves the cut points, so `hold_start` changes and the holdout ledger key resets. The "holdout used N times" limit never triggers. The sealed slice also moves, so "once per parameter set" no longer means the same data was tested.
- **Fix:** Pin the split to fixed calendar dates stored in config. Key the ledger and sealed result on those dates, not on lengths.

### H5. A malformed LLM verdict is cached and crashes every cycle for the day
- **Where:** `llm/budget.py`, `CachedAnalyst.run()`; `agent/multi_agent.py`, `AnalystAgent.run()` and `Orchestrator.run()`
- **Problem:**
  - A non-dict JSON reply raises `TypeError` in `{**cand, **verdict}`.
  - A TAKE with a non-numeric confidence (`"high"`) hits `float()` in `Orchestrator.run()` and raises `ValueError`.
  - The bad verdict is cached before that check runs. Every later cycle that UTC day replays the cache hit and crashes again, so the alert pipeline sends nothing until midnight UTC.
- **Fix:** Validate the verdict schema before caching: `decision` in {TAKE, SKIP}, `confidence` a number in [0, 1]. On failure, return SKIP and don't cache.

---

## Medium

### M1. Validation does not test the config that goes live
- **Where:** `backtest/validate.py`, `framework_replay()`; `framework/e2e.py`, `run_e2e()`
- **Problem:** The replay runs with `prop_enabled=False`, but a live prop account applies lot caps and entry blocks. The replay also uses a hard-coded `SPREADS` dict with a 0.0002 default, not the costs in `config/backtest.json`.
- **Fix:** Replay with the active prop profile. Take spreads and commission from the same costs config the legacy backtest uses.

### M2. Staleness check looks only at the newest file
- **Where:** `ops/pipeline.py`, `data_age_minutes()`
- **Problem:** The check uses the newest mtime across all `*_HTF.csv` files. One frozen symbol feed passes while the others update.
- **Fix:** Check age per symbol. Skip or flag stale symbols instead of trusting the newest file.

### M3. The alert path does not check validation status
- **Where:** `ops/pipeline.py`, `run_cycle()`
- **Problem:** "ARMED" alerts are sent with no validation check. Live trading is gated; the alerts are not. Anyone acting on an alert may be trading an unvalidated setting.
- **Fix:** Gate alerts on the same validation result, or label them "unvalidated" when the gate fails.

### M4. The prop-mode entry gate may skip the Bot's own gate **(verify)**
- **Where:** `live/framework_runner.py`, `combined_gate()`
- **Problem:** When a prop guard exists, the gate returns `guard.check_entry(...)` and never calls `prev_gate`. Any checks in the Bot's default gate (spread, session, and so on) are skipped whenever prop is enabled.
- **Fix:** Run `prev_gate` first, then the prop check, and return the first failure. Confirm what the Bot's default gate does.

### M5. Autotune apply has no rollback for exceptions after files are written
- **Where:** `optimize/autotune.py`, the `if apply:` block
- **Problem:** Rollback covers only a failed formal validation. If `M.fit` on the full dataset raises after `S.write` and `_sync_framework` have run, the new parameters stay on disk with the old model. The fingerprint mismatch blocks entries, so the damage is contained, but the state is inconsistent.
- **Fix:** Wrap the whole apply block in `try`/`except`, and restore all three files (strategy, model, framework) on any exception.

---

## Low

### L1. `algolab status` reads the wrong files and keys
- **Where:** `cli.py`, `cmd_status()`
- **Problem:** It reads `live_last_cycle.json`, but the live cycle writes `framework_last_cycle.json`. The live report has no `open_positions` field, so that value is always null. It reads the `cached` key from `tokens.json`, but the file stores `cache`.
- **Fix:** Read `framework_last_cycle.json`. Add `open_positions` to the live report. Use `len(cache)` for the cached count.

### L2. Alert delivery counts partial success as full success
- **Where:** `ops/alerts.py`, `send()`
- **Problem:** If Telegram succeeds and the webhook fails, `delivered` stays True. The cooldown is recorded, and the webhook failure is only logged, never retried.
- **Fix:** Track each channel separately. Retry or record the channel that failed.

### L3. The `algolab kill` wording overstates what it does
- **Where:** `cli.py` help text; `live/framework_runner.py`
- **Problem:** "Stop all trading now" blocks new entries only. Open positions keep being managed and exited. That may be the intent, but the message doesn't say so.
- **Fix:** Reword to "block new entries." Add a separate flatten command if an emergency close is wanted.

### L4. Net-deposits query fetches full account history every cycle
- **Where:** `live/adapter.py`, `net_deposits()`
- **Problem:** It requests deals from epoch 0 on every 5-minute cycle. The cost grows with account history.
- **Fix:** Cache the total and fetch only deals since the last cached timestamp.

### L5. Overlapping alert-pipeline runs can double-alert
- **Where:** `ops/pipeline.py`
- **Problem:** No lock is taken, so two overlapping timer runs can both read `seen_signals.json` before either writes it.
- **Fix:** Take `cycle_lock("ops")`.

### L6. Account staleness check trusts a timestamp that could be in the future
- **Where:** `ops/pipeline.py`, `guard()`
- **Problem:** `minutes_since` goes negative if the Windows sync clock is ahead. A negative age always passes the freshness check.
- **Fix:** Treat any negative age as stale, and log the clock skew.

### L7. Validation fingerprint omits some settings
- **Where:** `backtest/validate.py`, `config_fingerprint()`
- **Problem:** Changes to `ops.json`, `live.json`, or `symbols.json` don't invalidate the validation report. Some of them (`max_open_trades`, the symbol list) can affect results.
- **Fix:** Add them to the fingerprint, or document why they're excluded.

---

## Suggested order of work
1. C1, C2, C3: the three Critical items.
2. H1, H2, H3: DRY_RUN semantics, per-position isolation, fail-closed state.
3. H5: stops the daily alert outage.
4. H4: research integrity, before the next autotune run.
5. M1, M4: confirm the Bot's default gate and the replay costs.
6. Remaining Medium and Low items.

## Open questions
- Intended DRY_RUN behavior for existing positions (H1).
- Whether any prop account is live. If so, C3 is urgent.
