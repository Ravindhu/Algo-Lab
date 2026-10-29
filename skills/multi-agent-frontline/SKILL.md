---
name: multi-agent-frontline
description: Run, extend or debug AlgoLab's frontline multi-agent pipeline (HTF engulfing scanner, risk gate, Ollama analyst) and its MCP tools. Use when the user asks about agents, the forex engulfing scan, LLM trade review, Ollama setup, or the MCP server.
---

# Multi-agent frontline

## Agents (in order)
| Agent | File | Job |
|---|---|---|
| ScannerAgent | `src/algolab/signals/engulfing.py` | Finds HTF engulfing candles confirmed by the Hilega-Milega RSI/EMA rule, on the last closed bar. |
| RiskAgent | `src/algolab/agent/multi_agent.py` | Limits candidates to the free open-trade slots (`risk.max_open_trades`). |
| AnalystAgent | same file | Sends each candidate to Ollama. Expects JSON `{decision, confidence, reason}`. Unparseable output becomes SKIP. |

The Orchestrator keeps only `TAKE` decisions at or above `analyst.min_confidence`.

## Rules
1. Agents recommend. They never place orders. Execution stays in `live/pipeline.py` (orders go through `live/adapter.py`), which respects `DRY_RUN`.
2. Keep prompts short. Send candidates as compact JSON, never raw candles.
3. The Ollama endpoint comes from `OLLAMA_URL` (and `OLLAMA_API_KEY` if the remote machine needs one). Never hard-code it.
4. MCP tools are read-only and advisory: `scan_forex`, `run_agents`, `ask_llm`. Do not add an order tool.

## Run
```bash
# local (stdio, for Claude Desktop/Code)
ALGOLAB_PROFILE=local python -m algolab.mcp.server

# VPS / Docker (streamable-http on 127.0.0.1:8765, reached via SSH tunnel)
ALGOLAB_PROFILE=vps python -m algolab.mcp.server

# one-off scan
python -m algolab.signals.engulfing H4
```

## Extend
Add a class with a `run()` method, then call it in `Orchestrator.run()`. Put new thresholds in `config/profiles/*.json`, not in code.

## Checks before reporting success
- The scan returns a list (possibly empty) without errors.
- `run_agents` returns `dry_run: true` unless the operator set it otherwise.
- The Ollama URL responds (`curl $OLLAMA_URL/api/tags`).
