"""AlgoLab MCP server.

Exposes READ-ONLY and ADVISORY tools. No order placement is exposed, on purpose.
  local profile -> stdio (for Claude Desktop / Claude Code on your machine)
  vps profile   -> streamable-http on port 8765 (bound to 127.0.0.1 by Docker;
                   reach it through an SSH tunnel)
"""
import json

from mcp.server.fastmcp import FastMCP

from algolab.agent.multi_agent import Orchestrator
from algolab.llm.ollama_client import OllamaClient
from algolab.profile import load_profile
from algolab.signals.engulfing import scan_all

profile = load_profile()
mcp = FastMCP("algolab", host=profile["mcp"]["host"], port=profile["mcp"]["port"])


@mcp.tool()
def scan_forex(htf: str = "") -> str:
    """Find higher-timeframe engulfing signals with RSI at oversold (BUY) or overbought (SELL)."""
    return json.dumps(scan_all(htf or None), indent=2)


@mcp.tool()
def run_agents(open_positions: int = 0) -> str:
    """Run the frontline pipeline (scanner, risk gate, LLM analyst). Places NO orders."""
    return json.dumps(Orchestrator().run(open_positions), indent=2, default=str)


@mcp.tool()
def ask_llm(prompt: str) -> str:
    """Ask the configured Ollama model a question. Input is capped at 4000 characters."""
    o = profile["ollama"]
    client = OllamaClient(o["url"], o["api_key"], o["model"])
    return client.chat([{"role": "user", "content": prompt[:4000]}])


def main() -> None:
    mcp.run(transport=profile["mcp"]["transport"])


if __name__ == "__main__":
    main()
