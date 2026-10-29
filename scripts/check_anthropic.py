"""Feasibility check for using the Anthropic API from this project's agents.

Reports, without printing any secret:
  1. whether ANTHROPIC_API_KEY is set (and its length only, never the value)
  2. whether the `anthropic` Python SDK is importable
  3. whether api.anthropic.com is reachable over HTTPS (no key is sent)
  4. with --ping: one minimal Messages API call (costs a few tokens; needs the key)

Usage:  python scripts/check_anthropic.py          (safe, no spend)
        python scripts/check_anthropic.py --ping   (one tiny paid call)
Exit code 0 = ready, 1 = not ready.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import ssl
import sys

HOST = "api.anthropic.com"
PING_MODEL = os.environ.get("ANTHROPIC_PING_MODEL", "claude-haiku-5-5")


def check_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("[key]     ANTHROPIC_API_KEY is NOT set")
        return False
    looks_ok = key.startswith("sk-ant-")
    print(f"[key]     ANTHROPIC_API_KEY is set (length {len(key)}, prefix "
          f"{'ok' if looks_ok else 'UNEXPECTED'})")
    return looks_ok


def check_sdk() -> bool:
    ok = importlib.util.find_spec("anthropic") is not None
    print(f"[sdk]     anthropic package {'installed' if ok else 'NOT installed (pip install anthropic)'}")
    return ok


def check_network() -> bool:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((HOST, 443), timeout=8) as raw:
            with ctx.wrap_socket(raw, server_hostname=HOST):
                pass
        print(f"[network] {HOST}:443 reachable over TLS")
        return True
    except OSError as exc:
        print(f"[network] {HOST}:443 NOT reachable ({type(exc).__name__}: {exc})")
        return False


def ping() -> bool:
    import anthropic  # imported here so the safe checks run without the SDK

    client = anthropic.Anthropic()
    msg = client.messages.create(model=PING_MODEL, max_tokens=5,
                                 messages=[{"role": "user", "content": "Reply with OK."}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    print(f"[ping]    model {PING_MODEL} answered: {text.strip()[:40]!r}")
    return bool(text.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ping", action="store_true", help="make one tiny paid Messages API call")
    args = ap.parse_args(argv)
    results = [check_key(), check_sdk(), check_network()]
    if args.ping and all(results):
        try:
            results.append(ping())
        except Exception as exc:  # report, do not crash the check
            print(f"[ping]    failed: {type(exc).__name__}: {exc}")
            results.append(False)
    ready = all(results)
    print("READY" if ready else "NOT READY")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
