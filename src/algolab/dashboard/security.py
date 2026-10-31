"""Access control for both dashboards (the FastAPI page in api.py and the Dash page in app.py).

Why a password on every request: a reverse proxy makes requests arrive from 127.0.0.1, so binding to localhost
does not keep outsiders out. Every route therefore needs HTTP Basic auth with DASH_PASSWORD. Without that
setting, every request is refused and the dashboard stays locked.

Write actions (start trading, promote a candidate, edit prop rules) also need DASH_ALLOW_WRITES=true. Stopping
trading is always allowed once a request is authenticated, because stopping is the safe direction.
The username is ignored; only the password is checked."""
from __future__ import annotations

import base64
import binascii
import hmac
import os

REALM = 'Basic realm="AlgoLab", charset="UTF-8"'


class DashboardSecurityError(RuntimeError):
    pass


def configured_password() -> str:
    return os.getenv("DASH_PASSWORD", "")


def writes_enabled() -> bool:
    return os.getenv("DASH_ALLOW_WRITES", "false").strip().lower() == "true"


def is_authorized(header: str | None, password: str | None = None) -> bool:
    """True only when the Authorization header is Basic auth with the configured password. Fails closed:
    no configured password, a missing header, or a malformed header all return False."""
    expected = configured_password() if password is None else password
    if not expected or not header:
        return False
    parts = header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(parts[1], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    _, _, supplied = decoded.partition(":")
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def require_password_for_start() -> None:
    """Called before a dashboard server starts. Refusing to start beats starting unlocked."""
    if not configured_password():
        raise DashboardSecurityError("DASH_PASSWORD is not set. Add it to .env before starting the dashboard.")
