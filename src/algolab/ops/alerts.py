"""Alerts to Telegram and/or a generic webhook, with per-key cooldown so one problem does not spam you."""
import logging
import os

from algolab.ops.state import minutes_since, now_utc, read_json, write_json

log = logging.getLogger("algolab.alerts")


def send(text: str, level: str = "WARNING", key: str | None = None,
         cooldown_min: float = 60) -> bool:
    if key:
        sent = read_json("alerts.json", {})
        if key in sent and minutes_since(sent[key]) < cooldown_min:
            return False
        sent[key] = now_utc().isoformat()
        write_json("alerts.json", sent)

    import httpx  # imported here so modules that only check state do not need it

    msg = f"[AlgoLab {level}] {text}"
    delivered = False
    try:
        token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if token and chat:
            httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                       json={"chat_id": chat, "text": msg}, timeout=10).raise_for_status()
            delivered = True
        hook = os.getenv("ALERT_WEBHOOK_URL")
        if hook:
            httpx.post(hook, json={"text": msg}, timeout=10).raise_for_status()
            delivered = True
    except httpx.HTTPError as e:
        # Log only the exception type: the URL contains the bot token.
        log.error("alert delivery failed (%s)", type(e).__name__)
    if not delivered:
        log.warning("no alert channel delivered: %s", msg)
    return delivered
