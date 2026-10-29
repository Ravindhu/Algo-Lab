#!/usr/bin/env bash
# Installs the set-and-forget timers. Run once, after install_server.sh.
#   sudo bash deploy/install_ops.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash deploy/install_ops.sh"
  exit 1
fi

APP_DIR="/opt/algolab"
APP_USER="${SUDO_USER:-algo}"

mkdir -p "$APP_DIR"/{state,backups,reports,logs}
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod +x "$APP_DIR/scripts/backup.sh"

cp "$APP_DIR"/deploy/systemd/*.service "$APP_DIR"/deploy/systemd/*.timer /etc/systemd/system/
sed -i "s/^User=.*/User=${APP_USER}/" /etc/systemd/system/algolab-*.service

systemctl daemon-reload
systemctl enable --now algolab-pipeline.timer algolab-health.timer \
  algolab-backup.timer

echo
systemctl list-timers 'algolab-*' --no-pager
echo
echo "Set alert channels in $APP_DIR/.env: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (or ALERT_WEBHOOK_URL)."
echo "Parameter changes are NOT automatic: run 'algolab optimize' to review, then 'algolab optimize --apply' to approve."
echo "Kill switch: sudo -u $APP_USER touch $APP_DIR/state/KILL   (resume: rm $APP_DIR/state/KILL)"
