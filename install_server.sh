#!/usr/bin/env bash
# install_server.sh - one-command AlgoLab server installer (Ubuntu 22.04 / 24.04)
# Installs: Chrome, WineHQ + MetaTrader 5, desktop GUI (xRDP), Docker, firewall, Python env.
# Usage: sudo bash install_server.sh [TRUSTED_IP]
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo bash $0 [TRUSTED_IP]"
  exit 1
fi

TRUSTED_IP="${1:-}"
APP_USER="${SUDO_USER:-algo}"
APP_DIR="/opt/algolab"
MT5_PREFIX="/home/${APP_USER}/.mt5"
MT5_EXE="${MT5_PREFIX}/drive_c/Program Files/MetaTrader 5/terminal64.exe"
CODENAME="$(lsb_release -cs)"

echo "==> [1/9] Base packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl wget ca-certificates gnupg software-properties-common \
  build-essential ufw cabextract unzip xvfb winbind

echo "==> [2/9] Desktop GUI (XFCE + xRDP) for MT5 setup"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  xfce4 xfce4-goodies xrdp dbus-x11
systemctl enable --now xrdp
# Each new RDP login starts XFCE
echo "xfce4-session" > "/home/${APP_USER}/.xsession"
chown "${APP_USER}:${APP_USER}" "/home/${APP_USER}/.xsession"
adduser xrdp ssl-cert >/dev/null 2>&1 || true

echo "==> [3/9] Google Chrome"
wget -qO /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
DEBIAN_FRONTEND=noninteractive apt-get install -y /tmp/chrome.deb
rm -f /tmp/chrome.deb

echo "==> [4/9] WineHQ (runs MetaTrader 5)"
dpkg --add-architecture i386
mkdir -pm755 /etc/apt/keyrings
wget -qO /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
wget -qNP /etc/apt/sources.list.d/ \
  "https://dl.winehq.org/wine-builds/ubuntu/dists/${CODENAME}/winehq-${CODENAME}.sources"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --install-recommends winehq-stable

echo "==> [5/9] MetaTrader 5 into ${MT5_PREFIX}"
sudo -u "$APP_USER" env WINEPREFIX="$MT5_PREFIX" WINEARCH=win64 \
  xvfb-run -a bash -c 'wineboot -i; wineserver -w' || true
sudo -u "$APP_USER" bash -c "
  cd /tmp
  wget -qO mt5setup.exe https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe
  WINEPREFIX='$MT5_PREFIX' timeout 900 xvfb-run -a wine mt5setup.exe /auto
  WINEPREFIX='$MT5_PREFIX' wineserver -w
" || echo "WARNING: MT5 silent install did not finish. Finish it from the desktop (step 2) with the mt5-gui launcher."

# GUI launcher for MT5 (run from the xRDP desktop)
cat > /usr/local/bin/mt5-gui <<EOF
#!/usr/bin/env bash
export WINEPREFIX="${MT5_PREFIX}"
if [[ ! -f "${MT5_EXE}" ]]; then
  echo "MT5 not found. Run the installer from the desktop or reinstall."
  exit 1
fi
exec wine "${MT5_EXE}"
EOF
chmod 755 /usr/local/bin/mt5-gui

echo "==> [6/9] Docker (for the MCP server and scanner containers)"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker
usermod -aG docker "$APP_USER"

echo "==> [7/9] Firewall (UFW)"
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
if [[ -n "$TRUSTED_IP" ]]; then
  ufw allow from "$TRUSTED_IP" to any port 3389 proto tcp comment 'xRDP desktop (MT5 GUI)'
  ufw allow from "$TRUSTED_IP" to any port 8050 proto tcp comment 'AlgoLab dashboard'
else
  echo "WARNING: no TRUSTED_IP given. Desktop (3389) and dashboard (8050) stay closed."
fi
# MCP (8765) stays bound to 127.0.0.1 in Docker. Reach it through an SSH tunnel.
ufw --force enable

echo "==> [8/9] Python 3.13 and AlgoLab environment"
if [[ ! -d "$APP_DIR" ]]; then
  echo "Copy the AlgoLab project to $APP_DIR first, then re-run this script."
  exit 1
fi
mkdir -p "$APP_DIR"/{data,reports,logs}
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

sudo -u "$APP_USER" bash -lc "
  set -e
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\$HOME/.local/bin:\$PATH
  cd '$APP_DIR'
  uv python install 3.13
  uv venv -p 3.13 .venv
  . .venv/bin/activate
  uv pip install -e .
"

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  echo "Created $APP_DIR/.env. Fill in credentials and OLLAMA_URL."
fi

echo "==> [9/9] Services"
if [[ -f "$APP_DIR/deploy/algolab-dashboard.service" ]]; then
  install -m 644 "$APP_DIR/deploy/algolab-dashboard.service" /etc/systemd/system/algolab-dashboard.service
  sed -i "s/^User=.*/User=${APP_USER}/" /etc/systemd/system/algolab-dashboard.service
  systemctl daemon-reload
  systemctl enable --now algolab-dashboard
fi
if [[ -f "$APP_DIR/docker-compose.yml" ]]; then
  sudo -u "$APP_USER" bash -c "cd '$APP_DIR' && docker compose up -d --build"
fi

echo
echo "Installation finished."
echo "  Desktop GUI : RDP to <server-ip>:3389 (trusted IP), then run: mt5-gui"
echo "  Set a login password for the desktop user:  sudo passwd ${APP_USER}"
echo "  Dashboard   : http://<server-ip>:8050 (trusted IP only)"
echo "  MCP server  : ssh -L 8765:127.0.0.1:8765 <user>@<server-ip>  (then http://127.0.0.1:8765)"
echo "  Next step   : edit $APP_DIR/.env (OLLAMA_URL, credentials), keep DRY_RUN=true"
echo "  Firewall    : sudo ufw status verbose"
