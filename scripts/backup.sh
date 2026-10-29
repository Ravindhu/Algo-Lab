#!/usr/bin/env bash
# Daily backup of config, reports, state and logs. Secrets (.env) are NOT included on purpose.
set -euo pipefail
cd "$(dirname "$0")/.."

KEEP_DAYS="${KEEP_DAYS:-14}"
mkdir -p backups
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
tar -czf "backups/algolab-${STAMP}.tar.gz" config reports state logs 2>/dev/null || true
find backups -name 'algolab-*.tar.gz' -mtime +"$KEEP_DAYS" -delete
echo "backup written: backups/algolab-${STAMP}.tar.gz (keeping ${KEEP_DAYS} days)"
