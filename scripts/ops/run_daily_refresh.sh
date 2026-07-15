#!/usr/bin/env bash
# Daily: fetch Poker44 public benchmark + retrain competitive miner model.
# Example cron (UTC morning):
#   15 6 * * * /smile/poker/poker-bot-detect/scripts/ops/run_daily_refresh.sh >>/smile/poker/poker-bot-detect/logs/daily_refresh/cron.log 2>&1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv_ml/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi
mkdir -p "${ROOT}/logs/daily_refresh"
exec "$PY" "${ROOT}/scripts/ops/daily_refresh_retrain.py" "$@"
