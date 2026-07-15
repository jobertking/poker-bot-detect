#!/bin/bash

# Poker44 Miner Startup Script

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-poker44-miner-ck}"
HOTKEY="${HOTKEY:-poker44-miner-hk}"
NETWORK="${NETWORK:-finney}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
PM2_NAME="${PM2_NAME:-poker44_miner}"  ##  name of Miner, as you wish
AXON_PORT="${AXON_PORT:-8091}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"
export POKER44_MODEL_PATH="${POKER44_MODEL_PATH:-$(pwd)/models/competitive/current.joblib}"
# MODEL_PATH takes precedence over MODEL_DIR when both are set.
export POKER44_MODEL_DIR="${POKER44_MODEL_DIR:-$(pwd)/models/competitive}"
export POKER44_BATCH_CALIBRATION="${POKER44_BATCH_CALIBRATION:-topk_v1}"
# Optional: pm2 process name to reload after daily retrain (hot-reload also works in-process).
export POKER44_PM2_RELOAD="${POKER44_PM2_RELOAD:-}"
export POKER44_ARCHIVE_KEEP="${POKER44_ARCHIVE_KEEP:-30}"
# Validator request dumps: plain JSON, no rotation (watch disk usage).
export POKER44_LOG_REQUESTS="${POKER44_LOG_REQUESTS:-1}"
export POKER44_REQUEST_LOG_DIR="${POKER44_REQUEST_LOG_DIR:-$(pwd)/logs/requests}"
export POKER44_LOG_REQUEST_FULL="${POKER44_LOG_REQUEST_FULL:-1}"
export POKER44_LOG_REQUEST_GZIP="${POKER44_LOG_REQUEST_GZIP:-0}"
export POKER44_REQUEST_LOG_MAX_FILES="${POKER44_REQUEST_LOG_MAX_FILES:-0}"

MINER_ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

pm2 start $MINER_SCRIPT \
  --name $PM2_NAME -- \
  "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY axon_port=$AXON_PORT"
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
else
    echo "Access mode: validator_permit fallback"
fi
