#!/bin/bash
set -euo pipefail
cd /smile/poker/poker-bot-detect
LIVE_MTIME=$(stat -c '%Y' models/competitive/current.joblib)
{
  echo "live_mtime_before=$LIVE_MTIME"
  OMP_NUM_THREADS=8 .venv_ml/bin/python scripts/train/train_robust_staging.py \
    --feature-set beat_v3 --seed 42 --holdout-days 2
  echo "==== BEAT_V3 DONE ===="
  OMP_NUM_THREADS=8 .venv_ml/bin/python scripts/train/train_robust_staging.py \
    --feature-set beat_v3_coherent --seed 43 --holdout-days 2
  echo "==== COHERENT DONE ===="
  LIVE_MTIME_AFTER=$(stat -c '%Y' models/competitive/current.joblib)
  echo "live_mtime_after=$LIVE_MTIME_AFTER"
  if [ "$LIVE_MTIME" = "$LIVE_MTIME_AFTER" ]; then
    echo "LIVE_UNTOUCHED=OK"
  else
    echo "LIVE_TOUCHED=BAD"
  fi
  echo "ALL_DONE"
} > logs/train_robust_staging.log 2>&1
