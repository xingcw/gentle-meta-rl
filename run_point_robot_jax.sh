#!/usr/bin/env bash
# Full Point-Robot pipeline in JAX (gentle_jax), mirroring run_point_robot.sh.
# Stages: SAC behavior policies -> collection -> dynamics pretraining -> GENTLE.
#
# Stages 1-2 are skipped by default: the dataset the torch pipeline produced is
# already on disk and the two writers share a format. Set REGEN_DATA=1 to
# regenerate it with the JAX SAC instead.
set -euo pipefail
cd "$(dirname "$0")"

SEED=${SEED:-0}
REGEN_DATA=${REGEN_DATA:-0}
SAC_PRECISION=${SAC_PRECISION:-tensorfloat32}   # 'highest' matches torch fp32 exactly
LOGDIR=./run_logs_jax
mkdir -p $LOGDIR

echo "[jax] point-robot pipeline, seed $SEED   $(date)"
uv run python -u -m gentle_jax \
    --seed "$SEED" \
    --regen-data "$REGEN_DATA" \
    --sac-precision "$SAC_PRECISION" \
    2>&1 | tee "$LOGDIR/pipeline_seed${SEED}.log"

echo "done $(date)"
echo "compare against the paper with:"
echo "  uv run python report_results.py --glob '$LOGDIR/progress_seed*.csv'"
