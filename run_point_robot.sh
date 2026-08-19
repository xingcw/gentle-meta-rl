#!/usr/bin/env bash
# Full Point-Robot reproduction pipeline (GENTLE, AAAI'24).
# Stages: SAC behavior policies -> trajectory collection -> dynamics pretraining -> GENTLE.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=./configs/point-robot.json
GPU=${GPU:-0}
SEED=${SEED:-0}
CKPT=100000          # must match algo_params.train_epoch / eval_epoch in $CONFIG
NUM_EPISODES=100     # must be >= algo_params.n_trj
LOGDIR=./run_logs
mkdir -p $LOGDIR

# one torch thread per worker: 10 concurrent processes otherwise oversubscribe the CPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[1/4] SAC behavior policies (20 tasks x ${CKPT} steps)  $(date)"
uv run python policy_train.py $CONFIG --gpu $GPU --n_workers 10 > $LOGDIR/1_policy_train.log 2>&1

echo "[2/4] collecting ${NUM_EPISODES} trajectories/task        $(date)"
uv run python policy_eval.py --config $CONFIG --checkpoint_step $CKPT \
    --num_episodes $NUM_EPISODES --gpu $GPU > $LOGDIR/2_policy_eval.log 2>&1

echo "[3/4] pretraining task dynamics ensembles                $(date)"
uv run python pretrain_dynamics.py $CONFIG --gpu $GPU --seed_list "[$SEED]" > $LOGDIR/3_pretrain_dynamics.log 2>&1

echo "[4/4] training GENTLE                                    $(date)"
uv run python train_gentle.py $CONFIG --gpu $GPU --seed_list "[$SEED]" > $LOGDIR/4_train_gentle.log 2>&1

echo "done $(date)"
