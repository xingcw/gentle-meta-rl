# Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations
Code for AAAI'24 paper "Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations".

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The environment is
pinned in `pyproject.toml` / `uv.lock`:

```bash
uv sync --extra cuda   # NVIDIA GPU
uv sync --extra tpu    # Cloud TPU VM
```

The accelerator backend for `gentle_jax` is an extra, because `jax[cuda12]` and
`jax[tpu]` cannot share an environment; plain `uv sync` leaves JAX on CPU. The
two extras are declared as conflicting, so switching between them is a re-sync
rather than a manual uninstall.

This installs Python 3.11, PyTorch with CUDA 12.8 wheels, and the legacy
`hydra-core==0.11.3` / `gym==0.25.2` stack the code targets. Run everything with
`uv run python ...` (no `conda activate` needed). The original `environment.yaml`
is kept for reference only — its `torch==1.9.0+cu111` pin does not support GPUs
newer than Ampere.

### TPU notes

A v4-8 VM exposes 4 chips. `gentle_jax` runs one seed on one chip, so four seeds
fit on the node at once:

```bash
for d in 0 1 2 3; do
  GENTLE_DATA_DIR=$PWD/data_seed$d SEED=$d REGEN_DATA=1 \
  TPU_VISIBLE_DEVICES=$d TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_ADDRESSES=local \
  ./run_point_robot_jax.sh &
done
```

`GENTLE_DATA_DIR` has to differ per seed: `--regen-data 1` writes the dataset
under it, and concurrent seeds would otherwise overwrite each other.

Dropping the `TPU_*` variables instead gives one run the whole node. Stage 1
then splits its 20 SAC agents across every visible device (5 per chip on a
v4-8), which is where a run spends ~90% of its time: 1542s -> 450s, a 3.4x
speedup that takes a full seed from 28.5 to 10.5 minutes. The split is
automatic -- `run_datagen` shards the vmap task axis over `local_device_count()`
and falls back to one device when the task count does not divide evenly, so the
single-GPU path is unaffected.

Which layout to use depends on the unit of work. A 4-seed sweep is faster
one-seed-per-chip (28.5 min for all four) than sharded seeds run back to back
(~42 min); sharding wins when iterating on a single configuration.

Sharding changes how XLA partitions the computation, so stage 1 is
reproducible-in-distribution rather than bit-identical: reset states still match
exactly, but 100k chaotic SAC steps amplify float32 noise into visibly different
policies. The collected datasets agree statistically -- mean episode return
-5.3139 sharded vs -5.3142 on one chip -- and both land inside the paper's error
bars downstream.

### MuJoCo domains (optional)

Point-Robot needs no MuJoCo. For Cheetah/Ant install MuJoCo 2.0+, for
Hopper/Walker MuJoCo 1.31, then:

```bash
uv sync --extra mujoco
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro${VERSION_NUM}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro${VERSION_NUM}/bin
```

`rlkit/envs/__init__.py` keys off `MUJOCO_PY_MJPRO_PATH`: when it is unset only
the MuJoCo-free envs are registered, which is what the Point-Robot pipeline uses.

### Where data goes

All generated data lives under `$GENTLE_DATA_DIR` (default: `./data`), resolved in
`rlkit/paths.py`:

```
$GENTLE_DATA_DIR/gentle_data/$env_name/goal_idx$i/    SAC checkpoints + trajectories
$GENTLE_DATA_DIR/gentle_data/asset/dynamics/$env/     pretrained dynamics ensembles
```

## Data Generation

Train behavior policies (one SAC run per task, `--n_workers` tasks at a time):

```bash
uv run python policy_train.py ./configs/ant-dir.json --gpu 0 --n_workers 10
```

Collect trajectories from the trained policies:

```bash
uv run python policy_eval.py --config ./configs/ant-dir.json \
    --checkpoint_step 1000000 --num_episodes 50
```

`--checkpoint_step` must match `algo_params.train_epoch`/`eval_epoch` in the
config, and `--num_episodes` must be at least `algo_params.n_trj`. Both stages
write to `$GENTLE_DATA_DIR/gentle_data/$env_name/goal_idx$i/`.

## Training GENTLE

The configuration files are in `./configs`. To train GENTLE on Ant-Dir, first
pretrain the dynamics model:

```bash
uv run python pretrain_dynamics.py ./configs/ant-dir.json
```

Then run:

```bash
uv run python train_gentle.py ./configs/ant-dir.json
```

Logs will be written to `./logs/ant-dir/gentle/seed$SEED/`.

## Reproducing the Point-Robot result

Point-Robot is the one domain that runs end-to-end without MuJoCo. The whole
pipeline is scripted:

```bash
./run_point_robot.sh          # ~3h on a single GPU; stage logs in ./run_logs/
uv run python report_results.py --env point-robot
```

`report_results.py` reads `logs/point-robot/gentle/seed*/*/progress.csv` and
prints the four reported metrics next to the paper's values:

| code metric | paper protocol |
| --- | --- |
| `AverageReturn_all_train_tasks` | given-context, in-distribution |
| `AverageReturn_all_test_tasks` | given-context, OOD |
| `AverageReturn_all_train_tasks_expl` | one-shot, in-distribution |
| `AverageReturn_all_test_tasks_expl` | one-shot, OOD |

### Config constraints

Two coupling constraints are implicit in the code and fail late if violated:

- `algo_params.batch_size` must equal `algo_params.embedding_batch_size`. In
  `GENTLE._take_step` the latent `task_z` is computed over the RL batch and then
  reshaped to the context batch's `(meta_batch, embedding_batch_size, -1)`, so a
  mismatch silently changes the latent width and blows up in the decoder.
- `algo_params.train_epoch` / `eval_epoch` name the SAC checkpoint the offline
  data is collected from, so they must be a multiple of the SAC
  `save_frequency`, and `policy_eval.py --checkpoint_step` must match them.
  Per-domain overrides of `pytorch_sac`'s `train.yaml` go in a `sac_params`
  block in the config JSON.

## Reference

```bash
@inproceedings{gentle,
  author={Renzhe Zhou, Chen-Xiao Gao, Zongzhang Zhang, Yang Yu},
  title={Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2024}
}
```

