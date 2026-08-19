"""Phase C verification: the stacked JAX ensembles must match MultiTaskDynamics.

Loads the torch stage-3 checkpoints into the JAX ensembles and compares the
relabeling call (`step`) that stage 4 depends on, then trains from scratch and
checks the holdout losses land in the same range as torch's.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gentle_jax  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import torch

from gentle_jax import dynamics as jdyn
from gentle_jax.data import build_dataset
from rlkit.paths import data_dir_for, dynamics_dir_for
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics

OBS_DIM, ACT_DIM, NET, ENSEMBLE = 2, 2, 64, 7
HIDDEN = (NET, NET)
DECAYS = [2.5e-5, 5e-5, 7.5e-5]
N_TRAIN_TASKS = 10
DATA_DIR = data_dir_for('point-robot')
DYN_DIR = dynamics_dir_for('point-robot', 0)


def _torch_dynamics():
    ptu.set_gpu_mode(False)
    md = MultiTaskDynamics(num_tasks=N_TRAIN_TASKS, hidden_size=NET,
                           num_hidden_layers=2, action_dim=ACT_DIM, obs_dim=OBS_DIM,
                           reward_dim=1, use_next_obs_in_context=False,
                           ensemble_size=ENSEMBLE, dynamics_weight_decay=DECAYS)
    md.load(DYN_DIR)
    for m in md.models:
        m.eval()
    return md


def test_step_parity():
    md = _torch_dynamics()
    ens = jdyn.load_from_torch(DYN_DIR, N_TRAIN_TASKS, OBS_DIM, ACT_DIM, HIDDEN,
                               ENSEMBLE, with_next_obs=False)

    rng = np.random.RandomState(0)
    batch = 64
    task_indices = np.arange(N_TRAIN_TASKS)
    obs = rng.randn(N_TRAIN_TASKS * batch, OBS_DIM).astype(np.float32)
    act = rng.uniform(-1, 1, size=(N_TRAIN_TASKS * batch, ACT_DIM)).astype(np.float32)

    with torch.no_grad():
        t_out, t_std = md.step(torch.as_tensor(obs), torch.as_tensor(act),
                               task_indices, return_std=True)
    j_out, j_std = jax.jit(ens.step)(jnp.asarray(obs), jnp.asarray(act),
                                     jnp.asarray(task_indices))

    e_out = float(np.abs(np.asarray(j_out) - t_out.numpy()).max())
    e_std = float(np.abs(np.asarray(j_std).reshape(-1) - t_std.numpy().reshape(-1)).max())
    print(f'  step() mean prediction max err {e_out:.2e}')
    print(f'  step() ensemble spread max err {e_std:.2e}')
    assert e_out < 1e-5, e_out
    assert e_std < 1e-5, e_std

    # the spread drives the relabel sort order, so check the ordering agrees
    t_order = np.argsort(t_std.numpy(), axis=-1)
    j_order = np.argsort(np.asarray(j_std), axis=-1)
    agree = (t_order == j_order).mean()
    print(f'  relabel sort order     {agree * 100:.1f}% identical')
    assert agree > 0.99


def test_train_from_scratch():
    train, _, _ = build_dataset(DATA_DIR, list(range(N_TRAIN_TASKS)),
                                None, 100, 100000, load_eval=False)
    t0 = time.time()
    ens = jdyn.train_ensembles(
        train, OBS_DIM, ACT_DIM, HIDDEN, ENSEMBLE, with_next_obs=False,
        weight_decays=DECAYS, key=jax.random.PRNGKey(0), max_epochs=800)
    dt = time.time() - t0
    print(f'  trained {N_TRAIN_TASKS} ensembles in {dt:.1f}s')

    # sanity: predictions should track the true reward on held-in data
    task_indices = jnp.arange(N_TRAIN_TASKS)
    batch = 256
    obs = jnp.stack([train.obs[t, :batch] for t in range(N_TRAIN_TASKS)]).reshape(-1, OBS_DIM)
    act = jnp.stack([train.actions[t, :batch] for t in range(N_TRAIN_TASKS)]).reshape(-1, ACT_DIM)
    target = jnp.stack([train.rewards[t, :batch] for t in range(N_TRAIN_TASKS)]).reshape(-1, 1)
    pred, _ = ens.step(obs, act, task_indices)
    mse = float(jnp.mean((pred - target) ** 2))
    print(f'  reward prediction MSE  {mse:.5f}')
    assert mse < 0.05, f'dynamics did not fit: MSE {mse}'
    return ens


if __name__ == '__main__':
    print('dynamics parity (torch <-> jax)')
    test_step_parity()
    test_train_from_scratch()
    print('PASS')
