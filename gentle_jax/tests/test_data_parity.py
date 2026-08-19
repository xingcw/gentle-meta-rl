"""Phase C verification: the shared loader must reproduce the torch buffers.

Compares the JAX loader's normalizer and normalized transitions against
rlkit's RunningMeanStd fed the same trajectories.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gentle_jax  # noqa: F401

import numpy as np

from gentle_jax.data import RunningMeanStd, build_dataset, load_tasks
from rlkit.paths import data_dir_for
from rlkit.torch import pytorch_util as ptu

DATA_DIR = data_dir_for('point-robot')
TRAIN_TASKS = list(range(10))
EVAL_TASKS = list(range(10, 20))
N_TRJ = 100
EPOCH = 100000


def test_running_mean_std():
    rng = np.random.RandomState(0)
    arr = rng.randn(5000, 2) * [2.0, 0.5] + [1.0, -3.0]

    mine = RunningMeanStd(shape=2)
    mine.update(arr)
    theirs = ptu.RunningMeanStd(shape=2)
    theirs.update(list(arr))

    e_mean = np.abs(mine.mean - theirs.mean).max()
    e_var = np.abs(mine.var - theirs.var).max()
    e_fwd = np.abs(mine.forward(arr) - theirs.forward(arr)).max()
    print(f'  RunningMeanStd         mean err {e_mean:.2e}  var err {e_var:.2e}  '
          f'forward err {e_fwd:.2e}')
    assert max(e_mean, e_var, e_fwd) < 1e-12


def test_dataset_shapes_and_stats():
    train, eval_, norm = build_dataset(DATA_DIR, TRAIN_TASKS, EVAL_TASKS, N_TRJ, EPOCH)
    print(f'  train                  {train.obs.shape} obs, {train.num_transitions} '
          f'transitions/task')
    print(f'  eval                   {eval_.obs.shape}')
    assert train.obs.shape == (10, 2000, 2), train.obs.shape
    assert eval_.obs.shape == (10, 2000, 2)
    assert train.rewards.shape == (10, 2000, 1)

    # one terminal per 20-step trajectory
    assert train.terminals.sum() == 10 * N_TRJ

    # normalizer fitted on train tasks only -> normalized train obs are ~N(0,1)
    flat = train.obs.reshape(-1, 2)
    print(f'  normalized train obs   mean {flat.mean(axis=0)}  std {flat.std(axis=0)}')
    assert np.abs(flat.mean(axis=0)).max() < 1e-3
    assert np.abs(flat.std(axis=0) - 1.0).max() < 1e-3
    print(f'  normalizer             mean {norm.mean}  var {norm.var}')


def test_matches_torch_loader():
    """Replicate init_buffer's list-based load and compare against ours."""
    raw = load_tasks(DATA_DIR, TRAIN_TASKS, N_TRJ, EPOCH)
    obs_lst = list(raw[0].reshape(-1, 2))

    theirs = ptu.RunningMeanStd(shape=2)
    theirs.update(obs_lst)
    torch_obs = theirs.forward(raw[0])

    train, _, mine = build_dataset(DATA_DIR, TRAIN_TASKS, EVAL_TASKS, N_TRJ, EPOCH,
                                   load_eval=False)
    e_stats = max(np.abs(mine.mean - theirs.mean).max(), np.abs(mine.var - theirs.var).max())
    e_obs = np.abs(train.obs - torch_obs.astype(np.float32)).max()
    print(f'  vs torch init_buffer   stats err {e_stats:.2e}  obs err {e_obs:.2e}')
    assert e_stats < 1e-12
    assert e_obs < 1e-6


if __name__ == '__main__':
    print('data loader parity')
    test_running_mean_std()
    test_dataset_shapes_and_stats()
    test_matches_torch_loader()
    print('PASS')
