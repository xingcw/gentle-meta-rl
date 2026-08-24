"""Loading of the offline trajectories written by stage 2.

The torch pipeline carries two near-identical copies of this logic
(``pretrain_dynamics.experiment`` and ``OfflineMetaRLAlgorithm.init_buffer``);
stages 3 and 4 share this one instead.

Two on-disk layouts are read. The dense one is preferred:

    <data_dir>/goal_idx<i>/task_step<epoch>.npz

holding obs, actions, rewards, ep_lengths and last_next_obs for the whole
task in one file. The legacy layout is one object array per episode:

    <data_dir>/goal_idx<i>/trj_evalsample<n>_step<epoch>.npy

with [obs, action, reward, next_obs] rows. Both yield identical arrays, so
torch-generated roots and either JAX layout stay interchangeable.
"""
import glob
import os

import numpy as np
from flax import struct


class RunningMeanStd(object):
    """Port of rlkit.torch.pytorch_util.RunningMeanStd.

    The epsilon-seeded count matters: it makes the statistics differ slightly
    from a plain mean/var, and the dataset normalization must match torch's.
    """

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, arr):
        arr = np.asarray(arr)
        self.update_from_moments(arr.mean(axis=0), arr.var(axis=0), len(arr))

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = tot_count

    def forward(self, x, inverse=False):
        if inverse:
            return x * np.sqrt(self.var) + self.mean
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


@struct.dataclass
class TaskData:
    """Per-task transitions, stacked as (num_tasks, num_transitions, dim)."""
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminals: np.ndarray

    @property
    def num_tasks(self):
        return self.obs.shape[0]

    @property
    def num_transitions(self):
        return self.obs.shape[1]

    def context(self, use_next_obs_in_context):
        """[obs, action, reward] (+ next_obs), the context encoder's input."""
        parts = [self.obs, self.actions, self.rewards]
        if use_next_obs_in_context:
            parts.append(self.next_obs)
        return np.concatenate(parts, axis=-1)


def _task_files(data_dir, task_idx, n_trj, epoch):
    files = []
    for n in range(n_trj):
        pattern = os.path.join(data_dir, f'goal_idx{task_idx}',
                               'trj_evalsample%d_step%d.npy' % (n, epoch))
        files.extend(sorted(glob.glob(pattern)))
    return files


def task_npz(data_dir, task_idx, epoch):
    """Path of the dense one-file-per-task format."""
    return os.path.join(data_dir, f'goal_idx{task_idx}', f'task_step{epoch}.npz')


def _load_task_npz(path):
    """One task from the dense format, in the legacy loader's dtypes.

    next_obs is not stored: within an episode it is obs shifted by one, so
    only the final row of each episode needs keeping. terminals likewise
    follow from the episode lengths.
    """
    with np.load(path) as z:
        obs, act = z['obs'], z['actions']
        rew = z['rewards'].astype(np.float64).reshape(-1, 1)
        ep_lengths, tails = z['ep_lengths'], z['last_next_obs']
    next_obs = np.empty_like(obs)
    terminals = np.zeros((obs.shape[0], 1))
    start = 0
    for e, length in enumerate(ep_lengths):
        stop = start + int(length)
        next_obs[start:stop - 1] = obs[start + 1:stop]
        next_obs[stop - 1] = tails[e]
        terminals[stop - 1] = 1
        start = stop
    if start != obs.shape[0]:
        raise ValueError(f'{path}: ep_lengths sum to {start}, obs has {obs.shape[0]}')
    return obs, act, rew, next_obs, terminals


def load_tasks(data_dir, task_indices, n_trj, epoch):
    """Read every trajectory of the given tasks into stacked float32 arrays.

    Prefers the dense per-task npz and falls back to the per-episode object
    arrays, so roots written under either layout load identically.
    """
    obs, actions, rewards, next_obs, terminals = [], [], [], [], []
    for task_idx in task_indices:
        npz = task_npz(data_dir, task_idx, epoch)
        if os.path.exists(npz):
            t_obs, t_act, t_rew, t_next, t_term = _load_task_npz(npz)
            obs.append(t_obs)
            actions.append(t_act)
            rewards.append(t_rew)
            next_obs.append(t_next)
            terminals.append(t_term)
            continue
        files = _task_files(data_dir, task_idx, n_trj, epoch)
        if not files:
            raise FileNotFoundError(
                f'no trajectories for task {task_idx} under {data_dir} '
                f'(n_trj={n_trj}, epoch={epoch})')
        t_obs, t_act, t_rew, t_next, t_term = [], [], [], [], []
        for path in files:
            trj = np.load(path, allow_pickle=True)
            t_obs.append(np.stack(trj[:, 0]))
            t_act.append(np.stack(trj[:, 1]))
            t_rew.append(np.asarray(list(trj[:, 2]), dtype=np.float64).reshape(-1, 1))
            t_next.append(np.stack(trj[:, 3]))
            term = np.zeros((trj.shape[0], 1))
            term[-1] = 1
            t_term.append(term)
        obs.append(np.concatenate(t_obs))
        actions.append(np.concatenate(t_act))
        rewards.append(np.concatenate(t_rew))
        next_obs.append(np.concatenate(t_next))
        terminals.append(np.concatenate(t_term))

    lengths = {a.shape[0] for a in obs}
    if len(lengths) != 1:
        raise ValueError(f'tasks have differing transition counts: {sorted(lengths)}')
    return [np.stack(x) for x in (obs, actions, rewards, next_obs, terminals)]


def build_dataset(data_dir, train_tasks, eval_tasks, n_trj, train_epoch,
                  eval_epoch=None, load_eval=True):
    """Load train (and eval) tasks and z-score observations.

    The normalizer is fitted on the training tasks only and then applied to
    both splits, matching OfflineMetaRLAlgorithm.init_buffer.
    """
    tr = load_tasks(data_dir, train_tasks, n_trj, train_epoch)
    obs_dim = tr[0].shape[-1]

    normalizer = RunningMeanStd(shape=obs_dim)
    normalizer.update(tr[0].reshape(-1, obs_dim))

    def finish(arrays):
        obs, actions, rewards, next_obs, terminals = arrays
        return TaskData(
            obs=normalizer.forward(obs).astype(np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            next_obs=normalizer.forward(next_obs).astype(np.float32),
            terminals=terminals.astype(np.float32),
        )

    train_data = finish(tr)
    eval_data = None
    if load_eval:
        ev = load_tasks(data_dir, eval_tasks, n_trj,
                        train_epoch if eval_epoch is None else eval_epoch)
        eval_data = finish(ev)
    return train_data, eval_data, normalizer
