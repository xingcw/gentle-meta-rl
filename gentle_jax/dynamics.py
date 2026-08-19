"""Stage 3: per-task dynamics ensembles.

Port of rlkit.torch.multi_task_dynamics.MultiTaskDynamics. The torch version
holds a Python list of per-task models and loops over it; here the per-task
parameters are stacked on a leading axis and every task trains -- and later
predicts -- under a single `jax.vmap`.

Training keeps the torch early-stopping rule: a member is "saved" whenever its
holdout loss improves by more than 1%, and a task stops after
`max_epochs_since_update` epochs without any member improving. Because tasks
stop at different epochs, the scan runs to `max_epochs` and freezes each task's
parameters once it is done.
"""
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from gentle_jax.networks import EnsembleDynamics, dynamics_params_from_torch

DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 256
HOLDOUT_RATIO = 0.2
MAX_HOLDOUT = 1000
IMPROVEMENT_THRESHOLD = 0.01
MAX_EPOCHS_SINCE_UPDATE = 5


@struct.dataclass
class DynamicsEnsembles:
    """Stacked ensembles: every leaf carries a leading task axis."""
    params: dict
    num_tasks: int = struct.field(pytree_node=False)
    obs_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    hidden_dims: tuple = struct.field(pytree_node=False)
    num_ensemble: int = struct.field(pytree_node=False)
    with_next_obs: bool = struct.field(pytree_node=False)

    @property
    def module(self):
        return EnsembleDynamics(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            num_ensemble=self.num_ensemble,
            with_next_obs=self.with_next_obs,
        )

    def step(self, obs, actions, task_indices):
        """Ensemble-mean prediction and disagreement, as MultiTaskDynamics.step.

        `obs`/`actions` are (num_tasks * batch, dim) laid out task-major, and
        `task_indices` selects which task's ensemble each block uses. Returns
        the mean prediction (num_tasks * batch, out_dim) and the per-sample
        ensemble spread (num_tasks, batch).
        """
        obs_act = jnp.concatenate([obs, actions], axis=-1)
        n_tasks = task_indices.shape[0]
        batch = obs_act.shape[0] // n_tasks
        obs_act = obs_act.reshape(n_tasks, batch, -1)

        module = self.module
        task_params = jax.tree_util.tree_map(lambda p: p[task_indices], self.params)

        def one_task(params, x):
            out = module.apply({'params': params}, x)          # (ensemble, batch, out)
            # torch.std defaults to the unbiased estimator; jnp.std does not.
            spread = jnp.std(out, axis=0, ddof=1).sum(axis=-1)  # (batch,)
            return out.mean(axis=0), spread

        mean, spread = jax.vmap(one_task)(task_params, obs_act)
        return mean.reshape(n_tasks * batch, -1), spread


def _decay_loss(params, weight_decays, num_hidden):
    names = [f'backbones_{i}_weight' for i in range(num_hidden)] + ['output_layer_weight']
    return sum(wd * 0.5 * jnp.sum(params[name] ** 2)
               for name, wd in zip(names, weight_decays))


def _make_apply(module):
    def apply(params, x):
        return module.apply({'params': params}, x)
    return apply


def train_ensembles(data, obs_dim, action_dim, hidden_dims, num_ensemble,
                    with_next_obs, weight_decays, key, max_epochs=800,
                    batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR, verbose=True):
    """Train one ensemble per task. `data` is a gentle_jax.data.TaskData."""
    inputs = jnp.concatenate([data.obs, data.actions], axis=-1)
    targets = (jnp.concatenate([data.rewards, data.next_obs], axis=-1)
               if with_next_obs else data.rewards)

    num_tasks, data_size, _ = inputs.shape
    holdout_size = min(int(data_size * HOLDOUT_RATIO), MAX_HOLDOUT)
    train_size = data_size - holdout_size
    num_hidden = len(hidden_dims)

    module = EnsembleDynamics(obs_dim=obs_dim, action_dim=action_dim,
                              hidden_dims=tuple(hidden_dims),
                              num_ensemble=num_ensemble, with_next_obs=with_next_obs)
    apply = _make_apply(module)
    tx = optax.adam(lr)

    key, init_key, split_key = jax.random.split(key, 3)
    init_keys = jax.random.split(init_key, num_tasks)
    dummy = jnp.zeros((1, obs_dim + action_dim))
    params = jax.vmap(lambda k: module.init(k, dummy)['params'])(init_keys)
    opt_state = jax.vmap(tx.init)(params)

    # Per-task train/holdout split, matching torch.utils.data.random_split.
    split_keys = jax.random.split(split_key, num_tasks)
    perm = jax.vmap(lambda k: jax.random.permutation(k, data_size))(split_keys)
    train_idx, holdout_idx = perm[:, :train_size], perm[:, train_size:]
    take = jax.vmap(lambda a, i: a[i])
    train_inputs, train_targets = take(inputs, train_idx), take(targets, train_idx)
    holdout_inputs, holdout_targets = take(inputs, holdout_idx), take(targets, holdout_idx)

    n_full = train_size // batch_size
    remainder = train_size % batch_size

    def loss_fn(p, x, y):
        pred = apply(p, x)
        # average over batch and dim, sum over ensemble members
        mse = ((pred - y) ** 2).mean(axis=(1, 2)).sum()
        return mse + _decay_loss(p, weight_decays, num_hidden)

    def sgd_step(carry, batch):
        p, o = carry
        x, y = batch
        loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
        updates, o = tx.update(grads, o)
        return (optax.apply_updates(p, updates), o), loss

    def train_epoch(p, o, idxes, x_all, y_all):
        # idxes is (ensemble, train_size): each member sees its own bootstrap.
        x, y = x_all[idxes], y_all[idxes]
        head = n_full * batch_size
        xb = x[:, :head].reshape(num_ensemble, n_full, batch_size, -1)
        yb = y[:, :head].reshape(num_ensemble, n_full, batch_size, -1)
        (p, o), losses = jax.lax.scan(
            sgd_step, (p, o),
            (jnp.swapaxes(xb, 0, 1), jnp.swapaxes(yb, 0, 1)))
        losses = [losses]
        if remainder:
            # torch's final short batch carries the same weight as a full one
            (p, o), last = sgd_step((p, o), (x[:, head:], y[:, head:]))
            losses.append(jnp.atleast_1d(last))
        return p, o, jnp.concatenate(losses).mean()

    def validate(p, x, y):
        return ((apply(p, x) - y) ** 2).mean(axis=(1, 2))

    def run_task(p, o, x_tr, y_tr, x_ho, y_ho, task_key):
        def epoch_body(carry, epoch_key):
            p, o, idxes, best, saved, cnt, done = carry
            new_p, new_o, train_loss = train_epoch(p, o, idxes, x_tr, y_tr)
            new_holdout = validate(new_p, x_ho, y_ho)

            improved = ((best - new_holdout) / best) > IMPROVEMENT_THRESHOLD
            live = jnp.logical_not(done)
            keep = jnp.logical_and(improved, live)

            best = jnp.where(keep, new_holdout, best)
            saved = jax.tree_util.tree_map(
                lambda s, n: jnp.where(
                    keep.reshape((-1,) + (1,) * (n.ndim - 1)), n, s), saved, new_p)
            cnt = jnp.where(live, jnp.where(improved.any(), 0, cnt + 1), cnt)
            # a finished task keeps its parameters and optimizer state frozen
            p, o = jax.tree_util.tree_map(
                lambda old, new: jnp.where(live, new, old), (p, o), (new_p, new_o))
            done = jnp.logical_or(done, cnt >= MAX_EPOCHS_SINCE_UPDATE)

            # reshuffle each member's bootstrap indices, as torch's shuffle_rows
            idxes = jax.random.permutation(epoch_key, idxes, axis=1, independent=True)
            return ((p, o, idxes, best, saved, cnt, done),
                    (train_loss, new_holdout.mean(), done))

        boot_key, scan_key = jax.random.split(task_key)
        idxes = jax.random.randint(boot_key, (num_ensemble, train_size), 0, train_size)
        best = jnp.full((num_ensemble,), 1e10)
        carry = (p, o, idxes, best, p, jnp.array(0), jnp.array(False))
        carry, traces = jax.lax.scan(
            epoch_body, carry, jax.random.split(scan_key, max_epochs))
        return carry[4], carry[6], traces  # saved params, done flag, traces

    key, run_key = jax.random.split(key)
    task_keys = jax.random.split(run_key, num_tasks)
    saved, done, traces = jax.vmap(run_task)(
        params, opt_state, train_inputs, train_targets,
        holdout_inputs, holdout_targets, task_keys)

    if verbose:
        train_loss, holdout_loss, done_trace = traces
        stop_epoch = np.asarray(jnp.argmax(done_trace, axis=1))
        for t in range(num_tasks):
            last = int(stop_epoch[t]) if bool(done[t]) else max_epochs - 1
            print(f'  task {t:2d}  stopped epoch {last:3d}  '
                  f'train {float(train_loss[t, last]):.5f}  '
                  f'holdout {float(holdout_loss[t, last]):.5f}')
        if not bool(jnp.all(done)):
            print(f'  WARNING: {int((~done).sum())} task(s) hit max_epochs={max_epochs} '
                  f'without early stopping')

    return DynamicsEnsembles(
        params=saved, num_tasks=num_tasks, obs_dim=obs_dim, action_dim=action_dim,
        hidden_dims=tuple(hidden_dims), num_ensemble=num_ensemble,
        with_next_obs=with_next_obs)


def save(ensembles, path):
    os.makedirs(path, exist_ok=True)
    flat = {k: np.asarray(v) for k, v in ensembles.params.items()}
    np.savez(os.path.join(path, 'dynamics_jax.npz'), **flat)


def load(path, num_tasks, obs_dim, action_dim, hidden_dims, num_ensemble, with_next_obs):
    blob = np.load(os.path.join(path, 'dynamics_jax.npz'))
    params = {k: jnp.asarray(blob[k]) for k in blob.files}
    return DynamicsEnsembles(
        params=params, num_tasks=num_tasks, obs_dim=obs_dim, action_dim=action_dim,
        hidden_dims=tuple(hidden_dims), num_ensemble=num_ensemble,
        with_next_obs=with_next_obs)


def load_from_torch(path, num_tasks, obs_dim, action_dim, hidden_dims, num_ensemble,
                    with_next_obs):
    """Read the torch stage-3 checkpoints (task<i>_dynamics.pth) into stacked params.

    Lets the JAX stage 4 run against a dataset and ensembles produced by the
    torch pipeline, which is what the regression test needs.
    """
    import torch

    per_task = []
    for i in range(num_tasks):
        sd = torch.load(os.path.join(path, f'task{i}_dynamics.pth'),
                        map_location='cpu', weights_only=True)
        per_task.append(dynamics_params_from_torch(sd, len(hidden_dims)))
    params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *per_task)
    return DynamicsEnsembles(
        params=params, num_tasks=num_tasks, obs_dim=obs_dim, action_dim=action_dim,
        hidden_dims=tuple(hidden_dims), num_ensemble=num_ensemble,
        with_next_obs=with_next_obs)
