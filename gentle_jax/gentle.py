"""Stage 4: GENTLE meta-training.

Port of rlkit.torch.algo.gentle.GENTLE together with the parts of
rlkit.core.rl_algorithm.OfflineMetaRLAlgorithm that it actually exercises.

Structure of one training step, mirroring GENTLE._take_step:

1. z = mean over the context batch of encoder(context)
2. the reconstruction loss updates the encoder and decoder jointly
3. the TD3 critic loss updates qf1/qf2, reading z as a constant
4. every `policy_freq` steps the BC-regularized actor loss updates the policy
   and all three target networks move by `soft_target_tau`

Steps 3 and 4 use the z computed in step 1, i.e. the pre-update encoder, which
is what torch does: `task_z` is a tensor computed before the context optimizer
steps, and it enters the critic and actor losses detached.

Two data structures replace the torch replay buffers. The encoder buffer holds
indices into the offline dataset rather than copies of the transitions, because
that is all the torch version ever puts in it: transitions resampled uniformly
from the same task's offline data. The relabel buffer is a fixed-capacity array
per task; capacity cannot be exceeded, since a task receives one block from the
positive pass and at most one from each of the other tasks.
"""
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from gentle_jax.networks import (Mlp, MlpDecoder, TanhGaussianPolicy,
                                 tanh_normal_sample)


@struct.dataclass
class GentleConfig:
    obs_dim: int = struct.field(pytree_node=False, default=2)
    action_dim: int = struct.field(pytree_node=False, default=2)
    latent_dim: int = struct.field(pytree_node=False, default=5)
    net_size: int = struct.field(pytree_node=False, default=64)
    num_train_tasks: int = struct.field(pytree_node=False, default=10)
    meta_batch: int = struct.field(pytree_node=False, default=10)
    batch_size: int = struct.field(pytree_node=False, default=256)
    embedding_batch_size: int = struct.field(pytree_node=False, default=256)
    num_iterations: int = struct.field(pytree_node=False, default=500)
    num_train_steps_per_itr: int = struct.field(pytree_node=False, default=400)
    num_initial_steps: int = struct.field(pytree_node=False, default=200)
    num_tasks_sample: int = struct.field(pytree_node=False, default=5)
    num_steps_prior: int = struct.field(pytree_node=False, default=400)
    max_path_length: int = struct.field(pytree_node=False, default=20)
    num_steps_per_eval: int = struct.field(pytree_node=False, default=600)
    online_sample_num: int = struct.field(pytree_node=False, default=20)
    discount: float = 0.9
    reward_scale: float = 5.0
    soft_target_tau: float = 0.005
    policy_lr: float = 3e-4
    qf_lr: float = 3e-4
    context_lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = struct.field(pytree_node=False, default=2)
    bc_weight: float = 2.5
    recon_loss_weight: float = 10.0
    relabel_data_ratio: float = 0.95
    num_aug_neg_tasks: int = struct.field(pytree_node=False, default=3)
    relabel_sample_mult: int = struct.field(pytree_node=False, default=10)
    relabel_capacity_blocks: int = struct.field(pytree_node=False, default=0)
    use_next_obs_in_context: bool = struct.field(pytree_node=False, default=False)
    reward_in_context: bool = struct.field(pytree_node=False, default=True)

    @property
    def context_dim(self):
        base = self.obs_dim + self.action_dim + 1
        return base + self.obs_dim if self.use_next_obs_in_context else base

    @property
    def enc_in_dim(self):
        # context rows always store reward; the encoder may not see it
        return self.context_dim - (0 if self.reward_in_context else 1)

    @property
    def relabel_batch(self):
        return self.embedding_batch_size * self.relabel_sample_mult

    @property
    def relabel_size(self):
        return int(self.embedding_batch_size * self.relabel_data_ratio)

    @property
    def context_from_buffer(self):
        return self.embedding_batch_size - self.relabel_size

    @property
    def relabel_capacity(self):
        """Rows reserved per task in the relabel buffer.

        The worst case is one positive block plus one from every other task,
        which makes the buffer O(num_train_tasks^2) and is only reachable if
        every task happens to draw the same negative. A task actually
        receives 1 + Poisson(num_aug_neg_tasks) blocks, so past a few dozen
        tasks the worst-case reservation is orders of magnitude larger than
        anything that gets written -- 142 GB at 470 tasks against ~4 blocks
        of expected occupancy.

        `relabel_capacity_blocks` sizes for the tail of that Poisson instead:
        at num_aug_neg_tasks=3, 16 blocks leaves a 1.2e-7 chance of one task
        overflowing in one iteration. 0 keeps the worst-case sizing, so
        existing configs are unaffected.
        """
        blocks = self.relabel_capacity_blocks or self.num_train_tasks
        return self.relabel_batch * blocks


@struct.dataclass
class GentleState:
    encoder: TrainState
    qf1: TrainState
    qf2: TrainState
    policy: TrainState
    target_qf1: dict
    target_qf2: dict
    target_policy: dict
    enc_idx: jnp.ndarray      # (num_tasks, num_steps_prior) into the offline data
    enc_size: jnp.ndarray     # (num_tasks,) valid prefix of enc_idx
    relabel_ctx: jnp.ndarray  # (num_tasks, relabel_capacity, context_dim)
    relabel_len: jnp.ndarray  # (num_tasks,)
    step: jnp.ndarray
    rng: jnp.ndarray


class Nets(object):
    """The five modules, built once and reused by every jitted function."""

    def __init__(self, cfg):
        h = (cfg.net_size,) * 3
        self.encoder = Mlp(hidden_sizes=h, output_size=cfg.latent_dim,
                           output_activation='tanh')
        # column to hold out of the encoder input; None keeps the full row
        self.reward_col = (None if cfg.reward_in_context
                           else cfg.obs_dim + cfg.action_dim)
        self.decoder = MlpDecoder(hidden_size=cfg.net_size, num_hidden_layers=3,
                                  obs_dim=cfg.obs_dim,
                                  use_next_obs_in_context=cfg.use_next_obs_in_context)
        self.qf = Mlp(hidden_sizes=h, output_size=1)
        self.policy = TanhGaussianPolicy(hidden_sizes=h, action_dim=cfg.action_dim)

    def embed(self, encoder_params, context):
        """Per-transition embeddings; every encoder application goes through here."""
        if self.reward_col is not None:
            context = jnp.concatenate([context[..., :self.reward_col],
                                       context[..., self.reward_col + 1:]],
                                      axis=-1)
        return self.encoder.apply({'params': encoder_params}, context)

    def infer_z(self, encoder_params, context):
        """q(z|c) with the deterministic (non-bottleneck) encoder.

        z is the mean of the per-transition embeddings; the spread is reported
        with the unbiased estimator, as torch.std does.
        """
        params = self.embed(encoder_params, context)
        return params.mean(axis=1), jnp.std(params, axis=1, ddof=1)

    def act(self, policy_params, obs, z, key=None, deterministic=False):
        mean, log_std = self.policy.apply({'params': policy_params},
                                          jnp.concatenate([obs, z], axis=-1))
        if deterministic:
            return jnp.tanh(mean), None
        action, log_prob, _ = tanh_normal_sample(key, mean, log_std)
        return action, log_prob


def create_state(cfg, nets, key):
    keys = jax.random.split(key, 6)
    ctx = jnp.zeros((1, 1, cfg.enc_in_dim))
    obs_z = jnp.zeros((1, cfg.obs_dim + cfg.latent_dim))
    qf_in = jnp.zeros((1, cfg.obs_dim + cfg.action_dim + cfg.latent_dim))

    encoder_params = nets.encoder.init(keys[0], ctx)['params']
    decoder_params = nets.decoder.init(
        keys[1], jnp.zeros((1, 1, cfg.obs_dim)), jnp.zeros((1, 1, cfg.action_dim)),
        jnp.zeros((1, 1, cfg.latent_dim)))['params']
    qf1_params = nets.qf.init(keys[2], qf_in)['params']
    qf2_params = nets.qf.init(keys[3], qf_in)['params']
    policy_params = nets.policy.init(keys[4], obs_z)['params']

    # torch puts the encoder and decoder under one optimizer (itertools.chain)
    encoder = TrainState.create(
        apply_fn=None, params={'encoder': encoder_params, 'decoder': decoder_params},
        tx=optax.adam(cfg.context_lr))
    qf1 = TrainState.create(apply_fn=None, params=qf1_params, tx=optax.adam(cfg.qf_lr))
    qf2 = TrainState.create(apply_fn=None, params=qf2_params, tx=optax.adam(cfg.qf_lr))
    policy = TrainState.create(apply_fn=None, params=policy_params,
                               tx=optax.adam(cfg.policy_lr))

    return GentleState(
        encoder=encoder,
        qf1=qf1, qf2=qf2, policy=policy,
        target_qf1=qf1_params, target_qf2=qf2_params, target_policy=policy_params,
        enc_idx=jnp.zeros((cfg.num_train_tasks, cfg.num_steps_prior), jnp.int32),
        enc_size=jnp.zeros((cfg.num_train_tasks,), jnp.int32),
        relabel_ctx=jnp.zeros((cfg.num_train_tasks, cfg.relabel_capacity,
                               cfg.context_dim)),
        relabel_len=jnp.zeros((cfg.num_train_tasks,), jnp.int32),
        step=jnp.array(0, jnp.int32),
        rng=keys[5],
    )


def soft_update(params, target_params, tau):
    return jax.tree_util.tree_map(lambda p, t: tau * p + (1 - tau) * t,
                                  params, target_params)


# --------------------------------------------------------------------------
# encoder replay buffer
# --------------------------------------------------------------------------

def init_enc_buffer(state, cfg, num_transitions, key):
    """Fill every task's encoder buffer, as the it_ == 0 branch of train()."""
    idx = jax.random.randint(key, (cfg.num_train_tasks, cfg.num_steps_prior),
                             0, num_transitions)
    return state.replace(
        enc_idx=idx,
        enc_size=jnp.full((cfg.num_train_tasks,), cfg.num_initial_steps, jnp.int32),
    )


def resample_enc_buffer(state, cfg, num_transitions, key):
    """Clear and refill `num_tasks_sample` randomly chosen tasks.

    torch draws the tasks one at a time with replacement, so a task can be
    refilled twice in an iteration; only the last fill survives, which is what
    the sequential scatter below reproduces.
    """
    task_key, data_key = jax.random.split(key)
    tasks = jax.random.randint(task_key, (cfg.num_tasks_sample,), 0,
                               cfg.num_train_tasks)
    draws = jax.random.randint(data_key,
                               (cfg.num_tasks_sample, cfg.num_steps_prior),
                               0, num_transitions)

    def one(carry, xs):
        enc_idx, enc_size = carry
        task, row = xs
        enc_idx = enc_idx.at[task].set(row)
        enc_size = enc_size.at[task].set(cfg.num_steps_prior)
        return (enc_idx, enc_size), None

    (enc_idx, enc_size), _ = jax.lax.scan(
        one, (state.enc_idx, state.enc_size), (tasks, draws))
    return state.replace(enc_idx=enc_idx, enc_size=enc_size)


def sample_context_from_enc(state, cfg, context_data, indices, batch, key):
    """Draw `batch` context rows per task from the encoder buffer.

    One two-axis gather rather than a vmapped `arr[task][rows]`, which builds
    two chained gathers for the same result.
    """
    n = indices.shape[0]
    pos = jax.random.randint(key, (n, batch), 0, state.enc_size[indices][:, None])
    rows = state.enc_idx[indices[:, None], pos]
    return context_data[indices[:, None], rows]


# --------------------------------------------------------------------------
# relabeling
# --------------------------------------------------------------------------

def make_relabel(state, cfg, nets, context_data, dynamics, key):
    """Rebuild the relabel buffer, as GENTLE.make_relabel.

    The target policy relabels actions for every task's own data, the task's
    own dynamics relabel the rewards (the positive block), and
    `num_aug_neg_tasks` other tasks relabel the same transitions to give
    negative blocks.
    """
    n_tasks, n_ctx = cfg.num_train_tasks, cfg.relabel_batch
    obs_dim, act_dim = cfg.obs_dim, cfg.action_dim
    tasks = jnp.arange(n_tasks)

    sample_key, neg_key = jax.random.split(key)
    num_transitions = context_data.shape[1]
    rows = jax.random.randint(sample_key, (n_tasks, n_ctx), 0, num_transitions)
    context = context_data[tasks[:, None], rows]                      # (T, n_ctx, C)

    # actions from the target policy under the posterior of this context
    z, _ = nets.infer_z(state.encoder.params['encoder'], context)     # (T, latent)
    z_rep = jnp.repeat(z, n_ctx, axis=0)                              # (T*n_ctx, latent)
    obs_flat = context[..., :obs_dim].reshape(-1, obs_dim)
    actions, _ = nets.act(state.target_policy, obs_flat, z_rep, deterministic=True)

    base = context.at[..., obs_dim:obs_dim + act_dim].set(
        actions.reshape(n_tasks, n_ctx, act_dim))

    def relabel_with(model_tasks):
        """Predict rewards for each block with the given task's ensemble."""
        pred, spread = dynamics.step(obs_flat, actions, model_tasks)
        out = base.at[..., obs_dim + act_dim:].set(pred.reshape(n_tasks, n_ctx, -1))
        order = jnp.argsort(spread, axis=-1)
        return jnp.take_along_axis(out, order[..., None], axis=1)

    ctx = jnp.zeros((n_tasks, cfg.relabel_capacity, cfg.context_dim))
    lengths = jnp.zeros((n_tasks,), jnp.int32)

    def append(carry, xs):
        ctx, lengths = carry
        target, block = xs
        ctx = jax.lax.dynamic_update_slice(
            ctx, block[None], (target, lengths[target], 0))
        # dynamic_update_slice clamps a write that would run off the end, so
        # an overflowing block lands on top of the previous one rather than
        # erroring; clamp the length to match, otherwise sample_relabel would
        # draw positions past the rows that exist.
        lengths = jnp.minimum(lengths.at[target].add(block.shape[0]),
                              cfg.relabel_capacity)
        return (ctx, lengths), None

    positive = relabel_with(tasks)
    (ctx, lengths), _ = jax.lax.scan(append, (ctx, lengths), (tasks, positive))

    # for each task, `num_aug_neg_tasks` distinct other tasks
    neg_keys = jax.random.split(neg_key, n_tasks)

    def pick_negatives(task, k):
        others = jnp.where(jnp.arange(n_tasks) < task, jnp.arange(n_tasks),
                           jnp.arange(n_tasks) + 1)[:n_tasks - 1]
        return jax.random.choice(k, others, (cfg.num_aug_neg_tasks,), replace=False)

    negatives = jax.vmap(pick_negatives)(tasks, neg_keys)  # (T, num_aug)

    def one_pass(carry, p):
        ctx, lengths = carry
        model_tasks = negatives[:, p]
        blocks = relabel_with(model_tasks)
        (ctx, lengths), _ = jax.lax.scan(append, (ctx, lengths),
                                         (model_tasks, blocks))
        return (ctx, lengths), None

    (ctx, lengths), _ = jax.lax.scan(one_pass, (ctx, lengths),
                                     jnp.arange(cfg.num_aug_neg_tasks))
    return state.replace(relabel_ctx=ctx, relabel_len=lengths)


def sample_relabel(state, cfg, indices, batch, key):
    n = indices.shape[0]
    pos = jax.random.randint(key, (n, batch), 0, state.relabel_len[indices][:, None])
    return state.relabel_ctx[indices[:, None], pos]


# --------------------------------------------------------------------------
# training step
# --------------------------------------------------------------------------

def train_step(state, cfg, nets, data, context_data, key):
    """Sample a meta-batch and take one GENTLE gradient step."""
    n_tasks, b = cfg.meta_batch, cfg.batch_size
    k_perm, k_ctx, k_relabel, k_sac, k_step = jax.random.split(key, 5)

    # torch draws meta_batch tasks without replacement; with meta_batch equal to
    # the task count this is a permutation.
    indices = jax.random.permutation(k_perm, cfg.num_train_tasks)[:n_tasks]

    ctx_buffer = sample_context_from_enc(state, cfg, context_data, indices,
                                         cfg.context_from_buffer, k_ctx)
    ctx_relabel = sample_relabel(state, cfg, indices, cfg.relabel_size, k_relabel)
    context = jnp.concatenate([ctx_buffer, ctx_relabel], axis=1)

    # RL batch, always from the full offline data
    num_transitions = data.obs.shape[1]
    rows = jax.random.randint(k_sac, (n_tasks, b), 0, num_transitions)
    task_rows = indices[:, None]
    batch = (data.obs[task_rows, rows], data.actions[task_rows, rows],
             data.rewards[task_rows, rows], data.next_obs[task_rows, rows],
             data.terminals[task_rows, rows])
    return apply_train_step(state, cfg, nets, context, batch, k_step)


def apply_train_step(state, cfg, nets, context, batch, key):
    """The gradient step itself, on an already-sampled context and RL batch.

    Split out from `train_step` so the parity test can feed it the same fixed
    batch it feeds torch's GENTLE._take_step.
    """
    obs_dim, act_dim = cfg.obs_dim, cfg.action_dim
    n_tasks, b = cfg.meta_batch, cfg.batch_size
    k_act, k_noise = jax.random.split(key, 2)
    obs, actions, rewards, next_obs, terminals = batch

    z, z_vars = nets.infer_z(state.encoder.params['encoder'], context)
    z_rep = jnp.repeat(z, b, axis=0)
    flat = lambda x, d: x.reshape(n_tasks * b, d)
    obs_f, act_f = flat(obs, obs_dim), flat(actions, act_dim)
    next_obs_f = flat(next_obs, obs_dim)
    rewards_f, terms_f = flat(rewards, 1), flat(terminals, 1)

    # --- 1. encoder + decoder on the reconstruction loss -------------------
    target_recon = context[..., obs_dim + act_dim:]

    def context_loss_fn(params):
        z_grad, _ = nets.infer_z(params['encoder'], context)
        z_ctx = jnp.repeat(z_grad, context.shape[1], axis=0).reshape(
            context.shape[0], context.shape[1], -1)
        pred = nets.decoder.apply({'params': params['decoder']},
                                  context[..., :obs_dim],
                                  context[..., obs_dim:obs_dim + act_dim], z_ctx)
        recon = jnp.mean((target_recon - pred) ** 2)
        return cfg.recon_loss_weight * recon, recon

    (_, recon_loss), ctx_grads = jax.value_and_grad(context_loss_fn, has_aux=True)(
        state.encoder.params)
    encoder = state.encoder.apply_gradients(grads=ctx_grads)

    # --- 2. critics --------------------------------------------------------
    next_actions, _ = nets.act(state.target_policy, next_obs_f, z_rep,
                               deterministic=True)
    noise = jnp.clip(jax.random.normal(k_noise, next_actions.shape) * cfg.policy_noise,
                     -cfg.noise_clip, cfg.noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    qf_in_next = jnp.concatenate([next_obs_f, next_actions, z_rep], axis=-1)
    target_q = jnp.minimum(
        nets.qf.apply({'params': state.target_qf1}, qf_in_next),
        nets.qf.apply({'params': state.target_qf2}, qf_in_next))
    target_q = rewards_f * cfg.reward_scale + (1.0 - terms_f) * cfg.discount * target_q
    target_q = jax.lax.stop_gradient(target_q)

    qf_in = jnp.concatenate([obs_f, act_f, z_rep], axis=-1)

    def qf_loss_fn(p1, p2):
        q1 = nets.qf.apply({'params': p1}, qf_in)
        q2 = nets.qf.apply({'params': p2}, qf_in)
        loss = jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)
        return loss, (q1, q2)

    (qf_loss, (q1_pred, q2_pred)), (g1, g2) = jax.value_and_grad(
        qf_loss_fn, argnums=(0, 1), has_aux=True)(state.qf1.params, state.qf2.params)
    qf1 = state.qf1.apply_gradients(grads=g1)
    qf2 = state.qf2.apply_gradients(grads=g2)

    # --- 3. actor, every policy_freq steps ---------------------------------
    def policy_loss_fn(params):
        new_actions, _ = nets.act(params, obs_f, z_rep, key=k_act)
        q_in = jnp.concatenate([obs_f, new_actions, z_rep], axis=-1)
        q = jnp.minimum(nets.qf.apply({'params': qf1.params}, q_in),
                        nets.qf.apply({'params': qf2.params}, q_in))
        lmbda = cfg.bc_weight / jax.lax.stop_gradient(jnp.abs(q).mean())
        bc_loss = jnp.mean((new_actions - act_f) ** 2)
        return -lmbda * q.mean() + bc_loss, bc_loss

    def do_actor_update(_):
        (loss, bc), grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(
            state.policy.params)
        policy = state.policy.apply_gradients(grads=grads)
        return (policy,
                soft_update(qf1.params, state.target_qf1, cfg.soft_target_tau),
                soft_update(qf2.params, state.target_qf2, cfg.soft_target_tau),
                soft_update(policy.params, state.target_policy, cfg.soft_target_tau),
                loss, bc)

    def skip_actor_update(_):
        return (state.policy, state.target_qf1, state.target_qf2,
                state.target_policy, jnp.array(0.0), jnp.array(0.0))

    actor_updated = state.step % cfg.policy_freq == 0
    policy, tq1, tq2, tp, policy_loss, bc_loss = jax.lax.cond(
        actor_updated, do_actor_update, skip_actor_update, None)

    metrics = {
        'recon_loss': recon_loss,
        'qf_loss': qf_loss,
        # zero on skipped steps; train_iteration averages over updated steps only
        'policy_loss': policy_loss,
        'bc_loss': bc_loss,
        'actor_updated': actor_updated.astype(jnp.float32),
        'q1_pred': q1_pred.mean(),
        'q_target': target_q.mean(),
        'z_mean': z.mean(),
        'z_var': z_vars.mean(),
    }
    return state.replace(
        encoder=encoder, qf1=qf1, qf2=qf2, policy=policy,
        target_qf1=tq1, target_qf2=tq2, target_policy=tp,
        step=state.step + 1,
    ), metrics


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def _rollout(env, env_params, nets, policy_params, z, key, length, stochastic=False):
    """Roll the real env for `length` steps under a fixed z.

    Returns the trajectory and its undiscounted return. `step_env` is used
    rather than `step` because a path is exactly one episode, so the auto-reset
    branch would only add noise.
    """
    reset_key, key = jax.random.split(key)
    obs0, state0 = env.reset(reset_key, env_params)

    def body(carry, k):
        obs, state = carry
        if stochastic:
            action, _ = nets.act(policy_params, obs[None], z[None], key=k)
        else:
            action, _ = nets.act(policy_params, obs[None], z[None], deterministic=True)
        action = action[0]
        next_obs, next_state, reward, _, _ = env.step_env(k, state, action, env_params)
        return (next_obs, next_state), (obs, action, reward)

    (_, _), (obs, actions, rewards) = jax.lax.scan(
        body, (obs0, state0), jax.random.split(key, length))
    return obs, actions, rewards, rewards.sum()


def _z_from_context(nets, encoder_params, context, valid):
    """Posterior mean over the valid prefix of an accumulating context."""
    embeddings = nets.embed(encoder_params, context)
    mask = valid[:, None]
    return (embeddings * mask).sum(axis=0) / mask.sum()


def eval_given_context(state, cfg, nets, env, env_params, buffer_context, key):
    """Given-context protocol (`_do_eval`).

    Context is drawn from the task's offline buffer, 20 transitions per path,
    and accumulates across the 30 paths of the evaluation.
    """
    n_paths = cfg.num_steps_per_eval // cfg.max_path_length
    total = n_paths * cfg.max_path_length

    ctx = jnp.zeros((total, cfg.context_dim))
    valid = jnp.zeros((total,))

    def path(carry, xs):
        ctx, valid = carry
        i, k = xs
        draw_key, roll_key = jax.random.split(k)
        rows = jax.random.randint(draw_key, (cfg.max_path_length,), 0,
                                  buffer_context.shape[0])
        block = buffer_context[rows]
        start = i * cfg.max_path_length
        ctx = jax.lax.dynamic_update_slice(ctx, block, (start, 0))
        valid = jax.lax.dynamic_update_slice(
            valid, jnp.ones((cfg.max_path_length,)), (start,))
        z = _z_from_context(nets, state.encoder.params['encoder'], ctx, valid)
        _, _, _, ret = _rollout(env, env_params, nets, state.policy.params, z,
                                roll_key, cfg.max_path_length)
        return (ctx, valid), ret

    (_, _), returns = jax.lax.scan(
        path, (ctx, valid), (jnp.arange(n_paths), jax.random.split(key, n_paths)))
    return returns.mean()


def eval_one_shot(state, cfg, nets, env, env_params, key):
    """One-shot protocol (`_do_eval_with_online_context`).

    A single exploration episode is collected with the zero-latent policy
    sampling stochastically, and every path then draws its context from those
    `online_sample_num` transitions.
    """
    explore_key, eval_key = jax.random.split(key)

    zero_z = jnp.zeros((cfg.latent_dim,))
    obs, actions, rewards, _ = _rollout(
        env, env_params, nets, state.policy.params, zero_z, explore_key,
        cfg.online_sample_num, stochastic=True)
    online_context = jnp.concatenate([obs, actions, rewards[:, None]], axis=-1)

    n_paths = cfg.num_steps_per_eval // cfg.max_path_length
    total = n_paths * cfg.max_path_length
    ctx = jnp.zeros((total, cfg.context_dim))
    valid = jnp.zeros((total,))

    def path(carry, xs):
        ctx, valid = carry
        i, k = xs
        draw_key, roll_key = jax.random.split(k)
        rows = jax.random.randint(draw_key, (cfg.max_path_length,), 0,
                                  online_context.shape[0])
        block = online_context[rows]
        start = i * cfg.max_path_length
        ctx = jax.lax.dynamic_update_slice(ctx, block, (start, 0))
        valid = jax.lax.dynamic_update_slice(
            valid, jnp.ones((cfg.max_path_length,)), (start,))
        z = _z_from_context(nets, state.encoder.params['encoder'], ctx, valid)
        _, _, _, ret = _rollout(env, env_params, nets, state.policy.params, z,
                                roll_key, cfg.max_path_length)
        return (ctx, valid), ret

    (_, _), returns = jax.lax.scan(
        path, (ctx, valid), (jnp.arange(n_paths), jax.random.split(eval_key, n_paths)))
    return returns.mean()


def evaluate(state, cfg, nets, env, env_params, train_context, eval_context, key):
    """Both protocols on both task splits, vmapped over tasks.

    `env_params.goal` is (num_tasks, 2) covering the train tasks followed by the
    eval tasks; the context arrays supply each split's offline buffer.
    """
    n_train = train_context.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def params_for(goal):
        return env_params.replace(goal=goal)

    goals = env_params.goal
    train_goals, eval_goals = goals[:n_train], goals[n_train:]

    given_train = jax.vmap(
        lambda g, c, k: eval_given_context(state, cfg, nets, env, params_for(g), c, k)
    )(train_goals, train_context, jax.random.split(k1, n_train))
    given_eval = jax.vmap(
        lambda g, c, k: eval_given_context(state, cfg, nets, env, params_for(g), c, k)
    )(eval_goals, eval_context, jax.random.split(k2, eval_context.shape[0]))
    shot_train = jax.vmap(
        lambda g, k: eval_one_shot(state, cfg, nets, env, params_for(g), k)
    )(train_goals, jax.random.split(k3, n_train))
    shot_eval = jax.vmap(
        lambda g, k: eval_one_shot(state, cfg, nets, env, params_for(g), k)
    )(eval_goals, jax.random.split(k4, eval_goals.shape[0]))

    return {
        'AverageReturn_all_train_tasks': given_train.mean(),
        'AverageReturn_all_test_tasks': given_eval.mean(),
        'AverageReturn_all_train_tasks_expl': shot_train.mean(),
        'AverageReturn_all_test_tasks_expl': shot_eval.mean(),
    }


def train_iteration(state, cfg, nets, data, context_data, dynamics):
    """One outer iteration: resample encoder buffers, relabel, then N steps."""
    rng, k_resample, k_relabel, k_steps = jax.random.split(state.rng, 4)
    state = state.replace(rng=rng)

    state = resample_enc_buffer(state, cfg, data.obs.shape[1], k_resample)
    state = make_relabel(state, cfg, nets, context_data, dynamics, k_relabel)

    def body(state, k):
        return train_step(state, cfg, nets, data, context_data, k)

    state, traces = jax.lax.scan(
        body, state, jax.random.split(k_steps, cfg.num_train_steps_per_itr))

    # the actor runs every policy_freq steps, so a plain mean over its loss
    # would divide by the wrong denominator
    updated = traces.pop('actor_updated')
    n_updates = jnp.maximum(updated.sum(), 1.0)
    metrics = {k: v.mean() for k, v in traces.items()}
    for k in ('policy_loss', 'bc_loss'):
        metrics[k] = (traces[k] * updated).sum() / n_updates
    return state, metrics


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(cfg, data_dir, dynamics_dir, seed=0, num_train_tasks=10, num_eval_tasks=10,
        n_trj=100, train_epoch=100000, eval_epoch=None, eval_every=1,
        dynamics_source='jax', log_path=None):
    """Stage 4 end to end: load data, train, evaluate, write progress.csv."""
    import csv
    import time

    from gentle_jax import dynamics as jdyn
    from gentle_jax.data import build_dataset
    from gentle_jax.envs import ObsNorm, make_point_robot

    train_tasks = list(range(num_train_tasks))
    eval_tasks = list(range(num_train_tasks, num_train_tasks + num_eval_tasks))
    train_data, eval_data, normalizer = build_dataset(
        data_dir, train_tasks, eval_tasks, n_trj, train_epoch, eval_epoch)
    train_data = jax.tree_util.tree_map(jnp.asarray, train_data)

    nets = Nets(cfg)
    key = jax.random.PRNGKey(seed)
    key, init_key, buf_key = jax.random.split(key, 3)
    state = create_state(cfg, nets, init_key)
    state = init_enc_buffer(state, cfg, train_data.num_transitions, buf_key)

    hidden = (cfg.net_size,) * (3 if cfg.use_next_obs_in_context else 2)
    loader = jdyn.load_from_torch if dynamics_source == 'torch' else jdyn.load
    dynamics = loader(dynamics_dir, num_train_tasks, cfg.obs_dim, cfg.action_dim,
                      hidden, 7, cfg.use_next_obs_in_context)

    train_context = jnp.asarray(train_data.context(cfg.use_next_obs_in_context))
    eval_context = jnp.asarray(eval_data.context(cfg.use_next_obs_in_context))

    obs_norm = ObsNorm.from_stats(normalizer.mean.astype(np.float32),
                                  normalizer.var.astype(np.float32))
    env, env_params = make_point_robot(num_train_tasks + num_eval_tasks,
                                       cfg.max_path_length, obs_norm=obs_norm)

    jit_iteration = jax.jit(functools.partial(
        train_iteration, cfg=cfg, nets=nets, data=train_data,
        context_data=train_context, dynamics=dynamics))
    jit_eval = jax.jit(functools.partial(
        evaluate, cfg=cfg, nets=nets, env=env, env_params=env_params,
        train_context=train_context, eval_context=eval_context))

    rows = []
    start = time.time()
    for it in range(cfg.num_iterations):
        state, metrics = jit_iteration(state)
        if it % eval_every == 0 or it == cfg.num_iterations - 1:
            key, eval_key = jax.random.split(key)
            stats = jit_eval(state, key=eval_key)
            row = {'Epoch': it, **{k: float(v) for k, v in stats.items()},
                   **{k: float(v) for k, v in metrics.items()}}
            rows.append(row)
            if it % 25 == 0 or it == cfg.num_iterations - 1:
                print(f'  itr {it:4d}  '
                      f'given/in {row["AverageReturn_all_train_tasks"]:7.2f}  '
                      f'given/ood {row["AverageReturn_all_test_tasks"]:7.2f}  '
                      f'shot/in {row["AverageReturn_all_train_tasks_expl"]:7.2f}  '
                      f'shot/ood {row["AverageReturn_all_test_tasks_expl"]:7.2f}  '
                      f'({time.time() - start:.0f}s)')

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'  wrote {log_path}')
    return state, rows
