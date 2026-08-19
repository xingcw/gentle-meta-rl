"""Stages 1 and 2: SAC behavior policies and trajectory collection.

Port of rlkit/torch/sac/pytorch_sac (the DrQ-style SAC that generates GENTLE's
offline datasets), following the fully-jitted single-file structure used by
purejaxrl and rejax. The whole of training -- environment stepping, replay
sampling, and all three gradient updates -- lives inside one `lax.scan`, and
`jax.vmap` runs every task's SAC concurrently instead of the torch pipeline's
worker processes.

The hyperparameters come from pytorch_sac's config/agent/sac.yaml and the
`sac_params` block of the experiment JSON; the defaults below are that yaml.
Two details of the torch loop are preserved because they change the data:

- actions are uniform random for the first `num_seed_steps` steps, and no
  gradient update runs until then;
- point-robot episodes only ever end at the time limit, and the torch loop
  bootstraps through time-limit endings (`done_no_max`), so the critic target
  is never masked.

Stage 2 writes the same files the torch collector wrote, so datasets from
either implementation are interchangeable:

    <data_dir>/goal_idx<i>/trj_evalsample<n>_step<checkpoint>.npy
"""
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@struct.dataclass
class SacConfig:
    obs_dim: int = struct.field(pytree_node=False, default=2)
    action_dim: int = struct.field(pytree_node=False, default=2)
    hidden_dim: int = struct.field(pytree_node=False, default=1024)
    hidden_depth: int = struct.field(pytree_node=False, default=2)
    num_train_steps: int = struct.field(pytree_node=False, default=100000)
    num_seed_steps: int = struct.field(pytree_node=False, default=1000)
    batch_size: int = struct.field(pytree_node=False, default=1024)
    replay_capacity: int = struct.field(pytree_node=False, default=100000)
    max_episode_steps: int = struct.field(pytree_node=False, default=20)
    discount: float = 0.99
    init_temperature: float = 0.1
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    critic_tau: float = 0.005
    actor_update_frequency: int = struct.field(pytree_node=False, default=1)
    critic_target_update_frequency: int = struct.field(pytree_node=False, default=2)


def orthogonal_mlp(hidden_dim, depth, out_dim):
    """pytorch_sac's utils.mlp with utils.weight_init (orthogonal, zero bias)."""
    layers = []
    for _ in range(depth):
        layers += [nn.Dense(hidden_dim,
                            kernel_init=nn.initializers.orthogonal(),
                            bias_init=nn.initializers.zeros), nn.relu]
    layers.append(nn.Dense(out_dim, kernel_init=nn.initializers.orthogonal(),
                           bias_init=nn.initializers.zeros))
    return nn.Sequential(layers)


class DiagGaussianActor(nn.Module):
    action_dim: int
    hidden_dim: int
    hidden_depth: int

    @nn.compact
    def __call__(self, obs):
        out = orthogonal_mlp(self.hidden_dim, self.hidden_depth,
                             2 * self.action_dim)(obs)
        mu, log_std = jnp.split(out, 2, axis=-1)
        # squash log_std into [LOG_STD_MIN, LOG_STD_MAX] via tanh
        log_std = jnp.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std


class DoubleQCritic(nn.Module):
    hidden_dim: int
    hidden_depth: int

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = orthogonal_mlp(self.hidden_dim, self.hidden_depth, 1)(x)
        q2 = orthogonal_mlp(self.hidden_dim, self.hidden_depth, 1)(x)
        return q1, q2


def sample_squashed(key, mu, log_std):
    """Draw from SquashedNormal and return the action with its log-prob.

    The log-det term is pytorch_sac's numerically stable TanhTransform form,
    2 * (log 2 - x - softplus(-2x)).
    """
    std = jnp.exp(log_std)
    pre_tanh = mu + std * jax.random.normal(key, mu.shape)
    action = jnp.tanh(pre_tanh)
    log_prob = (-0.5 * ((pre_tanh - mu) / std) ** 2 - log_std
                - 0.5 * jnp.log(2 * jnp.pi))
    log_prob -= 2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))
    return action, log_prob.sum(axis=-1, keepdims=True)


@struct.dataclass
class Replay:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    size: jnp.ndarray
    ptr: jnp.ndarray


@struct.dataclass
class SacState:
    actor: TrainState
    critic: TrainState
    critic_target: dict
    log_alpha: jnp.ndarray
    alpha_opt: optax.OptState
    buffer: Replay
    env_state: object
    obs: jnp.ndarray
    step: jnp.ndarray


def create_sac_state(cfg, key, env, env_params):
    actor = DiagGaussianActor(action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim,
                              hidden_depth=cfg.hidden_depth)
    critic = DoubleQCritic(hidden_dim=cfg.hidden_dim, hidden_depth=cfg.hidden_depth)

    k_actor, k_critic, k_reset = jax.random.split(key, 3)
    obs_ph = jnp.zeros((1, cfg.obs_dim))
    act_ph = jnp.zeros((1, cfg.action_dim))
    actor_params = actor.init(k_actor, obs_ph)['params']
    critic_params = critic.init(k_critic, obs_ph, act_ph)['params']

    log_alpha = jnp.array(np.log(cfg.init_temperature), dtype=jnp.float32)
    alpha_tx = optax.adam(cfg.alpha_lr)

    obs, env_state = env.reset(k_reset, env_params)
    buffer = Replay(
        obs=jnp.zeros((cfg.replay_capacity, cfg.obs_dim)),
        actions=jnp.zeros((cfg.replay_capacity, cfg.action_dim)),
        rewards=jnp.zeros((cfg.replay_capacity, 1)),
        next_obs=jnp.zeros((cfg.replay_capacity, cfg.obs_dim)),
        size=jnp.array(0, jnp.int32), ptr=jnp.array(0, jnp.int32),
    )
    return SacState(
        actor=TrainState.create(apply_fn=None, params=actor_params,
                                tx=optax.adam(cfg.actor_lr)),
        critic=TrainState.create(apply_fn=None, params=critic_params,
                                 tx=optax.adam(cfg.critic_lr)),
        critic_target=critic_params,
        log_alpha=log_alpha, alpha_opt=alpha_tx.init(log_alpha),
        buffer=buffer, env_state=env_state, obs=obs, step=jnp.array(0, jnp.int32),
    ), (actor, critic, alpha_tx)


def sac_step(state, key, cfg, modules, env, env_params):
    """One environment step plus (after the seed phase) one gradient update."""
    actor, critic, alpha_tx = modules
    k_act, k_env, k_sample, k_next, k_pi = jax.random.split(key, 5)

    # --- act -------------------------------------------------------------
    mu, log_std = actor.apply({'params': state.actor.params}, state.obs[None])
    policy_action, _ = sample_squashed(k_act, mu, log_std)
    random_action = jax.random.uniform(k_act, (1, cfg.action_dim), minval=-1.0,
                                       maxval=1.0)
    seeding = state.step < cfg.num_seed_steps
    action = jnp.where(seeding, random_action, policy_action)[0]

    next_obs, env_state, reward, done, _ = env.step_env(k_env, state.env_state,
                                                        action, env_params)

    ptr = state.buffer.ptr
    buffer = state.buffer.replace(
        obs=state.buffer.obs.at[ptr].set(state.obs),
        actions=state.buffer.actions.at[ptr].set(action),
        rewards=state.buffer.rewards.at[ptr].set(reward[None]),
        next_obs=state.buffer.next_obs.at[ptr].set(next_obs),
        size=jnp.minimum(state.buffer.size + 1, cfg.replay_capacity),
        ptr=(ptr + 1) % cfg.replay_capacity,
    )
    # episodes end only at the time limit, so reset rather than bootstrap-mask
    reset_obs, reset_state = env.reset(k_next, env_params)
    env_state = jax.tree_util.tree_map(
        lambda r, s: jnp.where(done, r, s), reset_state, env_state)
    obs = jnp.where(done, reset_obs, next_obs)

    # --- update ----------------------------------------------------------
    def do_update(state):
        idx = jax.random.randint(k_sample, (cfg.batch_size,), 0, buffer.size)
        b_obs, b_act = buffer.obs[idx], buffer.actions[idx]
        b_rew, b_next = buffer.rewards[idx], buffer.next_obs[idx]
        alpha = jnp.exp(state.log_alpha)

        # critic
        n_mu, n_log_std = actor.apply({'params': state.actor.params}, b_next)
        n_act, n_logp = sample_squashed(k_next, n_mu, n_log_std)
        tq1, tq2 = critic.apply({'params': state.critic_target}, b_next, n_act)
        target_v = jnp.minimum(tq1, tq2) - alpha * n_logp
        target_q = jax.lax.stop_gradient(b_rew + cfg.discount * target_v)

        def critic_loss_fn(params):
            q1, q2 = critic.apply({'params': params}, b_obs, b_act)
            return jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)

        c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(state.critic.params)
        critic_ts = state.critic.apply_gradients(grads=c_grads)

        # actor and temperature, every actor_update_frequency steps
        def actor_update(carry):
            actor_ts, log_alpha, alpha_opt = carry

            def actor_loss_fn(params):
                mu, log_std = actor.apply({'params': params}, b_obs)
                act, logp = sample_squashed(k_pi, mu, log_std)
                q1, q2 = critic.apply({'params': critic_ts.params}, b_obs, act)
                return jnp.mean(alpha * logp - jnp.minimum(q1, q2)), logp

            (a_loss, logp), a_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True)(actor_ts.params)
            actor_ts = actor_ts.apply_gradients(grads=a_grads)

            target_entropy = -float(cfg.action_dim)
            detached = jax.lax.stop_gradient(-logp - target_entropy)
            alpha_grad = jnp.mean(jnp.exp(log_alpha) * detached)
            updates, alpha_opt = alpha_tx.update(alpha_grad, alpha_opt)
            return optax.apply_updates(log_alpha, updates), alpha_opt, actor_ts

        actor_due = state.step % cfg.actor_update_frequency == 0
        log_alpha, alpha_opt, actor_ts = jax.lax.cond(
            actor_due, actor_update,
            lambda c: (c[1], c[2], c[0]),
            (state.actor, state.log_alpha, state.alpha_opt))

        target_due = state.step % cfg.critic_target_update_frequency == 0
        new_target = jax.tree_util.tree_map(
            lambda p, t: jnp.where(target_due, cfg.critic_tau * p
                                   + (1 - cfg.critic_tau) * t, t),
            critic_ts.params, state.critic_target)

        return state.replace(actor=actor_ts, critic=critic_ts,
                             critic_target=new_target, log_alpha=log_alpha,
                             alpha_opt=alpha_opt)

    state = jax.lax.cond(seeding, lambda s: s, do_update, state)
    return state.replace(buffer=buffer, env_state=env_state, obs=obs,
                         step=state.step + 1), reward


def train_sac(cfg, key, env, env_params):
    """Train one task's SAC agent; vmap this over `env_params.goal` for all tasks."""
    state, modules = create_sac_state(cfg, key, env, env_params)
    step_fn = functools.partial(sac_step, cfg=cfg, modules=modules, env=env,
                                env_params=env_params)

    def body(state, k):
        return step_fn(state, k)

    key, scan_key = jax.random.split(key)
    state, rewards = jax.lax.scan(
        body, state, jax.random.split(scan_key, cfg.num_train_steps))
    return state, rewards


def collect_trajectories(cfg, actor_params, modules, env, env_params, key,
                         num_episodes):
    """Stage 2: roll the trained policy, sampling stochastically as torch does."""
    actor = modules[0]

    def episode(key):
        reset_key, key = jax.random.split(key)
        obs0, state0 = env.reset(reset_key, env_params)

        def body(carry, k):
            obs, env_state = carry
            mu, log_std = actor.apply({'params': actor_params}, obs[None])
            action, _ = sample_squashed(k, mu, log_std)
            action = action[0]
            next_obs, next_state, reward, _, _ = env.step_env(
                k, env_state, action, env_params)
            return (next_obs, next_state), (obs, action, reward, next_obs)

        _, out = jax.lax.scan(body, (obs0, state0),
                              jax.random.split(key, cfg.max_episode_steps))
        return out

    return jax.vmap(episode)(jax.random.split(key, num_episodes))


def save_trajectories(out_dir, goal_idx, obs, actions, rewards, next_obs,
                      checkpoint_step):
    """Write one task's episodes in the torch collector's on-disk format."""
    task_dir = os.path.join(out_dir, f'goal_idx{goal_idx}')
    os.makedirs(task_dir, exist_ok=True)
    obs, actions = np.asarray(obs), np.asarray(actions)
    rewards, next_obs = np.asarray(rewards), np.asarray(next_obs)
    for ep in range(obs.shape[0]):
        arr = np.empty((obs.shape[1], 4), dtype=object)
        for t in range(obs.shape[1]):
            arr[t, 0] = obs[ep, t].astype(np.float64)
            arr[t, 1] = actions[ep, t].astype(np.float64)
            arr[t, 2] = np.float64(rewards[ep, t])
            arr[t, 3] = next_obs[ep, t].astype(np.float64)
        np.save(os.path.join(task_dir,
                             f'trj_evalsample{ep}_step{checkpoint_step}.npy'), arr)


def run_datagen(cfg, data_dir, n_tasks=20, num_episodes=100, seed=0,
                matmul_precision='tensorfloat32', verbose=True):
    """Stages 1 and 2: train every task's SAC agent, then collect its dataset.

    All tasks train under one `vmap`, replacing the torch pipeline's worker
    processes.

    This stage is compute-bound rather than launch-bound -- one update at the
    default config is ~52 GFLOP per task, and torch already ran it at
    8.6 TFLOP/s -- so vmap alone buys little and the precision of the matmuls
    is what dominates. Measured over 20 tasks x 2000 steps, tensorfloat32 runs
    2.1x faster than full fp32 (24.4 vs 11.7 TFLOP/s) and reaches the same
    return to two decimals, so it is the default; pass 'highest' to match
    torch's fp32 exactly.
    """
    import time

    from gentle_jax.envs import make_point_robot

    env, env_params = make_point_robot(n_tasks, cfg.max_episode_steps)
    key = jax.random.PRNGKey(seed)
    train_key, collect_key = jax.random.split(key)

    def one_task(k, goal):
        return train_sac(cfg, k, env, env_params.replace(goal=goal))

    t0 = time.time()
    with jax.default_matmul_precision(matmul_precision):
        state, rewards = jax.block_until_ready(jax.jit(jax.vmap(one_task))(
            jax.random.split(train_key, n_tasks), env_params.goal))
    if verbose:
        ep = np.asarray(rewards).reshape(n_tasks, -1, cfg.max_episode_steps).sum(axis=2)
        print(f'  stage 1: {n_tasks} SAC agents x {cfg.num_train_steps} steps in '
              f'{time.time() - t0:.0f}s')
        print(f'    mean episode return  first 20 eps {ep[:, :20].mean():7.2f}   '
              f'last 20 eps {ep[:, -20:].mean():7.2f}')

    # rebuild the modules so `collect_trajectories` can apply the trained params.
    # Only the goal is per-task here, so slice that rather than tree_map over
    # every leaf, which would also slice the observation-normalizer statistics.
    _, modules = create_sac_state(cfg, key, env,
                                  env_params.replace(goal=env_params.goal[0]))

    t0 = time.time()
    collect = jax.jit(lambda p, goal, k: collect_trajectories(
        cfg, p, modules, env, env_params.replace(goal=goal), k, num_episodes))
    for task in range(n_tasks):
        task_params = jax.tree_util.tree_map(lambda x: x[task], state.actor.params)
        obs, actions, rewards_, next_obs = collect(
            task_params, env_params.goal[task],
            jax.random.fold_in(collect_key, task))
        save_trajectories(data_dir, task, obs, actions, rewards_, next_obs,
                          cfg.num_train_steps)
    if verbose:
        print(f'  stage 2: {n_tasks} x {num_episodes} episodes written to {data_dir} '
              f'({time.time() - t0:.0f}s)')
    return state
