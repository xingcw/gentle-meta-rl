"""Phase A verification: the JAX point robot must match the torch one exactly.

Drives both envs with identical reset states and action sequences and compares
observations and rewards step by step.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gentle_jax  # noqa: F401  sets matmul precision / XLA flags before jax loads

import jax
import jax.numpy as jnp
import numpy as np

from gentle_jax.envs import ObsNorm, make_point_robot, point_robot_goals, task_params
from rlkit.envs import ENVS as TORCH_ENVS  # registers every torch env exactly once
from rlkit.envs.wrappers import NormalizedBoxEnv

N_TASKS = 20
MAX_STEPS = 20
TOL = 1e-5


def _torch_env(obs_stats=None):
    # randomize_tasks=True comes from configs/default.py, which point-robot.json
    # does not override; without it the env falls back to 8 hand-coded goals.
    env = NormalizedBoxEnv(TORCH_ENVS['point-robot'](
        n_tasks=N_TASKS, randomize_tasks=True, max_episode_steps=MAX_STEPS))
    if obs_stats is not None:
        env.update_obs_mean_var(*obs_stats)
    return env


def test_goals_match():
    torch_goals = np.array(_torch_env().goals)
    jax_goals = point_robot_goals(N_TASKS)
    assert np.abs(torch_goals - jax_goals).max() == 0.0, 'goal sets differ'
    return torch_goals


def test_rollout_parity(obs_stats=None, label='raw obs'):
    torch_env = _torch_env(obs_stats)
    obs_norm = (ObsNorm.identity(2) if obs_stats is None
                else ObsNorm.from_stats(*obs_stats))
    jax_env, jax_params = make_point_robot(N_TASKS, MAX_STEPS, obs_norm=obs_norm)
    step = jax.jit(jax_env.step_env)

    rng = np.random.RandomState(0)
    max_obs_err = 0.0
    max_rew_err = 0.0

    for task in range(N_TASKS):
        torch_env.reset_task(task)
        p = task_params(jax_params, task)

        # Force both envs onto the same start state; reset() draws are
        # compared separately by test_reset_distribution.
        start = rng.uniform(-1.0, 1.0, size=(2,))
        torch_env._wrapped_env._state = start.copy()
        torch_env._wrapped_env._step = 0
        state = jax.tree_util.tree_map(lambda x: x, jax_env.reset(jax.random.PRNGKey(0), p)[1])
        state = state.replace(pos=jnp.asarray(start, dtype=jnp.float32),
                              time=jnp.array(0, dtype=jnp.int32))

        for t in range(MAX_STEPS):
            action = rng.uniform(-1.5, 1.5, size=(2,))  # exercises the clip
            t_obs, t_rew, t_done, _ = torch_env.step(action)
            j_obs, state, j_rew, j_done, _ = step(
                jax.random.PRNGKey(t), state, jnp.asarray(action, dtype=jnp.float32), p
            )
            max_obs_err = max(max_obs_err, float(np.abs(np.asarray(j_obs) - t_obs).max()))
            max_rew_err = max(max_rew_err, abs(float(j_rew) - float(t_rew)))
            assert bool(j_done) == bool(t_done), f'done mismatch task {task} step {t}'

    print(f'  {label:22s} max |obs| err {max_obs_err:.2e}  max |reward| err {max_rew_err:.2e}')
    assert max_obs_err < TOL, f'obs mismatch {max_obs_err}'
    assert max_rew_err < TOL, f'reward mismatch {max_rew_err}'


def test_reset_distribution():
    """reset() draws uniform(-1, 1) per component, like the torch env."""
    env, params = make_point_robot(N_TASKS, MAX_STEPS)
    p = task_params(params, 0)
    keys = jax.random.split(jax.random.PRNGKey(0), 20000)
    pos = jax.vmap(lambda k: env.reset(k, p)[1].pos)(keys)
    pos = np.asarray(pos)
    assert pos.min() > -1.0 and pos.max() < 1.0
    assert abs(pos.mean()) < 0.02, f'reset mean {pos.mean()}'
    assert abs(pos.std() - 1.0 / np.sqrt(3)) < 0.02, f'reset std {pos.std()}'
    print(f'  reset draw             mean {pos.mean():+.4f}  std {pos.std():.4f} '
          f'(uniform(-1,1) std {1/np.sqrt(3):.4f})')


if __name__ == '__main__':
    print('env parity (torch <-> jax)')
    test_goals_match()
    print('  goal sets              identical')
    test_rollout_parity()
    stats = (np.array([0.13, -0.27]), np.array([0.41, 0.55]))
    test_rollout_parity(stats, label='z-scored obs')
    test_reset_distribution()
    print('PASS')
