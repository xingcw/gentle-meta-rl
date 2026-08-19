"""JAX environments for the GENTLE pipeline.

The API is deliberately functional and gymnax-shaped so that an MJX environment
can be dropped in without touching the training code:

    obs, state           = env.reset(key, params)
    obs, state, r, d, {} = env.step(key, state, action, params)

`step` auto-resets on termination, like gymnax. Task identity lives in
`params.goal`, so `jax.vmap` over params runs all tasks in parallel.

Actions arrive in [-1, 1] and are rescaled to the raw action box, matching
rlkit.envs.wrappers.NormalizedBoxEnv. Observations are optionally z-scored with
statistics supplied after the offline dataset is loaded, matching
NormalizedBoxEnv.update_obs_mean_var.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class ObsNorm:
    """z-score statistics; `enabled=False` leaves observations untouched."""
    mean: jnp.ndarray
    var: jnp.ndarray
    enabled: bool = struct.field(pytree_node=False, default=False)

    @staticmethod
    def identity(obs_dim):
        return ObsNorm(mean=jnp.zeros(obs_dim), var=jnp.ones(obs_dim), enabled=False)

    @staticmethod
    def from_stats(mean, var):
        return ObsNorm(mean=jnp.asarray(mean), var=jnp.asarray(var), enabled=True)

    def forward(self, obs):
        if not self.enabled:
            return obs
        return (obs - self.mean) / jnp.sqrt(self.var + 1e-8)


@struct.dataclass
class PointRobotParams:
    goal: jnp.ndarray
    obs_norm: ObsNorm
    max_episode_steps: int = struct.field(pytree_node=False, default=20)
    act_low: float = struct.field(pytree_node=False, default=-0.1)
    act_high: float = struct.field(pytree_node=False, default=0.1)


@struct.dataclass
class PointRobotState:
    pos: jnp.ndarray
    time: jnp.ndarray


class PointRobot:
    """2-D point robot with position control; reward is negative L2 to the goal.

    Mirrors rlkit.envs.point_robot.PointEnv wrapped in NormalizedBoxEnv.
    """

    obs_dim = 2
    action_dim = 2

    def reset(self, key, params):
        pos = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)
        state = PointRobotState(pos=pos, time=jnp.array(0, dtype=jnp.int32))
        return params.obs_norm.forward(pos), state

    def step_env(self, key, state, action, params):
        # NormalizedBoxEnv: [-1, 1] -> [act_low, act_high], then clip.
        lb, ub = params.act_low, params.act_high
        scaled = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled = jnp.clip(scaled, lb, ub)

        pos = state.pos + scaled
        reward = -jnp.linalg.norm(pos - params.goal)
        time = state.time + 1
        done = time >= params.max_episode_steps
        new_state = PointRobotState(pos=pos, time=time)
        return params.obs_norm.forward(pos), new_state, reward, done, {}

    def step(self, key, state, action, params):
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset(key_reset, params)
        state = jax.tree_util.tree_map(
            lambda re, st: jnp.where(done, re, st), state_re, state_st
        )
        obs = jnp.where(done, obs_re, obs_st)
        return obs, state, reward, done, info


def point_robot_goals(n_tasks):
    """Goal set of the torch pipeline, reproduced bit-for-bit.

    rlkit.envs.point_robot.PointEnv seeds numpy with 1337 at construction and
    draws the goals one scalar at a time, so the draw order matters.
    """
    rng = np.random.RandomState(1337)
    goals = [[rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)] for _ in range(n_tasks)]
    return np.array(goals, dtype=np.float64)


def make_point_robot(n_tasks, max_episode_steps=20, obs_norm=None):
    """Return the env and a params pytree batched over `n_tasks` goals."""
    env = PointRobot()
    goals = jnp.asarray(point_robot_goals(n_tasks), dtype=jnp.float32)
    if obs_norm is None:
        obs_norm = ObsNorm.identity(env.obs_dim)
    params = PointRobotParams(
        goal=goals,
        obs_norm=obs_norm,
        max_episode_steps=max_episode_steps,
    )
    return env, params


def task_params(params, idx):
    """Slice the batched params down to a single task."""
    return params.replace(goal=params.goal[idx])


ENVS = {'point-robot': make_point_robot}
