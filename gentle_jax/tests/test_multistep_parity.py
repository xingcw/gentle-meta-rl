"""Phase B verification: many gradient steps, with all randomness removed.

The single-step test leaves the actor's reparameterized sample using different
RNG in each implementation, which hides slow divergence. Here both the actor
sample and the target-policy smoothing noise are made deterministic and both
implementations are driven with the same fixed context and RL batch per step,
so any drift is a real difference in the update rather than noise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gentle_jax  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import torch

from gentle_jax import gentle as jgentle
from gentle_jax import networks as jnets
from gentle_jax.data import build_dataset
from gentle_jax.gentle import GentleConfig, Nets, apply_train_step
from gentle_jax.tests.test_step_parity import build_torch_algorithm, jax_state_from_torch
from rlkit.paths import data_dir_for

N_STEPS = 40


def patch_torch_deterministic():
    """Make TanhNormal.rsample return tanh(mean)."""
    from rlkit.torch import distributions as d

    def rsample(self, return_pretanh_value=False):
        z = self.normal_mean
        return (torch.tanh(z), z) if return_pretanh_value else torch.tanh(z)

    d.TanhNormal.rsample = rsample


def patch_jax_deterministic():
    def det_sample(key, mean, log_std):
        action = jnp.tanh(mean)
        return action, jnp.zeros(mean.shape[:-1] + (1,)), mean

    jgentle.tanh_normal_sample = det_sample


def param_err(jax_params, torch_sd, num_hidden, log_std=False):
    conv = (jnets.policy_params_from_torch if log_std else jnets.mlp_params_from_torch)
    ref = conv(torch_sd, num_hidden)
    errs = jax.tree_util.tree_map(
        lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max()), jax_params, ref)
    return max(jax.tree_util.tree_leaves(errs))


def main():
    print(f'multi-step parity, deterministic actor, {N_STEPS} steps')
    patch_torch_deterministic()
    patch_jax_deterministic()

    cfg = GentleConfig(policy_noise=0.0)   # drop target smoothing noise too
    algo = build_torch_algorithm()
    algo.policy_noise = 0.0
    nets = Nets(cfg)
    state = jax_state_from_torch(algo, nets)

    train, _, _ = build_dataset(data_dir_for('point-robot'), list(range(10)),
                                None, 100, 100000, load_eval=False)
    ctx_all = train.context(False)
    rng = np.random.RandomState(7)
    n, b = cfg.meta_batch, cfg.batch_size

    contexts, batches = [], []
    for _ in range(N_STEPS):
        rows = rng.randint(0, ctx_all.shape[1], size=(n, cfg.embedding_batch_size))
        contexts.append(np.stack([ctx_all[t][rows[t]] for t in range(n)]).astype(np.float32))
        srows = rng.randint(0, train.num_transitions, size=(n, b))
        batches.append(tuple(
            np.stack([arr[t][srows[t]] for t in range(n)]).astype(np.float32)
            for arr in (train.obs, train.actions, train.rewards, train.next_obs,
                        train.terminals)))

    step_fn = jax.jit(lambda s, c, bt: apply_train_step(
        s, cfg, nets, c, bt, jax.random.PRNGKey(0)))

    indices = np.arange(n)
    print(f'{"step":>5} {"policy":>10} {"encoder":>10} {"qf1":>10} {"jax qf_loss":>12} '
          f'{"torch qf_loss":>13}')
    for i in range(N_STEPS):
        algo.sample_sac = lambda idx, _b=batches[i]: [torch.as_tensor(x) for x in _b]
        algo._take_step(indices, torch.as_tensor(contexts[i]))
        # _do_training, not _take_step, advances the counter that gates the
        # actor update; driving _take_step directly means doing it here.
        algo._num_steps += 1
        t_loss = dict(algo.loss)

        state, metrics = step_fn(state, jnp.asarray(contexts[i]),
                                 tuple(jnp.asarray(x) for x in batches[i]))

        if i % 5 == 0 or i == N_STEPS - 1:
            e_pi = param_err(state.policy.params, algo.agent.policy.state_dict(), 3,
                             log_std=True)
            e_enc = param_err(state.encoder.params['encoder'],
                              algo.agent.context_encoder.state_dict(), 3)
            e_qf = param_err(state.qf1.params, algo.qf1.state_dict(), 3)
            print(f'{i:5d} {e_pi:10.2e} {e_enc:10.2e} {e_qf:10.2e} '
                  f'{float(metrics["qf_loss"]):12.6f} {float(t_loss["qf_loss"]):13.6f}')

    worst = max(param_err(state.policy.params, algo.agent.policy.state_dict(), 3, True),
                param_err(state.encoder.params['encoder'],
                          algo.agent.context_encoder.state_dict(), 3),
                param_err(state.qf1.params, algo.qf1.state_dict(), 3))
    print(f'\nworst parameter error after {N_STEPS} steps: {worst:.2e}')
    print('MULTI-STEP PARITY OK' if worst < 1e-4 else 'MULTI-STEP PARITY MISMATCH')


if __name__ == '__main__':
    main()
