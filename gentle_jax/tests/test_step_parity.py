"""Phase B verification: one GENTLE gradient step must match torch's.

Instantiates the real rlkit GENTLE algorithm, converts its networks into the
flax state, feeds both implementations the same context and RL batch, and
compares every loss and every post-update parameter. This is the check that
tells a faithful port from a plausible-looking one: the regression run can look
reasonable while a detail of the update is wrong.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gentle_jax  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import torch

from gentle_jax import networks as jnets
from gentle_jax.data import build_dataset
from gentle_jax.gentle import GentleConfig, Nets, apply_train_step, create_state
from rlkit.paths import data_dir_for, dynamics_dir_for

CFG = GentleConfig()
SEED = 0


def build_torch_algorithm():
    """Construct GENTLE exactly as train_gentle.experiment does, on CPU."""
    from configs.default import default_config
    from rlkit.envs import ENVS
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.torch import pytorch_util as ptu
    from rlkit.torch.algo.gentle import GENTLE
    from rlkit.torch.autoencoder import MlpDecoder, MlpEncoder
    from rlkit.torch.multi_task_dynamics import MultiTaskDynamics
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.agent import Agent
    from rlkit.torch.sac.policies import ContextPolicyWrapper, TanhGaussianPolicy

    import json
    variant = dict(default_config)
    with open('configs/point-robot.json') as f:
        exp = json.load(f)

    def deep_update(fr, to):
        for k, v in fr.items():
            if isinstance(v, dict):
                deep_update(v, to[k])
            else:
                to[k] = v
        return to

    variant = deep_update(exp, variant)
    variant['algo_params']['data_dir'] = data_dir_for('point-robot')
    ptu.set_gpu_mode(False)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim, action_dim = 2, 2
    latent, net = variant['latent_size'], variant['net_size']
    ctx_dim = obs_dim + action_dim + 1
    variant['algo_params']['context_dim'] = ctx_dim

    encoder = MlpEncoder(hidden_sizes=[net] * 3, input_size=ctx_dim,
                         output_size=latent, output_activation=torch.tanh,
                         batch_attention=False)
    qf1 = FlattenMlp(hidden_sizes=[net] * 3,
                     input_size=obs_dim + action_dim + latent, output_size=1)
    qf2 = FlattenMlp(hidden_sizes=[net] * 3,
                     input_size=obs_dim + action_dim + latent, output_size=1)
    decoder = MlpDecoder(hidden_size=net, num_hidden_layers=3, z_dim=latent,
                         action_dim=action_dim, obs_dim=obs_dim, reward_dim=1,
                         use_next_obs_in_context=False)
    dynamics = MultiTaskDynamics(num_tasks=variant['n_train_tasks'], hidden_size=net,
                                 num_hidden_layers=2, action_dim=action_dim,
                                 obs_dim=obs_dim, reward_dim=1,
                                 use_next_obs_in_context=False,
                                 ensemble_size=variant['algo_params']['ensemble_size'],
                                 dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5])
    dynamics.load(dynamics_dir_for('point-robot', 0))
    policy = TanhGaussianPolicy(hidden_sizes=[net] * 3, obs_dim=obs_dim + latent,
                                latent_dim=latent, action_dim=action_dim)
    agent = Agent(latent, encoder, policy, **variant['algo_params'])

    algo = GENTLE(env=env, train_tasks=list(tasks[:variant['n_train_tasks']]),
                  eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
                  nets=[[agent, ContextPolicyWrapper(policy, latent)], qf1, qf2,
                        decoder, dynamics],
                  latent_dim=latent,
                  obs_normalizer=ptu.RunningMeanStd(shape=obs_dim),
                  **variant['algo_params'])
    return algo


def jax_state_from_torch(algo, nets):
    """Copy every torch network in the algorithm into a fresh GentleState."""
    state = create_state(CFG, nets, jax.random.PRNGKey(0))
    enc = jnets.mlp_params_from_torch(algo.agent.context_encoder.state_dict(), 3)
    dec = jnets.decoder_params_from_torch(algo.context_decoder.state_dict(), 3)
    return state.replace(
        encoder=state.encoder.replace(params={'encoder': enc, 'decoder': dec}),
        qf1=state.qf1.replace(params=jnets.mlp_params_from_torch(algo.qf1.state_dict(), 3)),
        qf2=state.qf2.replace(params=jnets.mlp_params_from_torch(algo.qf2.state_dict(), 3)),
        policy=state.policy.replace(
            params=jnets.policy_params_from_torch(algo.agent.policy.state_dict(), 3)),
        target_qf1=jnets.mlp_params_from_torch(algo.target_qf1.state_dict(), 3),
        target_qf2=jnets.mlp_params_from_torch(algo.target_qf2.state_dict(), 3),
        target_policy=jnets.policy_params_from_torch(
            algo.agent.target_policy.state_dict(), 3),
    )


def _max_err(jax_params, torch_sd, num_hidden, log_std=False):
    conv = (jnets.policy_params_from_torch if log_std else jnets.mlp_params_from_torch)
    ref = conv(torch_sd, num_hidden)
    errs = jax.tree_util.tree_map(
        lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max()), jax_params, ref)
    return max(jax.tree_util.tree_leaves(errs))


def main():
    print('single-step parity (torch GENTLE._take_step <-> jax apply_train_step)')
    algo = build_torch_algorithm()
    nets = Nets(CFG)
    state = jax_state_from_torch(algo, nets)

    rng = np.random.RandomState(1)
    n, b = CFG.meta_batch, CFG.batch_size
    indices = np.arange(n)

    # a fixed context batch and RL batch, shared by both implementations
    train, _, _ = build_dataset(data_dir_for('point-robot'), list(range(10)),
                                None, 100, 100000, load_eval=False)
    ctx_all = train.context(False)
    rows = rng.randint(0, ctx_all.shape[1], size=(n, CFG.embedding_batch_size))
    context_np = np.stack([ctx_all[t][rows[t]] for t in range(n)]).astype(np.float32)

    srows = rng.randint(0, train.num_transitions, size=(n, b))
    batch_np = tuple(
        np.stack([arr[t][srows[t]] for t in range(n)]).astype(np.float32)
        for arr in (train.obs, train.actions, train.rewards, train.next_obs,
                    train.terminals))

    # torch: feed the same RL batch through the real _take_step
    algo.sample_sac = lambda idx: [torch.as_tensor(x) for x in batch_np]
    torch.manual_seed(123)
    algo._take_step(indices, torch.as_tensor(context_np))
    t_loss = dict(algo.loss)

    # jax: same context, same batch
    _, metrics = jax.jit(lambda s: apply_train_step(
        s, CFG, nets, jnp.asarray(context_np),
        tuple(jnp.asarray(x) for x in batch_np), jax.random.PRNGKey(123)))(state)
    new_state, _ = apply_train_step(
        state, CFG, nets, jnp.asarray(context_np),
        tuple(jnp.asarray(x) for x in batch_np), jax.random.PRNGKey(123))

    print('  losses (torch vs jax)')
    pairs = [('recon_loss', 'recon_loss'), ('qf_loss', 'qf_loss'),
             ('q_target', 'q_target'), ('q1_pred', 'q1_pred')]
    worst = 0.0
    for tk, jk in pairs:
        tv, jv = float(t_loss[tk]), float(metrics[jk])
        rel = abs(tv - jv) / max(abs(tv), 1e-8)
        worst = max(worst, rel)
        print(f'    {tk:12s} {tv:12.6f}  {jv:12.6f}   rel {rel:.2e}')

    print('  post-update parameters (max abs err)')
    errs = {
        'context_encoder': _max_err(new_state.encoder.params['encoder'],
                                    algo.agent.context_encoder.state_dict(), 3),
        'context_decoder': _max_err(new_state.encoder.params['decoder']['backbones'],
                                    {k[len('backbones.'):]: v for k, v in
                                     algo.context_decoder.state_dict().items()}, 3),
        'qf1': _max_err(new_state.qf1.params, algo.qf1.state_dict(), 3),
        'qf2': _max_err(new_state.qf2.params, algo.qf2.state_dict(), 3),
        'target_qf1': _max_err(new_state.target_qf1, algo.target_qf1.state_dict(), 3),
        'policy': _max_err(new_state.policy.params,
                           algo.agent.policy.state_dict(), 3, log_std=True),
        'target_policy': _max_err(new_state.target_policy,
                                  algo.agent.target_policy.state_dict(), 3,
                                  log_std=True),
    }
    for k, v in errs.items():
        print(f'    {k:16s} {v:.3e}')

    print(f'\n  worst loss rel err {worst:.2e}; worst param err {max(errs.values()):.2e}')
    return worst, errs


if __name__ == '__main__':
    main()
