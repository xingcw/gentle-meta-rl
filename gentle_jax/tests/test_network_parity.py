"""Phase B verification: flax networks must reproduce the torch forward passes.

Builds each torch network, converts its weights into the flax tree, and compares
outputs on identical inputs. Uses the point-robot shapes from
configs/point-robot.json (net_size 64, latent 5, obs 2, action 2).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gentle_jax  # noqa: F401  sets matmul precision / XLA flags before jax loads

import jax
import jax.numpy as jnp
import numpy as np
import torch

from gentle_jax import networks as jnets
from rlkit.torch.autoencoder import MlpDecoder, MlpEncoder
from rlkit.torch.networks import EnsembleDynamicsModel, FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy

OBS_DIM, ACT_DIM, LATENT, NET = 2, 2, 5, 64
CONTEXT_DIM = OBS_DIM + ACT_DIM + 1
TOL = 1e-5
# TPU approximates log/exp/tanh to ~1e-5 relative instead of correctly rounding
# them, so the log-prob tolerance has to follow the backend.
LOGPROB_TOL = 1e-3 if jax.default_backend() == 'tpu' else 1e-4


def _err(a, b):
    return float(np.abs(np.asarray(a) - b.detach().cpu().numpy()).max())


def test_encoder():
    torch_net = MlpEncoder(hidden_sizes=[NET] * 3, input_size=CONTEXT_DIM,
                           output_size=LATENT, output_activation=torch.tanh,
                           batch_attention=False)
    x = np.random.randn(10, 32, CONTEXT_DIM).astype(np.float32)
    expected = torch_net(torch.as_tensor(x))

    net = jnets.Mlp(hidden_sizes=(NET,) * 3, output_size=LATENT, output_activation='tanh')
    params = {'params': jnets.mlp_params_from_torch(torch_net.state_dict(), 3)}
    got = net.apply(params, jnp.asarray(x))
    e = _err(got, expected)
    print(f'  context encoder        max err {e:.2e}')
    assert e < TOL


def test_qf():
    torch_net = FlattenMlp(hidden_sizes=[NET] * 3,
                           input_size=OBS_DIM + ACT_DIM + LATENT, output_size=1)
    obs = np.random.randn(64, OBS_DIM).astype(np.float32)
    act = np.random.randn(64, ACT_DIM).astype(np.float32)
    z = np.random.randn(64, LATENT).astype(np.float32)
    expected = torch_net(8, 8, torch.as_tensor(obs), torch.as_tensor(act), torch.as_tensor(z))

    net = jnets.Mlp(hidden_sizes=(NET,) * 3, output_size=1)
    params = {'params': jnets.mlp_params_from_torch(torch_net.state_dict(), 3)}
    got = net.apply(params, jnp.asarray(np.concatenate([obs, act, z], axis=-1)))
    e = _err(got, expected)
    print(f'  q network              max err {e:.2e}')
    assert e < TOL


def test_decoder():
    torch_net = MlpDecoder(hidden_size=NET, num_hidden_layers=3, z_dim=LATENT,
                           action_dim=ACT_DIM, obs_dim=OBS_DIM, reward_dim=1,
                           use_next_obs_in_context=False)
    obs = np.random.randn(10, 32, OBS_DIM).astype(np.float32)
    act = np.random.randn(10, 32, ACT_DIM).astype(np.float32)
    z = np.random.randn(10, 32, LATENT).astype(np.float32)
    expected = torch_net(torch.as_tensor(obs), torch.as_tensor(act), torch.as_tensor(z))

    net = jnets.MlpDecoder(hidden_size=NET, num_hidden_layers=3, obs_dim=OBS_DIM,
                          use_next_obs_in_context=False)
    params = {'params': jnets.decoder_params_from_torch(torch_net.state_dict(), 3)}
    got = net.apply(params, jnp.asarray(obs), jnp.asarray(act), jnp.asarray(z))
    e = _err(got, expected)
    print(f'  context decoder        max err {e:.2e}')
    assert e < TOL


def test_policy():
    torch_net = TanhGaussianPolicy(hidden_sizes=[NET] * 3, obs_dim=OBS_DIM + LATENT,
                                   latent_dim=LATENT, action_dim=ACT_DIM)
    x = np.random.randn(64, OBS_DIM + LATENT).astype(np.float32)
    out = torch_net(8, 8, torch.as_tensor(x), deterministic=True)
    expected_action, expected_mean, expected_log_std = out[0], out[1], out[2]

    net = jnets.TanhGaussianPolicy(hidden_sizes=(NET,) * 3, action_dim=ACT_DIM)
    params = {'params': jnets.policy_params_from_torch(torch_net.state_dict(), 3)}
    mean, log_std = net.apply(params, jnp.asarray(x))
    e_mean = _err(mean, expected_mean)
    e_std = _err(log_std, expected_log_std)
    e_act = _err(jnp.tanh(mean), expected_action)
    print(f'  policy mean            max err {e_mean:.2e}')
    print(f'  policy log_std         max err {e_std:.2e}')
    print(f'  policy tanh(mean)      max err {e_act:.2e}')
    assert max(e_mean, e_std, e_act) < TOL


def test_tanh_normal_log_prob():
    """Our log-prob must match rlkit's TanhNormal on the same pre-tanh sample."""
    from rlkit.torch.distributions import TanhNormal
    mean = np.random.randn(64, ACT_DIM).astype(np.float32)
    log_std = np.random.uniform(-2, 0, size=(64, ACT_DIM)).astype(np.float32)
    std = np.exp(log_std)
    pre_tanh = mean + std * np.random.randn(64, ACT_DIM).astype(np.float32)
    action = np.tanh(pre_tanh)

    expected = TanhNormal(torch.as_tensor(mean), torch.as_tensor(std)).log_prob(
        torch.as_tensor(action), pre_tanh_value=torch.as_tensor(pre_tanh)
    ).sum(dim=1, keepdim=True)

    got = (jnets._normal_log_prob(jnp.asarray(pre_tanh), jnp.asarray(mean), jnp.asarray(std))
           - jnp.log(1 - jnp.asarray(action) ** 2 + 1e-6)).sum(axis=-1, keepdims=True)
    e = _err(got, expected)
    print(f'  tanh-normal log_prob   max err {e:.2e}')
    assert e < LOGPROB_TOL


def test_dynamics():
    torch_net = EnsembleDynamicsModel(obs_dim=OBS_DIM, action_dim=ACT_DIM,
                                      hidden_dims=[NET] * 2, num_ensemble=7,
                                      num_elites=7,
                                      weight_decays=[2.5e-5, 5e-5, 7.5e-5],
                                      with_next_obs=False)
    x = np.random.randn(128, OBS_DIM + ACT_DIM).astype(np.float32)
    expected = torch_net(x)

    net = jnets.EnsembleDynamics(obs_dim=OBS_DIM, action_dim=ACT_DIM,
                                hidden_dims=(NET,) * 2, num_ensemble=7,
                                with_next_obs=False)
    params = {'params': jnets.dynamics_params_from_torch(torch_net.state_dict(), 2)}
    got = net.apply(params, jnp.asarray(x))
    e = _err(got, expected)
    print(f'  dynamics ensemble      max err {e:.2e}  shape {tuple(got.shape)}')
    assert e < TOL




def test_init_distributions():
    """Freshly initialized weights must match torch's init scheme, not just its
    loaded values. rlkit's fanin_init bound is 1/sqrt(weight.size(0)), i.e. the
    output width, which is easy to get backwards when porting to flax."""
    import jax
    checks = [
        ('encoder', MlpEncoder(hidden_sizes=[NET] * 3, input_size=CONTEXT_DIM,
                               output_size=LATENT, output_activation=torch.tanh,
                               batch_attention=False),
         jnets.Mlp(hidden_sizes=(NET,) * 3, output_size=LATENT,
                   output_activation='tanh'),
         jnp.zeros((1, 1, CONTEXT_DIM))),
        ('q network', FlattenMlp(hidden_sizes=[NET] * 3,
                                 input_size=OBS_DIM + ACT_DIM + LATENT, output_size=1),
         jnets.Mlp(hidden_sizes=(NET,) * 3, output_size=1),
         jnp.zeros((1, OBS_DIM + ACT_DIM + LATENT))),
    ]
    worst = 0.0
    for name, torch_net, jax_net, dummy in checks:
        params = jax_net.init(jax.random.PRNGKey(0), dummy)['params']
        sd = torch_net.state_dict()
        for i in range(3):
            t_w = sd[f'fc{i}.weight'].detach().numpy()
            j_w = np.asarray(params[f'fc{i}']['kernel'])
            t_bound, j_bound = np.abs(t_w).max(), np.abs(j_w).max()
            rel = abs(t_bound - j_bound) / t_bound
            worst = max(worst, rel)
            print(f'  {name:10s} fc{i} weight range  torch {t_bound:.4f}  '
                  f'jax {j_bound:.4f}  rel {rel:.2f}')
        t_b = sd['fc0.bias'].detach().numpy()
        j_b = np.asarray(params['fc0']['bias'])
        assert np.allclose(t_b, 0.1) and np.allclose(j_b, 0.1), 'hidden bias init'
    assert worst < 0.15, f'init scale mismatch (worst relative diff {worst:.2f})'


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    print('network parity (torch -> flax)')
    test_encoder()
    test_qf()
    test_decoder()
    test_policy()
    test_tanh_normal_log_prob()
    test_dynamics()
    test_init_distributions()
    print('PASS')
