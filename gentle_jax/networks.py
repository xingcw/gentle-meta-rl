"""Flax ports of the GENTLE torch networks.

Layer names and shapes mirror rlkit.torch.networks / autoencoder / sac.policies
so torch checkpoints convert into flax parameter trees one-to-one, which is what
the parity tests rely on. Torch nn.Linear stores weight as (out, in) while flax
Dense stores kernel as (in, out), so conversion transposes.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

LOG_SIG_MAX = 0.0
LOG_SIG_MIN = -20.0


def fanin_init():
    """rlkit.torch.pytorch_util.fanin_init.

    The bound is 1/sqrt(weight.size(0)). torch stores an nn.Linear weight as
    (out_features, in_features), so size(0) is the *output* width despite the
    name -- reproduce that, or every first layer starts ~3x too wide here,
    where flax kernels are (in_features, out_features).
    """
    def init(key, shape, dtype=jnp.float32):
        bound = 1.0 / np.sqrt(shape[1])
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def uniform_init(w):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-w, maxval=w)

    return init


def swish(x):
    return x * jax.nn.sigmoid(x)


class Mlp(nn.Module):
    """rlkit Mlp: relu hidden stack, then `last_fc`, then an output activation."""

    hidden_sizes: tuple
    output_size: int
    init_w: float = 3e-3
    output_activation: str = 'identity'
    b_init_value: float = 0.1

    @nn.compact
    def __call__(self, x):
        for i, h in enumerate(self.hidden_sizes):
            x = nn.Dense(
                h,
                kernel_init=fanin_init(),
                bias_init=nn.initializers.constant(self.b_init_value),
                name=f'fc{i}',
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.output_size,
            kernel_init=uniform_init(self.init_w),
            bias_init=uniform_init(self.init_w),
            name='last_fc',
        )(x)
        if self.output_activation == 'tanh':
            x = jnp.tanh(x)
        return x


class TanhGaussianPolicy(nn.Module):
    """rlkit TanhGaussianPolicy: shared trunk, separate mean and log-std heads."""

    hidden_sizes: tuple
    action_dim: int
    init_w: float = 1e-3

    @nn.compact
    def __call__(self, obs):
        h = obs
        for i, size in enumerate(self.hidden_sizes):
            h = nn.Dense(
                size,
                kernel_init=fanin_init(),
                bias_init=nn.initializers.constant(0.1),
                name=f'fc{i}',
            )(h)
            h = nn.relu(h)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(self.init_w),
            bias_init=uniform_init(self.init_w),
            name='last_fc',
        )(h)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(self.init_w),
            bias_init=uniform_init(self.init_w),
            name='last_fc_log_std',
        )(h)
        log_std = jnp.clip(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std


def tanh_normal_sample(key, mean, log_std):
    """Reparameterized draw; returns the squashed action and its log-prob."""
    std = jnp.exp(log_std)
    pre_tanh = mean + std * jax.random.normal(key, mean.shape)
    action = jnp.tanh(pre_tanh)
    log_prob = _normal_log_prob(pre_tanh, mean, std) - jnp.log(1 - action ** 2 + 1e-6)
    return action, log_prob.sum(axis=-1, keepdims=True), pre_tanh


def _normal_log_prob(value, mean, std):
    var = std ** 2
    return -((value - mean) ** 2) / (2 * var) - jnp.log(std) - 0.5 * np.log(2 * np.pi)


class MlpDecoder(nn.Module):
    """rlkit.torch.autoencoder.MlpDecoder with ensemble_size == 1."""

    hidden_size: int
    num_hidden_layers: int
    obs_dim: int
    use_next_obs_in_context: bool

    @nn.compact
    def __call__(self, obs, action, z):
        out_dim = 1 + self.obs_dim if self.use_next_obs_in_context else 1
        x = jnp.concatenate([obs, action, z], axis=-1)
        out = Mlp(
            hidden_sizes=(self.hidden_size,) * self.num_hidden_layers,
            output_size=out_dim,
            name='backbones',
        )(x)
        if self.use_next_obs_in_context:
            reward, next_obs = out[..., :1], out[..., 1:]
            out = jnp.concatenate([reward, next_obs + obs], axis=-1)
        return out


class EnsembleDynamics(nn.Module):
    """rlkit EnsembleDynamicsModel: swish MLP replicated over `num_ensemble`."""

    obs_dim: int
    action_dim: int
    hidden_dims: tuple
    num_ensemble: int
    with_next_obs: bool

    @nn.compact
    def __call__(self, obs_act):
        # obs_act is (num_ensemble, batch, in_dim) or (batch, in_dim), matching
        # the torch einsum branches.
        out_dim = 1 + self.obs_dim if self.with_next_obs else 1
        dims = [self.obs_dim + self.action_dim] + list(self.hidden_dims)
        x = obs_act
        for i, (in_dim, h) in enumerate(zip(dims[:-1], dims[1:])):
            x = self._ensemble_dense(x, in_dim, h, f'backbones_{i}')
            x = swish(x)
        x = self._ensemble_dense(x, dims[-1], out_dim, 'output_layer')
        if self.with_next_obs:
            reward, next_obs = x[..., :1], x[..., 1:]
            obs = obs_act[..., :self.obs_dim]
            x = jnp.concatenate([reward, next_obs + obs], axis=-1)
        return x

    def _ensemble_dense(self, x, in_dim, out_dim, name):
        # torch calls trunc_normal_(std=1/(2*sqrt(in_dim))) with the default
        # bounds a=-2, b=2, which are absolute rather than in units of std --
        # at these std values that is +-8 sigma or wider, so nothing is
        # truncated in practice. flax's truncated_normal cuts at +-2 sigma and
        # would give a visibly narrower distribution, so use a plain normal.
        weight = self.param(
            f'{name}_weight',
            nn.initializers.normal(stddev=1 / (2 * in_dim ** 0.5)),
            (self.num_ensemble, in_dim, out_dim),
        )
        bias = self.param(
            f'{name}_bias', nn.initializers.zeros, (self.num_ensemble, 1, out_dim)
        )
        if x.ndim == 2:
            x = jnp.einsum('ij,bjk->bik', x, weight)
        else:
            x = jnp.einsum('bij,bjk->bik', x, weight)
        return x + bias


# --------------------------------------------------------------------------
# torch checkpoint -> flax parameter tree
# --------------------------------------------------------------------------

def _t(arr):
    return jnp.asarray(np.asarray(arr, dtype=np.float32))


def mlp_params_from_torch(state_dict, num_hidden, prefix=''):
    """Convert an rlkit Mlp state_dict into flax params."""
    params = {}
    for i in range(num_hidden):
        params[f'fc{i}'] = {
            'kernel': _t(state_dict[f'{prefix}fc{i}.weight']).T,
            'bias': _t(state_dict[f'{prefix}fc{i}.bias']),
        }
    params['last_fc'] = {
        'kernel': _t(state_dict[f'{prefix}last_fc.weight']).T,
        'bias': _t(state_dict[f'{prefix}last_fc.bias']),
    }
    return params


def policy_params_from_torch(state_dict, num_hidden):
    params = mlp_params_from_torch(state_dict, num_hidden)
    params['last_fc_log_std'] = {
        'kernel': _t(state_dict['last_fc_log_std.weight']).T,
        'bias': _t(state_dict['last_fc_log_std.bias']),
    }
    return params


def decoder_params_from_torch(state_dict, num_hidden):
    return {'backbones': mlp_params_from_torch(state_dict, num_hidden, prefix='backbones.')}


def dynamics_params_from_torch(state_dict, num_hidden):
    """Convert an EnsembleDynamicsModel state_dict; `saved_*` copies are dropped."""
    params = {}
    for i in range(num_hidden):
        params[f'backbones_{i}_weight'] = _t(state_dict[f'backbones.{i}.weight'])
        params[f'backbones_{i}_bias'] = _t(state_dict[f'backbones.{i}.bias'])
    params['output_layer_weight'] = _t(state_dict['output_layer.weight'])
    params['output_layer_bias'] = _t(state_dict['output_layer.bias'])
    return params
