"""Central resolution of on-disk data locations.

The published code hardcoded the authors' machine paths (``/data/zrz/...``).
Everything now hangs off a single root, overridable with ``GENTLE_DATA_DIR``:

    <root>/gentle_data/<env_name>/goal_idx<i>/   SAC checkpoints + collected trajectories
    <root>/gentle_data/asset/dynamics/<env>/expert_seed<s>/   pretrained dynamics ensembles
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.abspath(os.environ.get('GENTLE_DATA_DIR', os.path.join(REPO_ROOT, 'data')))


def data_dir_for(env_name):
    """Directory holding the per-task offline data for ``env_name``."""
    return os.path.join(DATA_ROOT, 'gentle_data', env_name)


def goal_dir_for(env_name, goal_idx):
    return os.path.join(data_dir_for(env_name), 'goal_idx{}'.format(goal_idx))


def dynamics_dir_for(env_name, seed):
    return os.path.join(DATA_ROOT, 'gentle_data', 'asset', 'dynamics',
                        env_name, 'expert_seed{}'.format(seed))


def resolve_data_dir(variant):
    """Fill in ``algo_params.data_dir`` unless the config gives an explicit one."""
    configured = variant['algo_params'].get('data_dir')
    if not configured or configured.startswith('/data/zrz'):
        configured = data_dir_for(variant['env_name'])
    variant['algo_params']['data_dir'] = configured
    return configured
