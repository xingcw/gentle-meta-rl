"""Phase B verification: the evaluation protocols, on torch's own weights.

Loads the policy and context encoder saved by the torch reproduction run at its
final iteration and evaluates them with the JAX implementations of both
protocols. Matching the numbers torch logged at that iteration isolates the
evaluation from training: if these agree, any gap in a full run comes from
training rather than from how returns are measured.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gentle_jax  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

from gentle_jax import networks as jnets
from gentle_jax.data import build_dataset
from gentle_jax.envs import ObsNorm, make_point_robot
from gentle_jax.gentle import GentleConfig, Nets, create_state, evaluate
from rlkit.paths import data_dir_for

CFG = GentleConfig()
ITR = 499
KEYS = ['AverageReturn_all_train_tasks', 'AverageReturn_all_test_tasks',
        'AverageReturn_all_train_tasks_expl', 'AverageReturn_all_test_tasks_expl']
LABELS = {KEYS[0]: 'given-context / in-distribution',
          KEYS[1]: 'given-context / OOD',
          KEYS[2]: 'one-shot      / in-distribution',
          KEYS[3]: 'one-shot      / OOD'}


def find_run_dir():
    dirs = sorted(glob.glob('logs/point-robot/gentle/seed*/*/progress.csv'))
    if not dirs:
        raise SystemExit('no torch run found under logs/')
    return os.path.dirname(dirs[0])


def main(n_repeats=8):
    run_dir = find_run_dir()
    print(f'torch run: {run_dir}')

    df = pd.read_csv(os.path.join(run_dir, 'progress.csv'))
    torch_row = df[df['Epoch'] == ITR].iloc[0]
    torch_last10 = df.tail(10)

    nets = Nets(CFG)
    state = create_state(CFG, nets, jax.random.PRNGKey(0))
    load = lambda name: torch.load(os.path.join(run_dir, f'{name}_itr_{ITR}.pth'),
                                   map_location='cpu', weights_only=True)
    state = state.replace(
        encoder=state.encoder.replace(params={
            'encoder': jnets.mlp_params_from_torch(load('context_encoder'), 3),
            'decoder': state.encoder.params['decoder']}),
        policy=state.policy.replace(
            params=jnets.policy_params_from_torch(load('policy'), 3)),
    )

    train, eval_, normalizer = build_dataset(data_dir_for('point-robot'),
                                             list(range(10)), list(range(10, 20)),
                                             100, 100000)
    train_ctx = jnp.asarray(train.context(False))
    eval_ctx = jnp.asarray(eval_.context(False))
    obs_norm = ObsNorm.from_stats(normalizer.mean.astype(np.float32),
                                  normalizer.var.astype(np.float32))
    env, env_params = make_point_robot(20, CFG.max_path_length, obs_norm=obs_norm)

    jit_eval = jax.jit(lambda k: evaluate(state, CFG, nets, env, env_params,
                                          train_ctx, eval_ctx, k))
    runs = [jit_eval(jax.random.PRNGKey(s)) for s in range(n_repeats)]
    got = {k: np.array([float(r[k]) for r in runs]) for k in KEYS}

    print(f'\njax eval of torch itr-{ITR} weights, {n_repeats} eval seeds\n')
    print(f'{"protocol":34s} {"jax (mean+-sd)":>20s} {"torch itr499":>13s} '
          f'{"torch last10":>13s}')
    ok = True
    for k in KEYS:
        j_m, j_s = got[k].mean(), got[k].std()
        t_499 = float(torch_row[k])
        t_10 = float(torch_last10[k].mean())
        flag = '' if abs(j_m - t_10) <= max(3.0, 2 * j_s) else '   <-- differs'
        if flag:
            ok = False
        print(f'{LABELS[k]:34s} {j_m:9.2f} +- {j_s:4.2f} {t_499:13.2f} '
              f'{t_10:13.2f}{flag}')
    print('\nEVAL PARITY OK' if ok else '\nEVAL PARITY MISMATCH')
    return got


if __name__ == '__main__':
    main()
