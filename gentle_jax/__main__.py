"""Point-Robot pipeline entry point: `python -m gentle_jax`.

Mirrors run_point_robot.sh stage for stage, but all four stages run in this one
process because nothing needs to be handed between them through files.
"""
import argparse
import json
import os
import time

import gentle_jax  # noqa: F401  applies the settings in __init__ before jax
import jax

from gentle_jax import dynamics as jdyn
from gentle_jax.data import build_dataset
from gentle_jax.gentle import GentleConfig, run
from gentle_jax.sac import SacConfig, run_datagen
from rlkit.paths import data_dir_for, dynamics_dir_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--regen-data', type=int, default=0,
                    help='1 to regenerate the offline dataset with the JAX SAC')
    ap.add_argument('--data-dir', default=None,
                    help='dataset location; defaults to the shared path when '
                         'reusing data, and to <path>_jax when regenerating, so '
                         'that a regen never overwrites the torch baseline')
    ap.add_argument('--sac-precision', default='tensorfloat32',
                    help="matmul precision for stage 1; tensorfloat32 is 2.1x "
                         "faster than 'highest' at the same return")
    ap.add_argument('--retrain-dynamics', type=int, default=1)
    ap.add_argument('--n-train-tasks', type=int, default=10)
    ap.add_argument('--n-eval-tasks', type=int, default=10)
    ap.add_argument('--n-trj', type=int, default=100)
    ap.add_argument('--num-iterations', type=int, default=500)
    ap.add_argument('--logdir', default='./run_logs_jax')
    ap.add_argument('--config', default='./configs/point-robot.json')
    args = ap.parse_args()

    with open(args.config) as f:
        exp = json.load(f)
    sac_params = exp['sac_params']

    env_name = 'point-robot'
    shared = data_dir_for(env_name)
    # regenerating writes a fresh dataset; keep it away from the torch-produced
    # one, which every parity and regression check is measured against
    data_dir = args.data_dir or (f'{shared}_jax' if args.regen_data else shared)
    dyn_dir = dynamics_dir_for(env_name, args.seed)
    n_tasks = args.n_train_tasks + args.n_eval_tasks
    cfg = GentleConfig(num_iterations=args.num_iterations,
                       num_train_tasks=args.n_train_tasks,
                       meta_batch=args.n_train_tasks)
    started = time.time()

    # the SAC block of the experiment JSON drives stage 1, as it does for torch
    sac_cfg = SacConfig(num_train_steps=sac_params['num_train_steps'],
                        num_seed_steps=sac_params['num_seed_steps'],
                        replay_capacity=sac_params['num_train_steps'],
                        max_episode_steps=exp['env_params']['max_episode_steps'])

    if args.regen_data:
        print(f'[1-2/4] SAC behavior policies and trajectory collection -> {data_dir}')
        run_datagen(sac_cfg, data_dir, n_tasks=n_tasks,
                    num_episodes=args.n_trj, seed=args.seed,
                    matmul_precision=args.sac_precision)
    else:
        print(f'[1-2/4] reusing the dataset in {data_dir} (--regen-data 1 to rebuild)')

    train_tasks = list(range(args.n_train_tasks))
    eval_tasks = list(range(args.n_train_tasks, n_tasks))
    checkpoint = sac_cfg.num_train_steps

    if args.retrain_dynamics:
        print('[3/4] pretraining task dynamics ensembles')
        train_data, _, _ = build_dataset(data_dir, train_tasks, eval_tasks,
                                         args.n_trj, checkpoint, load_eval=False)
        ensembles = jdyn.train_ensembles(
            train_data, cfg.obs_dim, cfg.action_dim, (cfg.net_size,) * 2, 7,
            with_next_obs=cfg.use_next_obs_in_context,
            weight_decays=[2.5e-5, 5e-5, 7.5e-5],
            key=jax.random.PRNGKey(args.seed))
        jdyn.save(ensembles, dyn_dir)
        print(f'  saved dynamics ensembles to {dyn_dir}')
        dynamics_source = 'jax'
    else:
        print(f'[3/4] reusing the dynamics ensembles in {dyn_dir}')
        dynamics_source = 'jax' if os.path.exists(
            os.path.join(dyn_dir, 'dynamics_jax.npz')) else 'torch'

    print('[4/4] training GENTLE')
    os.makedirs(args.logdir, exist_ok=True)
    run(cfg, data_dir, dyn_dir, seed=args.seed,
        num_train_tasks=args.n_train_tasks, num_eval_tasks=args.n_eval_tasks,
        n_trj=args.n_trj, train_epoch=checkpoint, dynamics_source=dynamics_source,
        log_path=os.path.join(args.logdir, f'progress_seed{args.seed}.csv'))

    print(f'pipeline finished in {(time.time() - started) / 60:.1f} min')


if __name__ == '__main__':
    main()
