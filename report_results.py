#!/usr/bin/env python3
"""Summarize GENTLE runs and compare against the numbers reported in the paper.

Reads ./logs/<env>/<algo>/seed*/<run>/progress.csv, which OfflineMetaRLAlgorithm
writes one row per training iteration.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# AAAI'24 Table 1/2, GENTLE rows. (mean, std) of average return.
PAPER = {
    'point-robot': {
        'AverageReturn_all_train_tasks':      (-6.46, 1.57),   # given-context, in-distribution
        'AverageReturn_all_test_tasks':       (-9.71, 1.31),   # given-context, OOD
        'AverageReturn_all_train_tasks_expl': (-13.50, 3.26),  # one-shot, in-distribution
        'AverageReturn_all_test_tasks_expl':  (-17.02, 2.60),  # one-shot, OOD
    },
}

LABELS = {
    'AverageReturn_all_train_tasks':      'given-context / in-distribution',
    'AverageReturn_all_test_tasks':       'given-context / OOD',
    'AverageReturn_all_train_tasks_expl': 'one-shot      / in-distribution',
    'AverageReturn_all_test_tasks_expl':  'one-shot      / OOD',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='point-robot')
    ap.add_argument('--algo', default='gentle')
    ap.add_argument('--logs', default='./logs')
    ap.add_argument('--last', type=int, default=10,
                    help='average over the final N evaluations of each run')
    args = ap.parse_args()

    pattern = os.path.join(args.logs, args.env, args.algo, 'seed*', '*', 'progress.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'no progress.csv under {pattern}')

    per_seed = {k: [] for k in LABELS}
    print(f'runs found: {len(paths)}')
    for p in paths:
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            print(f'  skipping empty {p}')
            continue
        if df.empty:
            print(f'  skipping empty {p}')
            continue
        seed = p.split(os.sep)[-3]
        n = len(df)
        row = [f'  {seed:8s} ({n:4d} iters)']
        for key in LABELS:
            if key not in df.columns:
                row.append(f'{key}=n/a')
                continue
            vals = df[key].dropna().to_numpy()
            if len(vals) == 0:
                row.append(f'{key}=n/a')
                continue
            v = float(np.mean(vals[-args.last:]))
            per_seed[key].append(v)
            row.append(f'{key.replace("AverageReturn_all_", "")}={v:8.2f}')
        print(' '.join(row))

    paper = PAPER.get(args.env, {})
    n_used = max((len(v) for v in per_seed.values()), default=0)
    print(f'\nfinal-{args.last}-eval average, aggregated over {n_used} run(s)')
    print(f'{"protocol / tasks":34s} {"reproduced":>18s} {"paper (GENTLE)":>18s}')
    for key, label in LABELS.items():
        vals = per_seed[key]
        if not vals:
            continue
        mine = f'{np.mean(vals):8.2f} ± {np.std(vals):4.2f}'
        ref = paper.get(key)
        ref_s = f'{ref[0]:8.2f} ± {ref[1]:4.2f}' if ref else ' ' * 15
        print(f'{label:34s} {mine:>18s} {ref_s:>18s}')


if __name__ == '__main__':
    main()
