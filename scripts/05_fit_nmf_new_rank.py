#!/usr/bin/env python3
"""Fit an additional NMF rank to the existing shifted LAION-fMRI response matrix.

This script does NOT use food labels for model fitting. It saves arrays with the
same shape/name conventions as the earlier project, so the existing
03_test_food_selectivity.py can analyze the new rank.

Input:  responses_shifted.npy       (voxels, trials) = (12020, 1044)
Output: rank_50_trial_activations.npy   (trials, components)
        rank_50_voxel_components.npy   (components, voxels)
        rank_50_fit_summary.json
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def main():
    local = Path.home() / '85426-sude-berin-local-data' / 'nmf' / 'sub-01' / 'ses-01'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=local / 'responses_shifted.npy')
    parser.add_argument('--results', type=Path, default=local / 'nmf_results')
    parser.add_argument('--rank', type=int, default=50)
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f'Input matrix not found: {args.data}')
    X_vox_by_trial = np.load(args.data, mmap_mode='r')
    if X_vox_by_trial.ndim != 2:
        parser.error(f'Expected a 2-D matrix; got {X_vox_by_trial.shape}')
    n_voxels, n_trials = X_vox_by_trial.shape
    if not 1 <= args.rank <= min(n_voxels, n_trials):
        parser.error('Rank must be between 1 and min(number of voxels, trials).')
    print(f'Input: {args.data} shape={X_vox_by_trial.shape} dtype={X_vox_by_trial.dtype}', flush=True)

    # The earlier project stores the matrix as voxels x trials; sklearn NMF
    # takes samples x features. Each trial is a sample, and voxels are features.
    X = np.array(X_vox_by_trial.T, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(X).all() or float(X.min()) < 0:
        raise SystemExit('NMF needs finite, nonnegative input. Check preprocessing.')
    if np.all(X == 0):
        raise SystemExit('Input matrix is all zeros.')

    args.results.mkdir(parents=True, exist_ok=True)
    prefix = args.results / f'rank_{args.rank}'
    trial_path = Path(str(prefix) + '_trial_activations.npy')
    voxel_path = Path(str(prefix) + '_voxel_components.npy')
    summary_path = Path(str(prefix) + '_fit_summary.json')
    if not args.overwrite and any(p.exists() for p in (trial_path, voxel_path, summary_path)):
        parser.error('Rank output already exists; refusing to overwrite. Use --overwrite if intentional.')

    print(f'Fitting standard sklearn NMF: rank={args.rank}, solver=cd, init=nndsvda, '
          f'max_iter={args.max_iter}, tol={args.tol}, seed={args.seed}', flush=True)
    model = NMF(n_components=args.rank, init='nndsvda', solver='cd',
                max_iter=args.max_iter, tol=args.tol, random_state=args.seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        trial_activations = model.fit_transform(X)
    voxel_components = model.components_
    recon_err = float(model.reconstruction_err_)
    relative_err = recon_err / float(np.linalg.norm(X))

    if not np.isfinite(trial_activations).all() or not np.isfinite(voxel_components).all():
        raise SystemExit('NMF produced nonfinite factors; no outputs saved.')
    np.save(trial_path, trial_activations.astype(np.float32))
    np.save(voxel_path, voxel_components.astype(np.float32))
    summary = {
        'rank': args.rank,
        'input_path': str(args.data.resolve()),
        'input_shape_voxels_by_trials': [int(n_voxels), int(n_trials)],
        'trial_activations_shape': list(trial_activations.shape),
        'voxel_components_shape': list(voxel_components.shape),
        'model_type': 'standard sklearn NMF (not Bayesian NMF)',
        'init': 'nndsvda', 'solver': 'cd', 'random_state': args.seed,
        'max_iter': args.max_iter, 'tol': args.tol, 'n_iter': int(model.n_iter_),
        'reconstruction_frobenius_error': recon_err,
        'relative_reconstruction_error': relative_err,
        'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in captured),
        'note': 'Model fit was unsupervised. Food selectivity requires a separate label-aligned test.'
    }
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Wrote {trial_path} shape={trial_activations.shape}', flush=True)
    print(f'Wrote {voxel_path} shape={voxel_components.shape}', flush=True)
    print(f'Frobenius reconstruction error: {recon_err:.4f}; '
          f'relative error: {relative_err:.4f}; n_iter={model.n_iter_}', flush=True)
    if summary['convergence_warning']:
        print('WARNING: NMF hit max_iter before meeting tolerance; review fit summary.', flush=True)
    print(f'Wrote {summary_path}', flush=True)


if __name__ == '__main__':
    main()
