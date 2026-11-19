
# Compatibility patch for Nimfa with NumPy >= 2.0
import numpy as np
if not hasattr(np, "mat"):
    np.mat = np.asmatrix

#!/usr/bin/env python3
"""Fit 20 components with 50 restarts. Default method mirrors authors' Nimfa Bayesian Bd call."""
import argparse, json, warnings, hashlib
from pathlib import Path
import numpy as np
from khosla_common import DEFAULT_OUTPUT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--method',choices=['bayesian','sklearn'],default='bayesian')
    ap.add_argument('--rank',type=int,default=20)
    ap.add_argument('--runs',type=int,default=50)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--sklearn-max-iter',type=int,default=1200)
    ap.add_argument('--force',action='store_true',help='Overwrite existing runs')
    a=ap.parse_args()
    if a.rank<2 or a.runs<1:ap.error('Rank >=2 and runs >=1 required')
    folder=a.input_dir.expanduser()
    X=np.load(folder/'X_image_by_voxel.npy',mmap_mode='r')
    n,v=X.shape
    if a.rank>min(n,v):ap.error('Rank larger than image/voxel dimension')
    if X.min()<0 or not np.isfinite(X).all():raise SystemExit('Input is not nonnegative finite')
    runs_dir=folder/f'{a.method}_rank{a.rank}_runs'
    runs_dir.mkdir(parents=True,exist_ok=True)
    # Guard against resuming checkpoints computed on a DIFFERENT input/algorithm.
    # This is especially important when replacing general-ROI with ventral-ROI data.
    digest = hashlib.sha256()
    with (folder / 'X_image_by_voxel.npy').open('rb') as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
            digest.update(block)
    settings = {'method': a.method, 'rank': a.rank, 'seed': a.seed,
                'input_sha256': digest.hexdigest(), 'input_shape': [n, v],
                'max_iter': 12 if a.method == 'bayesian' else a.sklearn_max_iter}
    setting_file = runs_dir / 'settings.json'
    if setting_file.exists() and not a.force:
        previous = json.loads(setting_file.read_text())
        if any(previous.get(k) != val for k, val in settings.items()):
            raise SystemExit('Existing NMF run folder uses different inputs/settings. '
                             'Use another --input-dir or --force to replace all fit checkpoints.')
    elif not setting_file.exists() and list(runs_dir.glob('run_*.npz')) and not a.force:
        raise SystemExit('Existing run files have no settings.json provenance. '
                         'Inspect or rerun with --force.')
    if a.force:
        for stale in runs_dir.glob('run_*.npz'):
            stale.unlink()
        for stale in runs_dir.glob('run_*.partial.npz'):
            stale.unlink()
    setting_file.write_text(json.dumps(settings, indent=2))
    if a.method=='bayesian':
        try: import nimfa
        except ImportError as e: raise SystemExit('Install Nimfa first: pip install nimfa (a legacy NumPy environment may be needed)') from e
        V=np.asarray(X.T, dtype=np.float64,order='C')  # Nimfa basis: voxels x components
    else:
        from sklearn.decomposition import NMF
        V=np.asarray(X,dtype=np.float32,order='C')
    for run in range(a.runs):
        f=runs_dir/f'run_{run:03d}.npz'
        if f.exists() and not a.force:
            print(f'Existing {f.name}; skip (use --force to refit)')
            continue
        np.random.seed(a.seed+run)
        if a.method=='bayesian':
            # Directly modeled on Khosla et al. public run_nmf.py
            model=nimfa.Bd(V,seed='random_c',rank=a.rank,max_iter=12,
                  alpha=np.zeros((v,a.rank)),beta=np.zeros((a.rank,n)),
                  theta=0.,k=0.,sigma=1.,skip=100,stride=1,
                  n_w=np.zeros((a.rank,1)),n_h=np.zeros((a.rank,1)),
                  n_run=1,n_sigma=False)
            fitted=model()
            responses=np.asarray(fitted.coef().T,dtype=np.float32)  # image x comp
            weights=np.asarray(fitted.basis().T,dtype=np.float32)    # comp x vox
            nit=12
        else:
            model=NMF(n_components=a.rank,init='random',solver='cd',max_iter=a.sklearn_max_iter,
                      tol=1e-4, random_state=a.seed+run)
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter('always')
                responses=model.fit_transform(V).astype(np.float32)
            weights=model.components_.astype(np.float32)
            nit=int(model.n_iter_)
            if warn: print(f'  Note: {warn[0].message}')
        if responses.shape!=(n,a.rank) or weights.shape!=(a.rank,v):
            raise SystemExit('Unexpected factor shapes; refusing to save')
        if min(responses.min(),weights.min()) < -1e-6 or not np.isfinite(responses).all() or not np.isfinite(weights).all():
            raise SystemExit('Invalid factors')
        rss=float(np.linalg.norm(np.asarray(X,dtype=np.float64)-responses.astype(np.float64)@weights.astype(np.float64)))
        part=f.with_name(f.stem + '.partial.npz')
        np.savez_compressed(part, image_activations=responses, voxel_weights=weights,
                            rank=a.rank, method=a.method, seed=a.seed+run,
                            rss=rss, n_iter=nit, input_sha256=settings['input_sha256'])
        part.replace(f)
        print(f'Completed {run+1}/{a.runs}: RSS={rss:.3f}, iterations={nit}, saved {f.name}',flush=True)
    print('Provenance settings stored in', setting_file)
    print(f'Finished. Run directory: {runs_dir}')

if __name__=='__main__':main()
