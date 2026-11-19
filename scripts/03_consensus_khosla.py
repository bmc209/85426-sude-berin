#!/usr/bin/env python3
"""Consensus of repeated 20-component NMF fits; adapts authors' get_consensus.py.

Input: images x components for each run. Outlier removal uses the authors'
0.5 L2-neighborhood distance criterion; surviving columns are clustered.
One-to-one Hungarian matching is used to aggregate voxel maps across runs.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from khosla_common import DEFAULT_OUTPUT, standardize


def unit_l2(x, axis=0):
    norms=np.linalg.norm(x,axis=axis,keepdims=True)
    return x/np.where(norms>1e-12,norms,1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--method',choices=['bayesian','sklearn'],default='bayesian')
    ap.add_argument('--rank',type=int,default=20)
    ap.add_argument('--runs',type=int,default=50)
    ap.add_argument('--density-threshold',type=float,default=.5)
    ap.add_argument('--seed',type=int,default=1)
    a=ap.parse_args()
    root=a.input_dir.expanduser()
    run_dir=root/f'{a.method}_rank{a.rank}_runs'
    data=[]
    setting_path=run_dir/'settings.json'
    if not setting_path.exists():
        raise SystemExit(f'Missing run provenance: {setting_path}')
    settings=json.loads(setting_path.read_text())
    if settings.get('method')!=a.method or settings.get('rank')!=a.rank:
        raise SystemExit('Run directory settings mismatch')
    for k in range(a.runs):
        file=run_dir/f'run_{k:03d}.npz'
        if not file.exists():
            raise SystemExit(f'Missing {file}. Fit all requested runs first.')
        with np.load(file) as npz:
            A=np.asarray(npz['image_activations'],dtype=np.float64)
            B=np.asarray(npz['voxel_weights'],dtype=np.float64)
            if 'input_sha256' in npz and str(npz['input_sha256']) != settings.get('input_sha256'):
                raise SystemExit(f'Input hash mismatch in {file}')
            if 'seed' in npz and int(npz['seed']) != int(settings['seed'])+k:
                raise SystemExit(f'Unexpected random seed in {file}')
        if A.shape[1]!=a.rank or B.shape[0]!=a.rank or A.shape[0]<a.rank:
            raise SystemExit(f'Invalid shapes in {file}: {A.shape}, {B.shape}')
        if data and (A.shape != data[0][0].shape or B.shape != data[0][1].shape):
            raise SystemExit('Run shapes differ')
        data.append((A,B))
    if a.runs<2:
        raise SystemExit('Consensus requires at least 2 runs (50 for paper)')
    # Each NMF response component becomes a unit-length sample of length n_images.
    spectra=np.concatenate([unit_l2(A).T for A,_ in data],axis=0)
    n_neighbors=max(1,min(len(spectra)-1,int(round(.30*a.runs))))
    neighbor=NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean').fit(spectra)
    dists,_=neighbor.kneighbors(spectra)
    density=dists[:,1:].mean(axis=1)
    keep=density < a.density_threshold
    if keep.sum()<a.rank:
        raise SystemExit(f'Outlier filtering retained {keep.sum()} of {len(spectra)} profiles, '
                         f'less than rank {a.rank}. Check runs or set --density-threshold higher.')
    km=KMeans(n_clusters=a.rank,n_init=15,random_state=a.seed)
    labels=km.fit_predict(spectra[keep])
    med=np.array([np.median(spectra[keep][labels==c],axis=0) for c in range(a.rank)])
    if np.any(med.sum(axis=1)<1e-12):raise SystemExit('Degenerate consensus cluster')
    # Paper: median response spectra normalized to sum to 1.
    Ac=(med/med.sum(axis=1,keepdims=True)).T  # images x rank
    # Match by Pearson correlation, the criterion described in the paper, rather
    # than cosine similarity (which is influenced by positive baselines).
    ref, _ = standardize(Ac)
    ref = ref / np.sqrt(Ac.shape[0])
    voxel_maps=[]
    corr_by_component=[]
    for A,B in data:
        cand, _ = standardize(A)
        cand = cand / np.sqrt(A.shape[0])
        sim=ref.T@cand
        rr,cc=linear_sum_assignment(-sim)
        w=np.zeros_like(B)
        scores=np.zeros(a.rank)
        for target,source in zip(rr,cc):
            b=B[source]
            w[target]=b/max(float(b.sum()),1e-12)
            scores[target]=sim[target,source]
        voxel_maps.append(w)
        corr_by_component.append(scores)
    Bc=np.mean(voxel_maps,axis=0)
    out=root/f'consensus_{a.method}_rank{a.rank}'
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'consensus_image_activations.npy',Ac.astype(np.float32))
    np.save(out/'consensus_voxel_weights.npy',Bc.astype(np.float32))
    per_cluster=np.bincount(labels,minlength=a.rank)
    stability=np.mean(corr_by_component,axis=0)
    support=pd.DataFrame({'component':np.arange(1,a.rank+1),
                          'cluster_profiles':per_cluster,
                          'average_run_cosine_similarity':stability})
    support.to_csv(out/'component_stability.csv',index=False)
    info={'algorithm':'L2 normalized response profiles + nearest-neighbor density filtering + KMeans medians; Pearson-correlation matching of voxel weights',
          'method':a.method,'rank':a.rank,'runs':a.runs,
          'density_threshold':a.density_threshold,'neighbor_count':n_neighbors,
          'profiles_before_filter':int(len(spectra)),
          'profiles_after_filter':int(keep.sum()),
          'caveat':'This independently implemented consensus approximates authors method; not their missing utils_nmf.py.'}
    (out/'consensus_summary.json').write_text(json.dumps(info,indent=2))
    print('Consensus written to',out)
    print('Filtered NMF profiles:',len(spectra),'->',int(keep.sum()))
    print(support.to_string(index=False))
    print('Note: component identities are not matched to paper component numbers; do NOT assume food is component 3.')

if __name__=='__main__':main()
