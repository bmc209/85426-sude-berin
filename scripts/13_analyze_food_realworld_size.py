#!/usr/bin/env python3
"""Food-only real-world size (1-10) vs Khosla-style consensus responses.

Primary: rank-based Spearman correlations across food images; image-level
permutation p-values and max-|r| FWER corrections across 20 components.
Sensitivity: stratified permutations after stimulus-family residualization.
Optional: rank-based voxel correlations across the 3,767 ventral ROI voxels,
with a separate voxelwise max-|r| FWER correction. Zero labels are NOT food.

This is observational/exploratory. CLIP-derived size categories are provisional.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def norm_key(s):
    if pd.isna(s): return ''
    p = str(s).strip().strip('"\'').replace('\\', '/').split('/')[-1]
    return Path(p).stem.casefold()


def family(s):
    u = s.upper()
    for name in ('LAION', 'NSD', 'OOD', 'THINGS+'):
        if name in u: return name
    return 'OTHER'


def zscore_columns(a):
    a = np.asarray(a, dtype=np.float64)
    d = a - a.mean(axis=0, keepdims=True)
    s = np.sqrt(np.mean(d*d, axis=0, keepdims=True))
    return np.divide(d,s,out=np.zeros_like(d),where=s>1e-12)


def residualize_families(x, groups):
    x = np.asarray(x, dtype=np.float64).copy()
    for g in np.unique(groups):
        ii = np.flatnonzero(groups==g)
        x[ii] -= x[ii].mean(axis=0, keepdims=True)
    return x


def run_permutations(X, y, groups, n_perm, seed, stratified=False, batch_size=128):
    """Return Spearman-like r, uncorrected p, and max-stat FWER p.

    Inputs must already be rank-transformed for Spearman interpretation.
    For within-family sensitivity, within-family centered rank values are used.
    """
    if stratified:
        X = residualize_families(X, groups)
        y = residualize_families(np.asarray(y)[:,None], groups)[:,0]
    Z = zscore_columns(X)
    zy = zscore_columns(np.asarray(y)[:,None])[:,0]
    if np.std(zy) < 1e-12:
        raise ValueError('Size values have no variation after family correction')
    obs = (Z.T @ zy) / len(zy)
    hits = np.ones(len(obs), dtype=np.int64)
    corrected = np.ones(len(obs), dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups_idx=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    for start in range(0,n_perm,batch_size):
        b=min(batch_size,n_perm-start)
        ys=np.empty((len(zy), b), dtype=np.float64)
        for j in range(b):
            if stratified:
                yp=zy.copy()
                for ix in groups_idx:
                    yp[ix] = zy[rng.permutation(ix)]
            else:
                yp=zy[rng.permutation(len(zy))]
            ys[:,j]=yp
        rp=(Z.T @ ys) / len(zy)
        hits += np.sum(np.abs(rp)>=np.abs(obs)[:,None]-1e-12,axis=1)
        maxabs=np.max(np.abs(rp), axis=0)
        corrected += np.sum(maxabs[None,:]>=np.abs(obs)[:,None]-1e-12,axis=1)
    return obs, hits/(n_perm+1), corrected/(n_perm+1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',required=True,type=Path,help='analysis/khosla_replication/sub-01/ses-01')
    ap.add_argument('--labels',required=True,type=Path,help='CSV from 12_clip_realworld_food_size.py')
    ap.add_argument('--method',choices=['bayesian','sklearn'],default='bayesian')
    ap.add_argument('--rank',type=int,default=20)
    ap.add_argument('--permutations',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--reviewed-only',action='store_true',help='Exclude provisional CLIP sizes')
    ap.add_argument('--min-food',type=int,default=12)
    ap.add_argument('--no-voxels',action='store_true',help='Only test NMF component associations')
    ap.add_argument('--output-dir',type=Path,default=None)
    a=ap.parse_args()
    if a.permutations<100: ap.error('Use >=100 permutations; 5000+ recommended')
    root=a.input_dir.expanduser()
    cd=root/f'consensus_{a.method}_rank{a.rank}'
    order=pd.read_csv(root/'image_order.csv')
    if 'image_key' not in order: raise SystemExit('image_order.csv missing image_key')
    order['image_key']=order.image_key.map(norm_key)
    if not order.image_key.is_unique: raise SystemExit('Duplicate image keys in image_order')
    A=np.load(cd/'consensus_image_activations.npy')
    if A.shape!=(len(order),a.rank):
        raise SystemExit(f'CONSENSUS ORDER MISMATCH: {A.shape} vs {(len(order),a.rank)}')
    labels=pd.read_csv(a.labels)
    for col in ['image_key','food_present','size_1to10']:
        if col not in labels:raise SystemExit(f'Labels lack {col}. Run script 12 first.')
    labels['image_key']=labels.image_key.map(norm_key)
    if labels.image_key.duplicated().any():raise SystemExit('Duplicate image keys in labels; fix conflicts first')
    labels['food_present']=pd.to_numeric(labels.food_present,errors='coerce')
    labels['size_1to10']=pd.to_numeric(labels.size_1to10,errors='coerce')
    if 'size_source' not in labels:
        labels['size_source']='unspecified'
    if (labels.size_1to10.notna() & (~labels.size_1to10.between(1,10))).any():
        raise SystemExit('Invalid size values; valid range is 1-10 inclusive')
    j=order[['image_key']].merge(labels[['image_key','food_present','size_1to10','size_source']].copy(),on='image_key',how='left',validate='one_to_one',sort=False)
    if j['image_key'].to_numpy().tolist() != order['image_key'].to_numpy().tolist():
        raise SystemExit('Image join changed order; refusing analysis')
    use=(j.food_present==1)&j.size_1to10.notna()
    if a.reviewed_only: use &= j.size_source=='manual'
    n_food=int(use.sum())
    print('NMF images:',len(order),'with any size:',int(j.size_1to10.notna().sum()),
          'eligible food images:',n_food,'reviewed-only:',a.reviewed_only)
    if n_food<a.min_food:
        raise SystemExit(f'Only {n_food} food images with size labels, need >= {a.min_food}. '
                         'Check missing local stimulus JPGs or add manual size scores.')
    sf=j.loc[use,'size_1to10'].to_numpy(dtype=float)
    if len(np.unique(sf))<3: raise SystemExit('Need at least 3 distinct size scores')
    groups=np.array([family(v) for v in j.loc[use,'image_key']])
    aa=np.asarray(A[use.to_numpy()],dtype=np.float64)
    size_ranks=rankdata(sf,method='average')
    component_ranks=rankdata(aa,axis=0,method='average')
    sr,sp,spadj=run_permutations(component_ranks,size_ranks,groups,a.permutations,a.seed)
    try:
        fr,fp,fpadj=run_permutations(component_ranks,size_ranks,groups,a.permutations,a.seed+1,stratified=True)
    except ValueError:
        fr=np.full(a.rank,np.nan);fp=fr.copy();fpadj=fr.copy()
        print('Within-family test unavailable because size has no within-family variation')
    out=(a.output_dir or cd/'realworld_food_size').expanduser()
    out.mkdir(parents=True,exist_ok=True)
    summary={'input_dir':str(root),'labels':str(a.labels),'rank':a.rank,'method':a.method,
             'images_total':len(order),'food_images_scored':n_food,
             'n_manual':int((j.loc[use,'size_source']=='manual').sum()),
             'n_clip_provisional':int((j.loc[use,'size_source']=='clip_provisional').sum()),
             'families':{str(k):int(v) for k,v in pd.Series(groups).value_counts().items()},
             'food_size_scale':'Ordinal typical real-life size of one identifiable food item; rice grain=1, watermelon=10',
             'primary_test':'food-only Spearman (both variables ranked); two-sided max-stat FWER across components',
             'within_family_test':'family-residualized ranked associations, label permutations restricted within family',
             'limits':'Exploratory; CLIP identification is provisional; size is not image prominence or Khosla food salience; small food sample, no independent validation'}
    ranked=pd.DataFrame({'image_key':j.loc[use,'image_key'].to_numpy(),
                         'size_1to10':sf,'size_source':j.loc[use,'size_source'].to_numpy(),
                         'stimulus_family':groups})
    comp=pd.DataFrame({'component':np.arange(1,a.rank+1),
                       'size_spearman_r':sr,'permutation_p_uncorrected':sp,
                       'permutation_p_FWER_20components':spadj,
                       'within_family_r':fr,'within_family_p_uncorrected':fp,
                       'within_family_p_FWER_20components':fpadj})
    comp=comp.sort_values('size_spearman_r',ascending=False)
    comp.to_csv(out/'component_food_realworld_size.csv',index=False)
    best_idx=int(comp.iloc[0].component)-1
    summary['strongest_positive_component']=best_idx+1
    summary['strongest_positive_r']=float(sr[best_idx])
    summary['strongest_positive_p_FWER']=float(spadj[best_idx])
    ranked['candidate_component_activation']=aa[:,best_idx]
    ranked.to_csv(out/'food_images_and_sizes.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4.7))
    ids=np.arange(1,a.rank+1)
    ax.bar(ids,sr,color=np.where(sr>=0,'#287aa8','#bc7761'))
    ax.axhline(0,color='black',linewidth=.7)
    ax.set(xticks=ids,xlabel='Consensus component',ylabel='Spearman r (food size)',
           title=f'Real-world food size vs NMF response (n={n_food}, food only)')
    fig.tight_layout();fig.savefig(out/'component_size_correlations.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    rng=np.random.default_rng(4)
    jitter=rng.uniform(-.12,.12,len(sf))
    z=zscore_columns(aa[:,[best_idx]])[:,0]
    ax.scatter(sf+jitter,z,alpha=.75,s=35)
    ax.set(xticks=np.arange(1,11),xlabel='Typical physical size (1 rice grain - 10 watermelon)',
           ylabel=f'Component {best_idx+1} activation (z-score)',
           title=f'Candidate component {best_idx+1}: r={sr[best_idx]:.3f}, FWER p={spadj[best_idx]:.3f}')
    fig.tight_layout();fig.savefig(out/'candidate_component_size_scatter.png',dpi=180);plt.close(fig)
    if not a.no_voxels:
        matrix_path=root/'X_image_by_voxel.npy'
        if not matrix_path.exists(): raise SystemExit(f'Cannot find image-by-voxel input: {matrix_path}')
        raw=np.load(matrix_path,mmap_mode='r')
        if raw.shape[0]!=len(order):
            raise SystemExit('Voxel matrix row count disagrees with image_order.csv')
        X=np.asarray(raw[use.to_numpy()],dtype=np.float64)
        Xr=rankdata(X,axis=0,method='average')
        print('Testing voxelwise association across',X.shape[1],'voxels...',flush=True)
        vr,vp,vpadj=run_permutations(Xr,size_ranks,groups,a.permutations,a.seed+2)
        try:
            vfr,vfp,vfpadj=run_permutations(Xr,size_ranks,groups,a.permutations,a.seed+3,stratified=True)
        except ValueError:
            vfr=np.full(len(vr),np.nan);vfp=vfr.copy();vfpadj=vfr.copy()
        np.savez_compressed(out/'voxel_food_size_stats.npz',spearman_r=vr,
                            p_uncorrected=vp,p_FWER_voxels=vpadj,
                            within_family_r=vfr,within_family_p_uncorrected=vfp,
                            within_family_p_FWER_voxels=vfpadj)
        summary['voxel_count']=len(vr)
        summary['n_voxels_p_FWER_below_005']=int((vpadj<.05).sum())
        summary['n_voxels_within_family_FWER_below_005']=int((vfpadj<.05).sum())
        print('Voxelwise FWER-significant (p<0.05):',int((vpadj<.05).sum()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\nTop positive components:')
    print(comp.head(8).to_string(index=False,float_format=lambda x:f'{x:.4g}'))
    print('Saved:',out)
    print('IMPORTANT: This tests REAL-WORLD FOOD SIZE, not visual size/prominence or food-present-vs-absent.')

if __name__=='__main__':main()
