#!/usr/bin/env python3
"""Voxel-wise OLS of fMRI response on real-world food size (1-10).

Uses X_image_by_voxel.npy and image_order.csv from 01_prepare_khosla_input.py.
Only food-present, size-scored images are included. Optional fixed effects for
stimulus family; coefficient is response-unit change per 1 size point.
Permutation max-|t| FWER uses Freedman-Lane nuisance residual permutation,
restricted within family when family adjustment is enabled.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def norm_key(s):
    if pd.isna(s): return ''
    return Path(str(s).strip().strip('"\'').replace('\\','/').split('/')[-1]).stem.casefold()


def family(s):
    v=str(s).upper()
    for f in ('LAION','NSD','OOD','THINGS+'):
        if f in v: return f
    return 'OTHER'


def bh(p):
    p=np.asarray(p,dtype=float)
    out=np.full(len(p),np.nan)
    ok=np.flatnonzero(np.isfinite(p))
    if not len(ok): return out
    ordering=ok[np.argsort(p[ok])]
    q=np.minimum.accumulate((p[ordering]*len(ok)/np.arange(1,len(ok)+1))[::-1])[::-1]
    out[ordering]=np.minimum(q,1)
    return out


def ols_t_and_beta(Y, size, nuisance):
    """FWL: regress both Y and size on nuisance, then compute size slope."""
    Q,_=np.linalg.qr(nuisance,mode='reduced')
    yr=Y-Q@(Q.T@Y)
    xr=size-Q@(Q.T@size)
    ss=float(xr@xr)
    if ss<1e-10: raise ValueError('Size has no variance after nuisance adjustment')
    beta=(xr@yr)/ss
    residual=yr-xr[:,None]*beta[None,:]
    df=Y.shape[0]-nuisance.shape[1]-1
    if df<3: raise ValueError('Insufficient residual degrees of freedom')
    sigma2=(residual*residual).sum(axis=0)/df
    se=np.sqrt(sigma2/ss)
    tstat=np.divide(beta,se,out=np.zeros_like(beta),where=se>1e-12)
    r_partial=tstat/np.sqrt(tstat*tstat+df)
    p=2*student_t.sf(np.abs(tstat),df)
    return beta,se,tstat,p,r_partial,yr,xr,Q,df


def map_and_plot(mask_file, beta, tstat, q_fdr, p_fwer, out):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit('nibabel required for brain maps: pip install nibabel') from exc
    ref=nib.load(str(mask_file))
    mask=np.asarray(ref.dataobj)>0
    if mask.ndim!=3: raise ValueError('Expected 3D mask')
    if int(mask.sum())!=len(beta):
        raise ValueError(f'Mask has {int(mask.sum())} voxels but analysis has {len(beta)}; refusing to map')
    def volume(v):
        arr=np.zeros(mask.shape,dtype=np.float32)
        arr[mask]=np.asarray(v,dtype=np.float32)
        return arr
    for name,vector in [('size_beta_per_point',beta),('size_t_statistic',tstat),
                        ('size_fdr_q',q_fdr),('size_fwer_p',p_fwer),
                        ('size_fdr_significant_beta',np.where(q_fdr<.05,beta,0)),
                        ('size_fwer_significant_beta',np.where(p_fwer<.05,beta,0))]:
        nii=nib.Nifti1Image(volume(vector),ref.affine,ref.header.copy())
        nii.set_data_dtype(np.float32)
        nib.save(nii,str(out/f'{name}_nativeT1w.nii.gz'))
    try:
        from nilearn import plotting
        img=nib.Nifti1Image(volume(tstat),ref.affine)
        fig=plt.figure(figsize=(12,4))
        display=plotting.plot_stat_map(img,display_mode='ortho',threshold=2.0,
            cmap='cold_hot',colorbar=True,figure=fig,
            title='Exploratory voxelwise real-world food size regression: t statistic')
        display.savefig(str(out/'size_tstat_brain_slices.png'),dpi=180)
        display.close()
    except Exception as e:
        print('NIfTI maps saved; nilearn slice plotting unavailable:',repr(e))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True,
                   help='Directory containing image_order.csv and X_image_by_voxel.npy')
    p.add_argument('--labels',type=Path,required=True,
                   help='CSV with image_key,food_present,size_1to10')
    p.add_argument('--mask',type=Path,default=None,
                   help='Exact native laionventral NIfTI mask; enables NIfTI brain maps')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--permutations',type=int,default=2000,
                   help='Max-T FWER permutations; 5000-10000 recommended for final results')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-family-adjustment',action='store_true')
    p.add_argument('--reviewed-only',action='store_true',help='Only size_source=manual')
    p.add_argument('--min-food',type=int,default=12)
    a=p.parse_args()
    if a.permutations<0: p.error('permutations cannot be negative')
    root=a.input_dir.expanduser()
    order=pd.read_csv(root/'image_order.csv')
    labels=pd.read_csv(a.labels.expanduser())
    for df,label in [(order,'order'),(labels,'labels')]:
        if 'image_key' not in df: raise SystemExit(f'{label} missing image_key')
        df['image_key']=df.image_key.map(norm_key)
        if (df.image_key=='').any() or not df.image_key.is_unique:
            raise SystemExit(f'{label} has duplicate/empty image_key; fix before proceeding')
    required={'food_present','size_1to10'}
    if not required.issubset(labels):
        raise SystemExit(f'Labels missing columns {required-set(labels.columns)}. Run script 12 first.')
    labels['food_present']=pd.to_numeric(labels.food_present,errors='coerce')
    labels['size_1to10']=pd.to_numeric(labels.size_1to10,errors='coerce')
    if 'size_source' not in labels:labels['size_source']='unspecified'
    if labels.size_1to10.notna().any() and not labels.size_1to10.dropna().between(1,10).all():
        raise SystemExit('size_1to10 must range from 1 to 10')
    joined=order[['image_key']].merge(labels[['image_key','food_present','size_1to10','size_source']],
                                      on='image_key',how='left',validate='one_to_one',sort=False)
    if not np.array_equal(joined.image_key.to_numpy(),order.image_key.to_numpy()):
        raise SystemExit('Image order changed during merge')
    select=(joined.food_present.eq(1)&joined.size_1to10.notna())
    if a.reviewed_only: select &= joined.size_source.eq('manual')
    n=int(select.sum())
    if n<a.min_food:raise SystemExit(f'Only {n} food images with size ratings; need {a.min_food}')
    size=joined.loc[select,'size_1to10'].to_numpy(dtype=float)
    if np.unique(size).size<3:raise SystemExit('Need >=3 unique size levels')
    groups=np.array([family(s) for s in joined.loc[select,'image_key']])
    X=np.load(root/'X_image_by_voxel.npy',mmap_mode='r')
    if X.shape[0]!=len(order):raise SystemExit(f'X shape {X.shape} incompatible with {len(order)} order rows')
    Y=np.asarray(X[select.to_numpy()],dtype=np.float64)
    if not np.isfinite(Y).all():raise SystemExit('Nonfinite fMRI values')
    if a.no_family_adjustment:
        N=np.ones((n,1))
    else:
        dummies=pd.get_dummies(pd.Series(groups),drop_first=True,dtype=float).to_numpy()
        N=np.column_stack([np.ones(n),dummies])
    beta,se,tstat,pval,partial_r,yr,xr,Q,df=ols_t_and_beta(Y,size,N)
    q=bh(pval)
    # Freedman-Lane: reduced-model fitted Y and nuisance residuals.
    # Permute residuals within family; regress permuted data against the same full model.
    fwer=np.full(Y.shape[1],np.nan)
    if a.permutations:
        fitted=Q@(Q.T@Y)
        yr0=Y-fitted
        rng=np.random.default_rng(a.seed)
        idxgroups=[np.flatnonzero(groups==g) for g in np.unique(groups)]
        max_t=np.empty(a.permutations)
        print(f'Running {a.permutations} max-|t| permutations on {Y.shape[1]} voxels...',flush=True)
        for i in range(a.permutations):
            ix=np.arange(n)
            for indices in idxgroups:ix[indices]=rng.permutation(indices)
            Yperm=fitted+yr0[ix]
            _,_,tp,_,_,_,_,_,_=ols_t_and_beta(Yperm,size,N)
            max_t[i]=float(np.max(np.abs(tp)))
            if (i+1)%1000==0:print(f'  {i+1}/{a.permutations}',flush=True)
        fwer=(1+(max_t[None,:]>=np.abs(tstat[:,None])-1e-12).sum(axis=1))/(a.permutations+1)
    out=(a.output_dir or root/'voxelwise_food_size_regression').expanduser()
    out.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame({'voxel_index_in_mask_order':np.arange(len(beta)),
        'beta_per_size_point':beta,'standard_error':se,'t_statistic':tstat,
        'degrees_of_freedom':df,'p_uncorrected':pval,'q_FDR_BH':q,
        'p_FWER_maxT':fwer,'partial_r':partial_r})
    result.to_csv(out/'voxelwise_food_size_regression.csv',index=False)
    joined.loc[select,['image_key','food_present','size_1to10','size_source']].assign(
        stimulus_family=groups).to_csv(out/'included_food_images.csv',index=False)
    summary={'images_total':len(order),'food_images_included':n,'voxels_tested':Y.shape[1],
      'size_distribution':{str(k):int(v) for k,v in pd.Series(size).value_counts().sort_index().items()},
      'family_distribution':{str(k):int(v) for k,v in pd.Series(groups).value_counts().items()},
      'family_adjusted':not a.no_family_adjustment,'residual_df':int(df),
      'manual_size_count':int((joined.loc[select,'size_source']=='manual').sum()),
      'provisional_size_count':int((joined.loc[select,'size_source']=='clip_provisional').sum()),
      'voxel_significant_FDR_005':int((q<.05).sum()),
      'voxel_significant_FWER_005':int(np.sum(fwer<.05)),
      'permutations':a.permutations,'interpretation':'beta: change in input image-averaged voxel response per 1-unit size increase, conditional on stimulus-family fixed effects when enabled. No causal interpretation. CLIP ratings provisional.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].hist(beta,bins=60)
    axs[0].axvline(0,color='black',lw=1)
    axs[0].set(xlabel='Voxel beta / 1 size point',ylabel='Voxel count',title='Size encoding across ventral ROI')
    axs[1].hist(tstat,bins=60)
    axs[1].axvline(0,color='black',lw=1)
    axs[1].set(xlabel='Voxel regression t statistic',ylabel='Voxel count',title=f'Food-only regression (n={n})')
    fig.tight_layout();fig.savefig(out/'voxel_size_regression_distributions.png',dpi=180);plt.close(fig)
    if a.mask is not None:map_and_plot(a.mask.expanduser(),beta,tstat,q,fwer,out)
    print(json.dumps(summary,indent=2))
    print('Saved results to:',out)
    print('NOTE: Original X matrix is shifted nonnegative; additive voxel-wise shifts do not alter regression slopes.')
    print('NOTE: Verify mask voxel ordering (mask>0, NumPy C-order) before interpreting spatial maps.')

if __name__=='__main__':main()
