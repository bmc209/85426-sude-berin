#!/usr/bin/env python3
"""Associate consensus component responses with reviewed food labels.

Unique-image inference; 20-component family-wise permutation correction;
within-family sensitivity; optional label-heldout check. No p-hacking by rank.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from khosla_common import DEFAULT_OUTPUT, DEFAULT_ALIGNMENT, food_alignment, standardize, stim_family


def residual_family(X,y,families):
    xx=np.asarray(X,float).copy(); yy=np.asarray(y,float).copy()
    for f in np.unique(families):
        keep=families==f
        xx[keep]-=xx[keep].mean(axis=0)
        yy[keep]-=yy[keep].mean()
    return xx,yy


def permtest(X,y,families,nperm=5000,seed=42,within_family=False):
    X=np.asarray(X,float)
    y=np.asarray(y,float)
    if within_family: X,y=residual_family(X,y,families)
    X,_=standardize(X)
    sd=y.std()
    if sd<1e-10:raise SystemExit('No label variation in test')
    yc=(y-y.mean())/sd
    r=X.T@yc/len(y)
    rng=np.random.default_rng(seed)
    hits=np.ones(len(r),dtype=int)
    fwer=np.ones(len(r),dtype=int)
    if within_family:
        groups=[np.flatnonzero(families==f) for f in np.unique(families)]
        # Important: shuffle ORIGINAL binary labels within family and residualize anew.
    for _ in range(nperm):
        yp=yc.copy()
        if within_family:
            for g in groups:yp[g]=yp[rng.permutation(g)]
        else: yp=yp[rng.permutation(len(yp))]
        rp=X.T@yp/len(y)
        abs_max=np.max(abs(rp))
        hits += (abs(rp)>=abs(r)-1e-12)
        fwer += (abs_max>=abs(r)-1e-12)
    return r, hits/(nperm+1), fwer/(nperm+1)


def auc(vals,y):
    ix=y==1; n1=int(ix.sum()); n0=int((~ix).sum())
    ranks=rankdata(np.concatenate([vals[ix],vals[~ix]]),method='average')
    return float((ranks[:n1].sum()-n1*(n1+1)/2)/(n1*n0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--alignment',type=Path,default=DEFAULT_ALIGNMENT)
    ap.add_argument('--method',choices=['bayesian','sklearn'],default='bayesian')
    ap.add_argument('--rank',type=int,default=20)
    ap.add_argument('--permutations',type=int,default=5000)
    ap.add_argument('--salience-column',default=None,help='Optional reviewed continuous 0-9 score column in alignment CSV')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--heldout-fraction',type=float,default=0.3,
                    help='Image-level split for exploratory component selection vs test; set 0 to disable')
    a=ap.parse_args()
    if a.permutations<100:ap.error('Require >=100 permutations, recommend >=5000')
    root=a.input_dir.expanduser()
    d=root/f'consensus_{a.method}_rank{a.rank}'
    A=np.load(d/'consensus_image_activations.npy')
    order=pd.read_csv(root/'image_order.csv')
    if A.shape!=(len(order),a.rank):raise SystemExit('Image order incompatible with consensus activations')
    labels=food_alignment(a.alignment)
    usecols=['food_present']
    if a.salience_column is not None:
        if a.salience_column not in labels.columns:
            raise SystemExit(f'Salience column {a.salience_column!r} missing from {a.alignment}')
        usecols.append(a.salience_column)
    j=order.set_index('image_key').join(labels[usecols],how='left')
    keep=j.food_present.isin([0,1]).to_numpy()
    X=A[keep]
    y=j.food_present.to_numpy()[keep].astype(int)
    images=j.index.to_numpy()[keep]
    fam=np.array([stim_family(k) for k in images])
    n_food=int(y.sum());n_nonfood=len(y)-n_food
    if min(n_food,n_nonfood)<5:raise SystemExit(f'Too few labeled images: food={n_food}, nonfood={n_nonfood}')
    print(f'{len(y)} distinct labeled images / {len(order)} total: {n_food} food and {n_nonfood} nonfood')
    print('Food counts by stimulus family:\n',pd.crosstab(fam,y))
    Z,good=standardize(X)
    r,raw,adjust=permtest(X,y,fam,a.permutations,a.seed)
    fr,fraw,fadj=permtest(X,y,fam,a.permutations,a.seed,within_family=True)
    rows=[]
    for k in range(a.rank):
        rows.append({'component':k+1,'food_r':r[k],
             'food_minus_nonfood_sd':Z[y==1,k].mean()-Z[y==0,k].mean(),
             'food_higher_auc':auc(Z[:,k],y),
             'p_uncorrected':raw[k], 'p_FWER_20components':adjust[k],
             'within_family_r':fr[k], 'within_family_p_uncorrected':fraw[k],
             'within_family_p_FWER_20components':fadj[k],
             'n_food':n_food,'n_nonfood':n_nonfood,
             'constant_component':not bool(good[k])})
    out=d/'food_evaluation';out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows).sort_values('food_r',ascending=False)
    df.to_csv(out/'component_food_selectivity.csv',index=False)
    best=int(df.iloc[0].component)-1
    ranked=pd.DataFrame({'image_key':images,'food_present':y,'stimulus_family':fam,
                         'component_activation_z':Z[:,best]})
    ranked.sort_values('component_activation_z',ascending=False).to_csv(out/'best_component_images.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    order_c=df.sort_values('component')
    ax.bar(order_c.component,order_c.food_r)
    ax.axhline(0,color='k',linewidth=.7)
    ax.set(xlabel='Consensus NMF component (1-based)',ylabel='Food label vs response (Pearson r)',
           title=f'20-component consensus ({a.method}); {n_food} food images')
    fig.tight_layout();fig.savefig(out/'all_component_food_correlations.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    rng=np.random.default_rng(12)
    for cls,label in [(0,'Non-food'),(1,'Food')]:
        vals=Z[y==cls,best]
        ax.scatter(rng.normal(cls,.075,len(vals)),vals,alpha=.5,s=14,label=f'{label}: {len(vals)}')
    ax.set(xticks=[0,1],xticklabels=['Non-food','Food'],ylabel='Component response (within-component z-score)',
           title=f'Candidate component {best+1}; r={r[best]:.3f}; corrected p={adjust[best]:.3f}')
    ax.legend();fig.tight_layout();fig.savefig(out/'best_component_food_scatter.png',dpi=180);plt.close(fig)
    summary={'method':a.method,'rank':a.rank,'labeled_images':int(len(y)),
             'food_images':n_food,'nonfood_images':n_nonfood,
             'test':'two-sided, unique-image label permutation; max absolute r over all 20 components',
             'labels_caveat':'Provisional keyword-derived labels, not validated human food salience ratings',
             'best_food_component':best+1,'best_food_r':float(r[best]),
             'best_food_corrected_p':float(adjust[best]),
             'not_a_paper_replication':'Paper used human-rated continuous food salience and 8 participant consistency; this is a one-subject LAION adaptation'}
    # Discovery/validation split uses labels only AFTER unsupervised factorization.
    if a.heldout_fraction>0:
        from sklearn.model_selection import train_test_split
        ids=np.arange(len(y))
        train,test=train_test_split(ids,test_size=a.heldout_fraction,random_state=a.seed,stratify=y)
        tr_corr=np.array([np.corrcoef(X[train,k],y[train])[0,1] if X[train,k].std()>1e-12 else 0 for k in range(a.rank)])
        selected=int(np.nanargmax(tr_corr))
        test_r,test_p,_=permtest(X[test][:,[selected]],
                                y[test],fam[test],a.permutations,a.seed+1)
        # Exact selection procedure is fixed on discovery split, so heldout p is single-component.
        summary['heldout']={'split_fraction':a.heldout_fraction,'component_selected_in_train':selected+1,
                            'train_r':float(tr_corr[selected]),'n_test':int(len(test)),
                            'n_test_food':int(y[test].sum()),'test_r':float(test_r[0]),
                            'test_p_two_sided_single_component':float(test_p[0]),
                            'caveat':'Weak power with few food examples; heldout labels not used to select component.'}
        print('Held-out selected in train:',selected+1,'test r:',float(test_r[0]),'p:',float(test_p[0]))
    if a.salience_column is not None:
        from scipy.stats import rankdata
        scores=pd.to_numeric(j[a.salience_column],errors='coerce').to_numpy()
        mask=np.isfinite(scores)
        if mask.sum()<30 or len(np.unique(scores[mask]))<3:
            raise SystemExit('Not enough reviewed salience scores to analyze (need >=30 scored images, >=3 distinct values).')
        scored_x=A[mask]; scored_y=scores[mask];scored_fam=np.array([stim_family(k) for k in j.index.to_numpy()[mask]])
        # Pearson is comparable with paper's continuous human salience correlations.
        sr,sp,spadj=permtest(scored_x,scored_y,scored_fam,a.permutations,a.seed)
        # Spearman as robust sensitivity to ordinal ratings.
        spr,sprp,spradj=permtest(scored_x,rankdata(scored_y),scored_fam,a.permutations,a.seed+10)
        pd.DataFrame({'component':np.arange(1,a.rank+1),'salience_pearson_r':sr,
                      'pearson_p_uncorrected':sp,'pearson_p_fwer':spadj,
                      'salience_spearman_r':spr,'spearman_p_fwer':spradj}).to_csv(out/'component_food_salience.csv',index=False)
        summary['salience']={'column':a.salience_column,'n_scored':int(mask.sum()),
            'note':'Salience was supplied by user; results are only comparable to Khosla if 0-9 ratings were collected consistently from actual images.'}
        print('Saved optional salience correlations for',mask.sum(),'images')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(df[['component','food_r','p_FWER_20components','within_family_r','within_family_p_FWER_20components']].head(10).to_string(index=False))
    print('Saved:',out)
    print('Food labels are provisional. Significant corrected p in this exploratory pipeline would still require independent replication.')

if __name__=='__main__':main()
