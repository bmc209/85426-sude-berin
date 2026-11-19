#!/usr/bin/env python3
"""Optional cross-participant component reproducibility for shared images.

Match each participant's 20 components to a reference participant with Hungarian
maximum correlation on common images, then report pairwise correlation. This is
an approximation of paper's 4-subject exhaustive intersubject ranking.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dirs',nargs='+',type=Path,required=True,
                    help='>=2 output dirs from 00-03 for different participants')
    ap.add_argument('--method',choices=['bayesian','sklearn'],default='bayesian')
    ap.add_argument('--rank',type=int,default=20)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args()
    if len(a.input_dirs)<2:ap.error('Provide two or more participants')
    mats=[]
    for root in a.input_dirs:
        root=root.expanduser()
        df=pd.read_csv(root/'image_order.csv')
        keys=df.image_key.astype(str)
        if keys.duplicated().any():raise SystemExit(f'Duplicate unique image keys: {root}')
        A=np.load(root/f'consensus_{a.method}_rank{a.rank}/consensus_image_activations.npy')
        if A.shape!=(len(df),a.rank):raise SystemExit(f'Bad consensus shape: {root}')
        mats.append(pd.DataFrame(A,index=keys))
    common=sorted(set.intersection(*[set(x.index) for x in mats]))
    if len(common)<50:raise SystemExit(f'Only {len(common)} common images; need >=50 for cross-subject matching')
    X=[m.loc[common].to_numpy(dtype=float) for m in mats]
    X=[(v-v.mean(axis=0))/np.where(v.std(axis=0)>1e-10,v.std(axis=0),1) for v in X]
    mappings=[np.arange(a.rank)]
    for i in range(1,len(X)):
        r=X[0].T@X[i]/len(common)
        ri,ci=linear_sum_assignment(-r)
        mapping=np.zeros(a.rank,dtype=int)
        mapping[ri]=ci
        mappings.append(mapping)
    rows=[]
    for comp in range(a.rank):
        cs=[]
        for i in range(len(X)):
            for j in range(i+1,len(X)):
                r=np.corrcoef(X[i][:,mappings[i][comp]],X[j][:,mappings[j][comp]])[0,1]
                if np.isfinite(r):cs.append(float(r))
        rows.append({'reference_component':comp+1,'mean_pairwise_r':np.mean(cs),
                     'median_pairwise_r':np.median(cs),'n_pairs':len(cs),
                     'consistent_at_0p5':bool(np.median(cs)>=.5)})
    out=a.out_dir.expanduser();out.mkdir(parents=True,exist_ok=True)
    res=pd.DataFrame(rows).sort_values('median_pairwise_r',ascending=False)
    res.to_csv(out/'cross_subject_component_consistency.csv',index=False)
    print(f'Participants={len(X)}, common distinct images={len(common)}')
    print(res.head(10).to_string(index=False))
    print('Unlike the paper this uses reference-subject Hungarian alignment, not full multi-subject greedy search.')

if __name__=='__main__':main()
