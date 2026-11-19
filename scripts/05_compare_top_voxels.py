#!/usr/bin/env python3
"""Paper-inspired control: food association of consensus component vs top 1% weighted voxels."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from khosla_common import DEFAULT_OUTPUT,DEFAULT_ALIGNMENT,food_alignment,corr_columns


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--alignment',type=Path,default=DEFAULT_ALIGNMENT)
    ap.add_argument('--method',choices=['bayesian','sklearn'],default='bayesian')
    ap.add_argument('--rank',type=int,default=20)
    ap.add_argument('--top-percent',type=float,default=1.0)
    a=ap.parse_args()
    root=a.input_dir.expanduser()
    consensus=root/f'consensus_{a.method}_rank{a.rank}'
    A=np.load(consensus/'consensus_image_activations.npy')
    B=np.load(consensus/'consensus_voxel_weights.npy')
    X=np.load(root/'X_image_by_voxel.npy',mmap_mode='r')
    image_order=pd.read_csv(root/'image_order.csv')
    labs=food_alignment(a.alignment)
    known=image_order.image_key.map(labs.food_present).isin([0,1]).to_numpy()
    y=image_order.image_key.map(labs.food_present)[known].to_numpy(dtype=int)
    if sum(y)<5 or sum(1-y)<5:raise SystemExit('Insufficient labeled images')
    if X.shape!=(len(image_order),B.shape[1]) or A.shape!=(len(image_order),a.rank):
        raise SystemExit('Voxel data, image order and consensus dimensions mismatch')
    nv=max(1,int(np.ceil(B.shape[1]*a.top_percent/100.)))
    rows=[]
    for k in range(a.rank):
        top=np.argsort(B[k])[-nv:]
        vox=np.asarray(X[:,top],float).mean(axis=1)
        r_comp=float(corr_columns(A[known,k:k+1],y)[0])
        r_vox=float(corr_columns(vox[known,None],y)[0])
        rows.append({'component':k+1,'component_food_r':r_comp,
                     'top_voxels_food_r':r_vox,'difference_r':r_comp-r_vox,
                     'top_n_voxels':nv})
    d=pd.DataFrame(rows).sort_values('component_food_r',ascending=False)
    target=consensus/'food_evaluation'
    target.mkdir(exist_ok=True)
    d.to_csv(target/'component_vs_top1pct_voxels.csv',index=False)
    print(d.head(10).to_string(index=False))
    print('Saved',target/'component_vs_top1pct_voxels.csv')
    print('Exploratory control only: correlations are related and CIs/paired tests are not provided.')

if __name__=='__main__':main()
