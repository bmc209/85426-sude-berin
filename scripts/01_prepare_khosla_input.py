#!/usr/bin/env python3
"""Session-wise voxel z-scores -> repeated-image average -> min shift (Khosla-style)."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from khosla_common import DEFAULT_METADATA, DEFAULT_RESPONSES, DEFAULT_OUTPUT, metadata, stim_family


def prepare(responses, meta, out_dir, min_repeats=1, expected_voxels=None):
    metadata_source = str(meta)
    meta = metadata(meta)
    n = len(meta)
    x = np.load(responses, mmap_mode='r')
    if x.ndim != 2:
        raise SystemExit('Response array must be 2D')
    if x.shape[0] == n:
        orient = 'trials_by_voxels'
        # Keep a memory-mapped view for multi-session datasets.
    elif x.shape[1] == n:
        orient = 'voxels_by_trials'
        x = x.T  # memory-mapped view; no full-array copy
    else:
        raise SystemExit(f'Response shape {x.shape} does not match {n} trial metadata rows')
    if not np.isfinite(x).all():
        raise SystemExit('Response matrix has NaN/Inf')
    if expected_voxels is not None and x.shape[1] != expected_voxels:
        raise SystemExit(f'Expected {expected_voxels} ROI voxels; got {x.shape[1]}')
    print(f'Loaded {n} trials x {x.shape[1]} voxels (source orientation: {orient})')
    if 'session' not in meta.columns:
        raise SystemExit('Metadata has no session column; session-wise normalization needs it')
    # When original responses were min-shifted per voxel only, subsequent z-score is
    # algebraically identical to z-scoring the original unshifted responses.
    # Allocate only the unique-image sums, not a second full trial x voxel array.
    img_ids, unique_keys = pd.factorize(meta.image_key, sort=True)
    counts = np.bincount(img_ids, minlength=len(unique_keys))
    sums = np.zeros((len(unique_keys), x.shape[1]), dtype=np.float64)
    session_stats=[]
    for sess, idx in meta.groupby('session', sort=False).indices.items():
        ix = np.asarray(idx, int)
        xm = np.asarray(x[ix], np.float64)
        avg = xm.mean(axis=0)
        sd = xm.std(axis=0)
        good = sd > 1e-8
        zi = np.zeros_like(xm, dtype=np.float32)
        zi[:,good] = ((xm[:,good]-avg[good])/sd[good]).astype(np.float32)
        np.add.at(sums, img_ids[ix], zi)
        session_stats.append({'session':str(sess),'trials':int(len(ix)),
                              'near_constant_voxels':int((~good).sum())})
        print(f'Session {sess}: {len(ix)} trials; {int((~good).sum())} near-constant voxels')
    # Rows are distinct image identities, with equal weight regardless of repetition.
    means = (sums / counts[:,None]).astype(np.float32)
    keep = counts >= min_repeats
    kept_keys = np.asarray(unique_keys)[keep]
    kept_counts = counts[keep]
    means = means[keep]
    # Paper's final baseline shift is across *unique image* responses after averaging.
    shifted = means - means.min(axis=0, keepdims=True)
    if shifted.min() < -1e-5 or not np.isfinite(shifted).all():
        raise SystemExit('Invalid shifted input')
    shifted = np.maximum(shifted, 0).astype(np.float32)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir/'X_image_by_voxel.npy', shifted)
    per_image = pd.DataFrame({'image_key':kept_keys,'n_presentations':kept_counts})
    per_image['stimulus_family'] = per_image.image_key.map(stim_family)
    per_image.to_csv(out_dir/'image_order.csv',index=False)
    summary={'responses_source':str(responses),'metadata_source':metadata_source,
             'trial_count':n,'voxel_count':int(x.shape[1]),'distinct_images_before_min_repeats':int(len(unique_keys)),
             'images_kept':int(len(kept_keys)),'min_repeats':min_repeats,
             'preprocessing':['session-wise per-voxel z-score','mean over repeated presentations of each image','voxel minimum shift'],
             'sessions':session_stats,'input_orientation':orient,
             'ROI_warning':'A 12,020 voxel general mask is NOT verified to equal the paper higher-level ventral stream ROI.'}
    (out_dir/'prepare_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Saved {shifted.shape} image x voxel NMF matrix in {out_dir}')
    print('Repetitions distribution:',pd.Series(counts).value_counts().sort_index().to_dict())
    print('IMPORTANT: confirm that input voxels are restricted to higher-level ventral visual ROI if claiming close replication')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata',type=Path,default=DEFAULT_METADATA)
    ap.add_argument('--responses',type=Path,default=DEFAULT_RESPONSES)
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--min-repeats',type=int,default=1, help='1 for LAION adaptation, 3 mimics paper repetition filter')
    ap.add_argument('--expected-voxels',type=int,default=None)
    a=ap.parse_args()
    if a.min_repeats<1: ap.error('--min-repeats must be >= 1')
    prepare(a.responses.expanduser(), a.metadata.expanduser(),a.out_dir.expanduser(),a.min_repeats,a.expected_voxels)

if __name__=='__main__':main()
