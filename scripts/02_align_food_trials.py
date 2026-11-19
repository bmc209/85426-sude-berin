#!/usr/bin/env python3
"""Safely join Moondream food labels to extracted LAION-fMRI trial metadata."""
import argparse
from pathlib import Path
import pandas as pd
from food_analysis_common import OUT_DIR, read_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=OUT_DIR / 'food_label_review.csv')
    parser.add_argument('--metadata', type=Path, default=None,
                        help='Trial metadata CSV with volume_index and label columns. Auto-discovered if unique.')
    parser.add_argument('--out', type=Path, default=OUT_DIR / 'trial_food_alignment.csv')
    args = parser.parse_args()
    if not args.labels.exists():
        raise SystemExit(f'Missing {args.labels}; run 01_prepare_food_labels.py first.')
    labels = pd.read_csv(args.labels)
    needed = {'image_key', 'image_name', 'label', 'food_present'}
    if not needed.issubset(labels.columns):
        raise SystemExit(f'Label review file missing {needed - set(labels.columns)}')
    if labels.image_key.duplicated().any():
        raise SystemExit('Label review contains duplicate image_key values.')
    metadata = read_metadata(args.metadata)
    # Prevent conflicting label column names; preserve Moondream text distinctly.
    label_sub = labels.rename(columns={'image_name': 'moondream_image_name', 'label': 'moondream_label'})
    match = metadata.merge(label_sub, how='left', on='image_key', validate='many_to_one', indicator=True, sort=False)
    if len(match) != len(metadata) or not match['trial_index'].equals(metadata['trial_index']):
        raise SystemExit('Merge changed row count/order. STOP: image–trial alignment is not trustworthy.')
    unmatched = match[match['_merge'] != 'both']
    n_images = match.loc[match['_merge'] == 'both', 'image_key'].nunique()
    print(f'Matched trials: {len(match)-len(unmatched)}/{len(match)}; '
          f'unique matched images: {n_images}/{metadata.image_key.nunique()}')
    if len(unmatched):
        print('Some stimuli are missing from Moondream CSV. They will NOT be treated as non-food.')
        for name in unmatched['label'].drop_duplicates().head(15):
            print('  UNMATCHED:', name)
    if not n_images:
        raise SystemExit('ZERO image matches. Check that trial metadata uses compatible image filenames.')
    if len(match) < 20:
        print('WARNING: Very small trial metadata; not suitable for main inference.')
    match = match.drop(columns=['_merge'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    match.to_csv(args.out, index=False)
    usable = pd.to_numeric(match.food_present, errors='coerce').isin([0, 1]).sum()
    print(f'Wrote {args.out}; {usable} trials have provisional 0/1 food labels.')
    print('The label CSV image_index is NOT a trial index; this script uses normalized image filenames.')


if __name__ == '__main__':
    main()
