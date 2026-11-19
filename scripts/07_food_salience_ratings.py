#!/usr/bin/env python3
"""Create five-rater 0-9 salience template and merge reviewed scores into food labels.

IMPORTANT: The numbers must come from people reviewing ACTUAL STIMULUS IMAGES;
Moondream object descriptions are not salience ratings.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RATING_COLUMNS = [f'rating_{i}' for i in range(1, 6)]


def read_labels(path):
    df = pd.read_csv(path)
    needed = {'image_key', 'food_present'}
    if not needed <= set(df.columns):
        raise SystemExit(f'Label table lacks {sorted(needed - set(df.columns))}')
    if df.image_key.isna().any() or df.image_key.duplicated().any():
        raise SystemExit('Labels must contain unique nonmissing image_key')
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    c = commands.add_parser('create', help='Make blank template for 5 independent food-prominence raters')
    c.add_argument('--labels', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    m = commands.add_parser('merge', help='Validate 0..9 ratings and save mean food salience')
    m.add_argument('--labels', type=Path, required=True)
    m.add_argument('--ratings', type=Path, required=True)
    m.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    labels = read_labels(a.labels)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.action == 'create':
        columns = [k for k in ('image_key', 'image_name', 'label', 'food_present') if k in labels.columns]
        template = labels[columns].copy()
        for col in RATING_COLUMNS:
            template[col] = pd.Series([''] * len(template), dtype='str')
        if a.out.exists():
            raise SystemExit(f'Refusing to overwrite possibly filled ratings: {a.out}')
        template.to_csv(a.out, index=False)
        print(f'Created blank template for {len(template)} images at {a.out}')
        print('Rate actual images on the 0-9 food prominence scale with 5 independent raters.')
        return
    ratings = pd.read_csv(a.ratings)
    required = {'image_key', *RATING_COLUMNS}
    if not required <= set(ratings.columns):
        raise SystemExit(f'Rating file lacks: {sorted(required-set(ratings.columns))}')
    if ratings.image_key.duplicated().any() or ratings.image_key.isna().any():
        raise SystemExit('Rating table contains duplicate/blank image_key')
    if not set(ratings.image_key) <= set(labels.image_key):
        raise SystemExit('Rating table includes unknown image keys; cannot align')
    numeric = ratings[['image_key', *RATING_COLUMNS]].copy()
    for col in RATING_COLUMNS:
        numeric[col] = pd.to_numeric(numeric[col], errors='coerce')
        bad = numeric[col].notna() & ((numeric[col] < 0) | (numeric[col] > 9) |
                                       (numeric[col] != numeric[col].round()))
        if bad.any():
            raise SystemExit(f'Invalid scores in {col}: expected integers 0-9 or blank')
        invalid_text = ratings[col].notna() & (ratings[col].astype(str).str.strip() != '') & numeric[col].isna()
        if invalid_text.any():
            raise SystemExit(f'Non-numeric entries found in {col}')
    complete = numeric[RATING_COLUMNS].notna().all(axis=1)
    numeric['food_salience_0to9'] = numeric[RATING_COLUMNS].mean(axis=1).where(complete)
    joined = labels.merge(numeric[['image_key', 'food_salience_0to9']], on='image_key',
                          how='left', validate='one_to_one', sort=False)
    if 'food_salience_0to9_x' in joined or 'food_salience_0to9_y' in joined:
        raise SystemExit('Input labels already include food_salience_0to9; remove/rename first')
    if len(joined) != len(labels):
        raise SystemExit('Unexpected row-count change after merge')
    if not joined.food_salience_0to9.notna().any():
        raise SystemExit('No images have five complete ratings; refusing empty output')
    joined.to_csv(a.out, index=False)
    print(f'Saved {a.out}: {int(joined.food_salience_0to9.notna().sum())}/{len(labels)} images have 5 independent ratings')


if __name__ == '__main__':
    main()
