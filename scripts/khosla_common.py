"""Shared utilities for Khosla-style LAION-fMRI replication (one image per row)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = Path.home() / '85426-sude-berin-local-data/processed/sub-01/ses-01/trial_metadata.tsv'
DEFAULT_RESPONSES = Path.home() / '85426-sude-berin-local-data/nmf/sub-01/ses-01/responses_shifted.npy'
DEFAULT_OUTPUT = ROOT / 'analysis/khosla_replication/sub-01/ses-01'
DEFAULT_ALIGNMENT = ROOT / 'analysis/food_selectivity/food_label_review.csv'


def image_key(v):
    if pd.isna(v):
        return ''
    s = str(v).strip().strip('"\'').replace('\\', '/').split('/')[-1]
    return Path(s).stem.casefold()


def metadata(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise SystemExit(f'No trial metadata: {path}')
    df = pd.read_csv(path, sep='\t' if path.suffix == '.tsv' else ',')
    for col in ['label', 'volume_index']:
        if col not in df.columns:
            raise SystemExit(f'Trial metadata lacks {col}; available columns: {df.columns.tolist()}')
    idx = pd.to_numeric(df.volume_index, errors='raise').astype(int)
    if idx.duplicated().any():
        raise SystemExit('Duplicate volume_index; cannot verify trial ordering')
    df = df.assign(volume_index=idx).sort_values('volume_index').reset_index(drop=True)
    if not (np.array_equal(df.volume_index.to_numpy(), np.arange(len(df))) or
            np.array_equal(df.volume_index.to_numpy(), np.arange(1, len(df)+1))):
        raise SystemExit('volume_index is neither consecutive 0-based nor 1-based: check extraction')
    df['image_key'] = df.label.map(image_key)
    if not df.image_key.astype(bool).all():
        raise SystemExit('Empty metadata image label')
    return df


def stim_family(key):
    k = str(key).upper()
    for x in ('LAION', 'NSD', 'THINGS+', 'OOD'):
        if x in k:
            return x
    return 'OTHER'


def standardize(a):
    a = np.asarray(a, dtype=np.float64)
    sd = a.std(axis=0)
    z = np.zeros_like(a)
    good = sd > 1e-12
    z[:, good] = (a[:, good] - a[:, good].mean(axis=0)) / sd[good]
    return z, good


def corr_columns(X, y):
    z, _ = standardize(X)
    y = np.asarray(y, float)
    if y.std() <= 1e-12:
        raise ValueError('Label variable is constant')
    yz = (y-y.mean()) / y.std()
    return z.T @ yz / len(y)


def food_alignment(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f'Alignment not found: {path}. Use analysis/food_selectivity/food_label_review.csv from the earlier Moondream package.')
    df = pd.read_csv(path)
    if not {'image_key','food_present'} <= set(df.columns):
        raise SystemExit('Alignment needs image_key and food_present columns')
    df['image_key'] = df.image_key.map(image_key)
    df['food_present'] = pd.to_numeric(df.food_present, errors='coerce')
    for k, sub in df.groupby('image_key'):
        vals = sub.food_present.dropna().unique()
        if len(vals)>1 or any(v not in (0,1) for v in vals):
            raise SystemExit(f'Conflicting or invalid food_present for {k}')
    return df.drop_duplicates('image_key').set_index('image_key')
