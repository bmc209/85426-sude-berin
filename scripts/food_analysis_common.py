"""Helpers shared by the LAION-fMRI Moondream / NMF analysis scripts."""
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = Path.home() / "85426-sude-berin-local-data" / "nmf" / "sub-01" / "ses-01"
RESULTS_DIR = LOCAL_DIR / "nmf_results"
OUT_DIR = REPO_ROOT / "analysis" / "food_selectivity"


def image_key(value):
    """Match on normalized filename stem, not CSV image_index."""
    if pd.isna(value):
        return ""
    # A metadata label may be a filename or a full path.
    name = str(value).strip().strip('"\'').replace('\\', '/').split('/')[-1]
    return Path(name).stem.casefold()


def read_metadata(path=None):
    if path is None:
        candidates = []
        if LOCAL_DIR.exists():
            for ext in ('*.csv', '*.tsv'):
                candidates.extend(LOCAL_DIR.rglob(ext))
        matches = []
        for candidate in candidates:
            try:
                separator = '\t' if candidate.suffix.lower() == '.tsv' else ','
                head = pd.read_csv(candidate, sep=separator, nrows=2)
                if 'label' in head.columns and ('volume_index' in head.columns or 'beta_index' in head.columns):
                    matches.append(candidate)
            except Exception:
                pass
        if len(matches) != 1:
            raise SystemExit(
                f'Could not uniquely locate trial metadata under {LOCAL_DIR}. '
                f'Found {len(matches)} candidates: {[str(p) for p in matches[:10]]}. '
                'Pass --metadata /full/path/to/trial_metadata.csv'
            )
        path = matches[0]
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f'Metadata file does not exist: {path}')
    df = pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',')
    if 'label' not in df.columns:
        raise SystemExit(f'Metadata must have a "label" column; got: {list(df.columns)}')
    if 'volume_index' in df.columns:
        idx = pd.to_numeric(df['volume_index'], errors='raise')
        if idx.isna().any() or idx.duplicated().any():
            raise SystemExit('volume_index contains missing or duplicate values: check metadata alignment.')
        # Metadata volume_index comes from the 4D betas; sort and verify its sequence.
        df = df.assign(volume_index=idx.astype(int)).sort_values('volume_index').reset_index(drop=True)
        if not np.array_equal(df['volume_index'].to_numpy(), np.arange(len(df))) and not np.array_equal(
            df['volume_index'].to_numpy(), np.arange(1, len(df) + 1)
        ):
            raise SystemExit('volume_index is not consecutive 0-based or 1-based; verify extraction order manually.')
    else:
        raise SystemExit('Metadata needs volume_index to verify the fMRI trial order; do not silently assume row order.')
    df.insert(0, 'trial_index', np.arange(len(df)))
    df['image_key'] = df['label'].map(image_key)
    if (df['image_key'] == '').any():
        raise SystemExit('Metadata has missing image labels.')
    print(f'Loaded metadata: {path} ({len(df)} trials, {df.image_key.nunique()} unique image names)')
    return df


def load_alignment(path):
    path = Path(path).expanduser()
    df = pd.read_csv(path)
    required = {'trial_index', 'image_key', 'food_present'}
    if not required.issubset(df.columns):
        raise SystemExit(f'Missing alignment columns: {sorted(required - set(df.columns))}')
    if df['trial_index'].isna().any() or df['trial_index'].duplicated().any():
        raise SystemExit('Duplicate or missing trial_index in alignment.')
    df['trial_index'] = pd.to_numeric(df['trial_index'], errors='raise').astype(int)
    df = df.sort_values('trial_index').reset_index(drop=True)
    if not np.array_equal(df['trial_index'].to_numpy(), np.arange(len(df))):
        raise SystemExit('Alignment trial indices are not exactly 0..N-1.')
    df['food_present'] = pd.to_numeric(df['food_present'], errors='coerce')
    invalid = df['food_present'].notna() & ~df['food_present'].isin([0, 1])
    if invalid.any():
        raise SystemExit('food_present must be 0, 1, or blank.')
    for key, sub in df.groupby('image_key'):
        if sub['food_present'].dropna().nunique() > 1:
            raise SystemExit(f'Conflicting food labels for repeated image {key!r}')
    return df


def read_activations(rank, results_dir, expected_trials):
    path = Path(results_dir).expanduser() / f'rank_{rank}_trial_activations.npy'
    if not path.exists():
        print(f'Skipping rank {rank}: missing {path}')
        return None
    h = np.load(path)
    if h.ndim != 2 or h.shape != (expected_trials, rank):
        raise SystemExit(
            f'Unexpected activations shape {h.shape} for rank {rank}; '
            f'expected ({expected_trials}, {rank}). STOP: do not align by image_index.'
        )
    if not np.isfinite(h).all():
        raise SystemExit(f'Nonfinite activations in {path}')
    print(f'Loaded rank {rank}: {h.shape}')
    return np.asarray(h, dtype=float)


def collect_image_means(alignment, results_dir, ranks):
    """Average repeated trials by image, then combine components across ranks.

    This avoids treating repeated presentations of the same photograph as
    independent samples when testing image-level food labels.
    """
    keys = alignment['image_key']
    image_metadata = alignment.groupby('image_key', sort=True).first()
    cols = []
    data = []
    for rank in ranks:
        h = read_activations(rank, results_dir, len(alignment))
        if h is None:
            continue
        frame = pd.DataFrame(h, columns=[f'rank{rank}_comp{i + 1}' for i in range(rank)])
        # Group along image identities, not trial indices or CSV image_index.
        frame.insert(0, 'image_key', keys.to_numpy())
        grouped = frame.groupby('image_key', sort=True).mean()
        if not grouped.index.equals(image_metadata.index):
            raise SystemExit('Internal image grouping mismatch')
        data.append(grouped.to_numpy())
        cols.extend([(rank, i + 1) for i in range(rank)])
    if not data:
        raise SystemExit(f'No NMF activation files found in {results_dir}')
    X = np.concatenate(data, axis=1)
    return image_metadata, X, cols


def standardize_columns(arr):
    arr = np.asarray(arr, dtype=float)
    means = arr.mean(axis=0)
    std = arr.std(axis=0)
    valid = std > 1e-12
    out = np.zeros_like(arr)
    out[:, valid] = (arr[:, valid] - means[valid]) / std[valid]
    return out, valid


def permutation_correlations(X, y, n_permutations=5000, seed=42):
    """Two-sided permutation p-values, max-|r| family-wise correction.

    Permutes at unique-image level, and corrects across every supplied
    component and rank simultaneously. Handles tied ordinal scores using
    caller-supplied rank-transformed y for a Spearman-like analysis.
    """
    X, valid = standardize_columns(X)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all() or y.std() < 1e-12:
        raise SystemExit('Outcome has insufficient variation for correlation.')
    zy = (y - y.mean()) / y.std()
    observed = X.T @ zy / len(y)
    rng = np.random.default_rng(seed)
    raw_counts = np.ones(len(observed), dtype=int)
    max_counts = np.ones(len(observed), dtype=int)
    magnitudes = np.abs(observed)
    for _ in range(n_permutations):
        yp = rng.permutation(zy)
        p = np.abs(X.T @ yp / len(y))
        raw_counts += p >= (magnitudes - 1e-12)
        max_counts += p.max() >= (magnitudes - 1e-12)
    p_raw = raw_counts / (n_permutations + 1)
    p_family = max_counts / (n_permutations + 1)
    observed[~valid] = 0.0
    p_raw[~valid] = 1.0
    p_family[~valid] = 1.0
    return observed, p_raw, p_family
