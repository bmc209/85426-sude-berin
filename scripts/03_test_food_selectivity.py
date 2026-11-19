#!/usr/bin/env python3
"""Test how strongly NMF components are associated with food vs non-food labels.

Unit of inference: distinct image (average repeated presentations). Permute
food labels across images. Correct across *all components and ranks* with a
max-absolute-correlation permutation test. Preliminary until labels are reviewed.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from food_analysis_family import stimulus_family, within_family_permutation
from food_analysis_common import (
    OUT_DIR, RESULTS_DIR, load_alignment, collect_image_means,
    standardize_columns, permutation_correlations,
)


def auc_from_groups(food, other):
    """Mann–Whitney AUC: probability a random food image has higher activation."""
    n1, n0 = len(food), len(other)
    if n1 == 0 or n0 == 0:
        return float('nan')
    z = np.r_[food, other]
    ranks = rankdata(z, method='average')
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def plot_rank_bars(rank, sub, path):
    sub = sub.sort_values('component')
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(sub))))
    colors = ['#cc6677' if x > 0 else '#4477aa' for x in sub['food_minus_nonfood_sd']]
    ax.barh(sub['component'].astype(str), sub['food_minus_nonfood_sd'], color=colors)
    ax.axvline(0, c='#555555', lw=1)
    ax.invert_yaxis()
    ax.set_xlabel('Mean activation: food − non-food (SD across unique images)')
    ax.set_ylabel('NMF component')
    ax.set_title(f'Rank {rank}: association with provisional food labels')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_top_component(rank, component, values, labels, corr, p_adj, path):
    a, b = values[labels == 0], values[labels == 1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    rng = np.random.default_rng(123)
    ax.scatter(rng.normal(0, 0.075, len(a)), a, s=14, alpha=0.45, label=f'No food ({len(a)})')
    ax.scatter(rng.normal(1, 0.075, len(b)), b, s=14, alpha=0.45, label=f'Food ({len(b)})')
    for pos, group in [(0, a), (1, b)]:
        ax.plot([pos - 0.15, pos + 0.15], [np.median(group)] * 2, lw=3, c='black')
    ax.set_xticks([0, 1], ['No food', 'Food'])
    ax.set_ylabel('NMF activation (z-score across unique images)')
    ax.set_title(f'Rank {rank}, component {component}: food association\nr = {corr:.3f}, maxT p = {p_adj:.4f}')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, default=OUT_DIR / 'trial_food_alignment.csv')
    parser.add_argument('--results', type=Path, default=RESULTS_DIR)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR / 'selectivity')
    parser.add_argument('--ranks', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--permutations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.permutations < 100:
        raise SystemExit('Use at least 100 permutations; recommend 5000 or more.')
    df = load_alignment(args.alignment)
    images, Xall, components = collect_image_means(df, args.results, args.ranks)
    known = images['food_present'].isin([0, 1]).to_numpy()
    X = Xall[known]
    images = images.loc[known].copy()
    y = images['food_present'].to_numpy(dtype=int)
    n_food = int(y.sum())
    n_nonfood = len(y) - n_food
    if min(n_food, n_nonfood) < 5:
        raise SystemExit(f'Not enough unique food vs non-food images: {n_food} food, {n_nonfood} non-food.')
    print(f'Analyzing {len(y)} DISTINCT images: {n_food} provisional food / {n_nonfood} non-food')
    print(f'Excluded {len(known)-int(known.sum())} unlabeled images; repeated presentations averaged.')
    if 'review_status' in images.columns:
        n_reviewed = int((images['review_status'].fillna('unreviewed') != 'unreviewed').sum())
        print(f'Label review status: {n_reviewed}/{len(images)} have changed from unreviewed.')
    families = images.index.to_series().map(stimulus_family).to_numpy()
    family_counts = pd.crosstab(pd.Series(families, name='stimulus_family'),
                                pd.Series(y, name='food_present'))
    family_counts = family_counts.rename(columns={0: 'nonfood_images', 1: 'food_images'})
    print('Food/non-food counts by stimulus family:')
    print(family_counts.to_string())
    Z, valid = standardize_columns(X)
    corrs, p_raw, p_adjusted = permutation_correlations(X, y, args.permutations, args.seed)
    adjusted = within_family_permutation(X, y, families, args.permutations, args.seed)
    if adjusted is None:
        print('WARNING: Insufficient within-family food/non-food variation for family-adjusted test.')
        fam_r = np.full(len(components), np.nan)
        fam_p = np.full(len(components), np.nan)
        fam_p_fwer = np.full(len(components), np.nan)
        n_family_images, families_used = 0, []
    else:
        fam_r, fam_p, fam_p_fwer, n_family_images, families_used = adjusted
        print(f'Within-family test: {n_family_images} images from {families_used}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    family_counts.to_csv(args.out_dir / 'family_food_counts.csv')
    rows = []
    for j, (rank, component) in enumerate(components):
        f = Z[y == 1, j]
        nf = Z[y == 0, j]
        delta = float(f.mean() - nf.mean())
        rows.append({
            'rank': rank, 'component': component,
            'point_biserial_r': float(corrs[j]),
            'food_minus_nonfood_sd': delta,
            'food_mean_z': float(f.mean()), 'nonfood_mean_z': float(nf.mean()),
            'food_higher_auc': auc_from_groups(f, nf),
            'permutation_p_uncorrected': float(p_raw[j]),
            'permutation_p_FWER_all_ranks': float(p_adjusted[j]),
            'within_family_r': float(fam_r[j]),
            'within_family_permutation_p_uncorrected': float(fam_p[j]),
            'within_family_permutation_p_FWER_all_ranks': float(fam_p_fwer[j]),
            'n_food_images': n_food, 'n_nonfood_images': n_nonfood,
            'constant_component': not bool(valid[j]),
        })
    stats = pd.DataFrame(rows).sort_values(
        ['permutation_p_FWER_all_ranks', 'permutation_p_uncorrected', 'point_biserial_r'],
        ascending=[True, True, False],
    )
    out_csv = args.out_dir / 'component_food_selectivity.csv'
    stats.to_csv(out_csv, index=False)
    print(f'\nSaved {out_csv}')
    print(stats[['rank', 'component', 'point_biserial_r', 'food_minus_nonfood_sd',
                 'permutation_p_FWER_all_ranks']].head(12).to_string(index=False))

    best = {}
    for rank in sorted({r for r, _ in components}):
        group = stats[stats['rank'] == rank]
        plot_rank_bars(rank, group, args.out_dir / f'rank_{rank}_food_effects.png')
        # Plot the strongest positive food association, not simply the largest absolute r.
        top = group.loc[group['point_biserial_r'].idxmax()]
        comp = int(top['component'])
        j = components.index((rank, comp))
        plot_top_component(rank, comp, Z[:, j], y, float(top['point_biserial_r']),
                           float(top['permutation_p_FWER_all_ranks']),
                           args.out_dir / f'rank_{rank}_best_food_component.png')
        # Export high/low examples for optional inspection against *actual* images.
        sortable = images[['label']].copy() if 'label' in images.columns else pd.DataFrame(index=images.index)
        for c in ['moondream_label', 'moondream_image_name', 'matched_food_terms']:
            if c in images.columns:
                sortable[c] = images[c]
        sortable['food_present'] = y
        sortable['stimulus_family'] = families
        sortable['component_activation_z'] = Z[:, j]
        sortable.sort_values('component_activation_z', ascending=False).to_csv(
            args.out_dir / f'rank_{rank}_component_{comp}_images_sorted.csv', index_label='image_key'
        )
        best[str(rank)] = {'component': comp, 'point_biserial_r': float(top['point_biserial_r']),
                           'permutation_p_FWER_all_ranks': float(top['permutation_p_FWER_all_ranks'])}
    summary = {
        'n_unique_images': len(y), 'n_food_images': n_food, 'n_nonfood_images': n_nonfood,
        'ranks': sorted({r for r, _ in components}),
        'n_images_in_within_family_test': int(n_family_images),
        'within_family_families_used': list(families_used),
        'permutations': args.permutations,
        'best_positive_component_by_rank': best,
        'label_warning': 'Food labels are an exploratory keyword proxy until manually reviewed.',
        'inference_unit': 'unique image (averaged across repeated presentations)',
        'correction': 'two-sided max-|r| permutation FWER across all components and ranks',
        'within_family_sensitivity': 'within-family centered correlations, labels shuffled within stimulus families',
    }
    with open(args.out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Plots and summaries: {args.out_dir}')
    print('TIP: Compare overall and within-family effects; a pooled-only effect may reflect stimulus-family confounding.')
    print('NOTE: exploratory associations, not proof of food-selective cortex. '
          'Preprocessing, label accuracy, run confounds, and generalization need review.')


if __name__ == '__main__':
    main()
