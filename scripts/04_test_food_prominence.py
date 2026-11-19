#!/usr/bin/env python3
"""Optional: correlate HAND-ANNOTATED food prominence with NMF components.

Only food-positive images with a numeric food_prominence score are included.
Use an ordinal 0-3 or 1-5 scale consistently. Correlation is Spearman-like:
rank transform of image-mean component activations and the prominence scores.
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
from food_analysis_common import (
    OUT_DIR, RESULTS_DIR, load_alignment, collect_image_means, permutation_correlations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment', type=Path, default=OUT_DIR / 'trial_food_alignment.csv')
    parser.add_argument('--results', type=Path, default=RESULTS_DIR)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR / 'prominence')
    parser.add_argument('--ranks', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--permutations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = load_alignment(args.alignment)
    if 'food_prominence' not in df.columns:
        raise SystemExit('No food_prominence column. Add manual scores to food_label_review.csv and rerun 02.')
    images, Xall, components = collect_image_means(df, args.results, args.ranks)
    scores = pd.to_numeric(images['food_prominence'], errors='coerce')
    mask = (images['food_present'] == 1) & scores.notna()
    if mask.sum() < 15 or scores[mask].nunique() < 3:
        raise SystemExit(
            f'Only {int(mask.sum())} food images have a numeric prominence score '
            f'({scores[mask].nunique()} distinct values). '
            'Hand-label at least 15 food images across at least 3 prominence levels, '
            'then rerun scripts 02 and 04.'
        )
    images = images.loc[mask]
    X = Xall[mask.to_numpy()]
    prom = scores[mask].to_numpy(dtype=float)
    # Spearman: apply average ranks, including ties, separately to each variable.
    ranks_X = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    ranks_y = rankdata(prom)
    corr, p_raw, p_adj = permutation_correlations(ranks_X, ranks_y, args.permutations, args.seed)
    stats = pd.DataFrame([
        {'rank': rank, 'component': comp, 'spearman_rho': float(corr[j]),
         'permutation_p_uncorrected': float(p_raw[j]),
         'permutation_p_FWER_all_ranks': float(p_adj[j]),
         'n_food_images_with_prominence': len(prom)}
        for j, (rank, comp) in enumerate(components)
    ]).sort_values(['permutation_p_FWER_all_ranks', 'permutation_p_uncorrected'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.out_dir / 'component_prominence_correlations.csv', index=False)
    for rank in sorted({r for r, _ in components}):
        group = stats[stats['rank'] == rank]
        top = group.loc[group['spearman_rho'].idxmax()]
        comp = int(top['component'])
        j = components.index((rank, comp))
        fig, ax = plt.subplots(figsize=(7, 5))
        rng = np.random.default_rng(args.seed)
        ax.scatter(prom + rng.normal(0, 0.06, len(prom)), X[:, j], alpha=0.5, s=25)
        # Per-score medians are robust and help explain ordered categories.
        for level in sorted(np.unique(prom)):
            ax.scatter([level], [np.median(X[prom == level, j])], s=80,
                       marker='_', linewidths=3, color='black')
        ax.set_xlabel('Hand-annotated food prominence score')
        ax.set_ylabel('Image-mean NMF activation (raw component units)')
        ax.set_title(f'Rank {rank}, component {comp}: prominence\n'
                     f'Spearman rho = {top.spearman_rho:.3f}; maxT p = {top.permutation_p_FWER_all_ranks:.4f}')
        fig.tight_layout()
        fig.savefig(args.out_dir / f'rank_{rank}_best_prominence_component.png', dpi=180)
        plt.close(fig)
    with open(args.out_dir / 'summary.json', 'w') as f:
        json.dump({'n_unique_food_images_scored': len(prom), 'prominence_scale_min': float(prom.min()),
                   'prominence_scale_max': float(prom.max()),
                   'note': 'Observational correlation; do not interpret as causal evidence.'}, f, indent=2)
    print(f'Analyzed {len(prom)} uniquely named food images with prominence ratings.')
    print(stats.head(10).to_string(index=False))
    print(f'Wrote prominence results to {args.out_dir}')


if __name__ == '__main__':
    main()
