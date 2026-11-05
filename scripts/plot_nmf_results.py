#!/usr/bin/env python3
"""Generate presentation-ready plots from saved NMF results.

Default project paths match the LAION-fMRI project discussed in chat.
Creates PNGs under nmf_results/plots/; does not refit NMF.

Plots per rank:
  * component activation heatmap (trials x components)
  * z-scored activation heatmap (easier to compare component patterns)
  * distribution of activations for each component
  * correlations between component activation profiles
  * voxel-weight distributions
  * sampled observed-vs-reconstructed voxel responses and trialwise pattern
    correlations, if the preprocessed response matrix is available.

Reminder: reconstruction fit and component correlations do not by themselves
show food selectivity. For that, aligned image labels/scores are needed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pearson_corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=os.path.expanduser(
            "~/85426-sude-berin-local-data/nmf/sub-01/ses-01/nmf_results"
        ),
        help="Directory containing rank_*_trial_activations.npy and rank_*_voxel_components.npy",
    )
    parser.add_argument(
        "--data",
        default=os.path.expanduser(
            "~/85426-sude-berin-local-data/nmf/sub-01/ses-01/responses_shifted.npy"
        ),
        help="Optional original preprocessed matrix (voxels x trials); pass '' to skip reconstruction plots",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--sample-points", type=int, default=25000)
    args = parser.parse_args()

    results_dir = Path(args.results).expanduser()
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data).expanduser() if args.data else None
    X = None
    if data_path and str(data_path) and data_path.exists():
        X = np.load(data_path, mmap_mode="r")
        print(f"Loaded preprocessed matrix: {X.shape}")
    else:
        print("Preprocessed matrix not found/provided; skipping observed-vs-reconstructed plots.")

    rank_fit_summary = []
    for rank in args.ranks:
        h_path = results_dir / f"rank_{rank}_trial_activations.npy"
        w_path = results_dir / f"rank_{rank}_voxel_components.npy"
        if not h_path.exists() or not w_path.exists():
            print(f"Skipping rank {rank}: missing {h_path.name} or {w_path.name}")
            continue

        H = np.load(h_path).astype(np.float64)  # trials x components
        W = np.load(w_path).astype(np.float64)  # components x voxels
        if H.ndim != 2 or W.ndim != 2 or H.shape[1] != rank or W.shape[0] != rank:
            print(f"Skipping rank {rank}: unexpected shapes H={H.shape}, W={W.shape}")
            continue
        print(f"Rank {rank}: activations {H.shape}; voxel components {W.shape}")

        # 1. Activation heatmap in original trial order.
        fig, ax = plt.subplots(figsize=(12, max(4.5, rank * 0.24)))
        im = ax.imshow(H.T, aspect="auto", interpolation="nearest")
        ax.set_xlabel("Trial index (original order)")
        ax.set_ylabel("NMF component")
        ax.set_title(f"Rank {rank}: NMF component activations by trial")
        fig.colorbar(im, ax=ax, label="Activation")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_activation_heatmap.png", dpi=200)
        plt.close(fig)

        # 2. Per-component z-scored heatmap, for visual comparison only.
        sd = H.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (H - H.mean(axis=0, keepdims=True)) / sd
        fig, ax = plt.subplots(figsize=(12, max(4.5, rank * 0.24)))
        im = ax.imshow(Z.T, aspect="auto", interpolation="nearest",
                       cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_xlabel("Trial index (original order)")
        ax.set_ylabel("NMF component")
        ax.set_title(f"Rank {rank}: component activation patterns (z-scores)")
        fig.colorbar(im, ax=ax, label="Activation z-score")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_activation_zscores.png", dpi=200)
        plt.close(fig)

        # 3. Activation distribution by component.
        fig, ax = plt.subplots(figsize=(max(8, rank * 0.45), 5))
        ax.boxplot([H[:, k] for k in range(rank)], showfliers=False)
        ax.set_xlabel("NMF component")
        ax.set_ylabel("Activation")
        ax.set_title(f"Rank {rank}: distribution of component activations")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_activation_distributions.png", dpi=200)
        plt.close(fig)

        # 4. Correlation among activation profiles across the trials.
        corr = np.corrcoef(H, rowvar=False)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xlabel("Component")
        ax.set_ylabel("Component")
        ax.set_title(f"Rank {rank}: correlation between component activation profiles")
        fig.colorbar(im, ax=ax, label="Pearson correlation")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_component_activation_correlations.png", dpi=200)
        plt.close(fig)

        # 5. Voxel-weight distributions by component.
        fig, ax = plt.subplots(figsize=(max(8, rank * 0.45), 5))
        ax.boxplot([W[k] for k in range(rank)], showfliers=False)
        ax.set_xlabel("NMF component")
        ax.set_ylabel("Voxel weight")
        ax.set_title(f"Rank {rank}: distribution of voxel weights")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_voxel_weight_distributions.png", dpi=200)
        plt.close(fig)

        # 6. Compare observed and reconstructed responses if X is available.
        if X is not None:
            if X.shape != (W.shape[1], H.shape[0]):
                print(f"  Cannot reconstruct plot: X shape {X.shape} does not match W/H.")
            else:
                # W is components x voxels, H is trials x components.
                Xhat = W.T @ H.T
                # Sample flattened voxel-trial entries to keep the plot readable.
                rng = np.random.default_rng(42)
                n = X.shape[0] * X.shape[1]
                ix = rng.choice(n, size=min(args.sample_points, n), replace=False)
                # X is memory-mapped; sample via flattened indices without loading all into RAM.
                cols = ix % X.shape[1]
                rows = ix // X.shape[1]
                observed = np.asarray(X[rows, cols], dtype=float)
                reconstructed = Xhat[rows, cols]
                overall_r = pearson_corr(observed, reconstructed)
                maxv = float(max(np.max(observed), np.max(reconstructed)))
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                ax.scatter(observed, reconstructed, s=5, alpha=0.2, rasterized=True)
                ax.plot([0, maxv], [0, maxv], "--", linewidth=1, label="Perfect reconstruction")
                ax.set_xlabel("Observed shifted voxel response")
                ax.set_ylabel("NMF reconstructed response")
                ax.set_title(f"Rank {rank}: observed vs reconstructed\nr = {overall_r:.3f}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(plot_dir / f"rank_{rank}_observed_vs_reconstructed.png", dpi=200)
                plt.close(fig)

                # Correlation of voxel patterns within each trial.
                trial_r = np.array([
                    pearson_corr(np.asarray(X[:, t]), Xhat[:, t])
                    for t in range(X.shape[1])
                ])
                trial_r = trial_r[np.isfinite(trial_r)]
                median_r = float(np.median(trial_r)) if len(trial_r) else float("nan")
                rank_fit_summary.append({
                    "rank": rank,
                    "sampled_observed_reconstructed_pearson_r": overall_r,
                    "median_trialwise_voxel_pattern_pearson_r": median_r,
                    "mean_trialwise_voxel_pattern_pearson_r": float(np.mean(trial_r)) if len(trial_r) else float("nan"),
                })
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.hist(trial_r, bins=35, edgecolor="black", linewidth=0.4)
                ax.axvline(median_r, linestyle="--", label=f"Median r = {median_r:.3f}")
                ax.set_xlabel("Pearson r across voxels within each trial")
                ax.set_ylabel("Number of trials")
                ax.set_title(f"Rank {rank}: observed/reconstructed voxel-pattern similarity")
                ax.legend()
                fig.tight_layout()
                fig.savefig(plot_dir / f"rank_{rank}_trialwise_voxel_pattern_correlations.png", dpi=200)
                plt.close(fig)

    # One compact slide-ready comparison across ranks, if reconstruction plots ran.
    if rank_fit_summary:
        with open(plot_dir / "reconstruction_correlations_summary.json", "w") as f:
            json.dump(rank_fit_summary, f, indent=2)
        ranks = [x["rank"] for x in rank_fit_summary]
        overall = [x["sampled_observed_reconstructed_pearson_r"] for x in rank_fit_summary]
        median = [x["median_trialwise_voxel_pattern_pearson_r"] for x in rank_fit_summary]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ranks, overall, marker="o", label="Sampled voxel-trial values")
        ax.plot(ranks, median, marker="s", label="Median trial-wise voxel pattern")
        ax.set_xlabel("NMF rank")
        ax.set_ylabel("Pearson correlation")
        ax.set_title("NMF reconstruction similarity across ranks")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "rank_comparison_reconstruction_correlations.png", dpi=200)
        plt.close(fig)

    print(f"\nPlots saved to: {plot_dir}")
    print("Reconstruction correlations describe NMF fit to the input data, not food selectivity.")


if __name__ == "__main__":
    main()
