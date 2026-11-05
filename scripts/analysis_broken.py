from pathlib import Path

analysis_script = r'''#!/usr/bin/env python3
"""
Analyze saved NMF fits for the LAION-fMRI project.

Default inputs:
  ~/85426-sude-berin-local-data/nmf/sub-01/ses-01/responses_shifted.npy
  ~/85426-sude-berin-local-data/nmf/sub-01/ses-01/nmf_results/rank_{k}_*.npy

The script evaluates reconstruction quality and summarizes component activations.
It does NOT infer food selectivity unless an aligned score file is provided.

Optional score file:
  CSV with columns: trial_index, food_score
  trial_index must be the 0-based row index in rank_{k}_trial_activations.npy.
  food_score may be binary (food/non-food) or continuous.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def corr_safe(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def load_scores(path, n_trials):
    if path is None:
        return None, None
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"trial_index", "food_score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Score file must have columns {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            idx = int(row["trial_index"])
            score = float(row["food_score"])
            if 0 <= idx < n_trials and np.isfinite(score):
                rows.append((idx, score))
    if not rows:
        raise ValueError("No valid score rows found in score file.")
    indices = np.array([r[0] for r in rows], dtype=int)
    values = np.array([r[1] for r in rows], dtype=float)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Score file has duplicate trial_index values.")
    return indices, values


def permutation_pvalue(x, y, n_perm=5000, seed=42):
    """Two-sided permutation p-value for Spearman correlation."""
    observed = float(spearmanr(x, y).statistic)
    if not np.isfinite(observed):
        return float("nan"), observed
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm_r = spearmanr(x, rng.permutation(y)).statistic
        if np.isfinite(perm_r) and abs(perm_r) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1), observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.expanduser(
            "~/85426-sude-berin-local-data/nmf/sub-01/ses-01/responses_shifted.npy"
        ),
        help="Preprocessed matrix, shape voxels x trials",
    )
    parser.add_argument(
        "--results",
        default=os.path.expanduser(
            "~/85426-sude-berin-local-data/nmf/sub-01/ses-01/nmf_results"
        ),
        help="Directory containing rank_* NPY files",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--scores",
        default=None,
        help="Optional CSV with trial_index,food_score columns",
    )
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    data_path = Path(args.data)
    results_dir = Path(args.results)
    out_dir = results_dir / "analysis"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D input matrix, got shape {X.shape}")
    n_voxels, n_trials = X.shape
    print(f"Loaded X: shape={X.shape}, dtype={X.dtype}")

    score_indices, score_values = load_scores(args.scores, n_trials) if args.scores else (None, None)
    all_summaries = []

    for rank in args.ranks:
        w_path = results_dir / f"rank_{rank}_voxel_components.npy"
        h_path = results_dir / f"rank_{rank}_trial_activations.npy"
        if not w_path.exists() or not h_path.exists():
            print(f"Skipping rank {rank}: missing {w_path.name} or {h_path.name}")
            continue

        # Saved W has shape components x voxels; saved H has shape trials x components.
        W = np.load(w_path)
        H = np.load(h_path)
        if W.shape != (rank, n_voxels):
            raise ValueError(f"Rank {rank}: unexpected W shape {W.shape}")
        if H.shape != (n_trials, rank):
            raise ValueError(f"Rank {rank}: unexpected H shape {H.shape}")

        # NMF reconstruction. This is the appropriate comparison to the input data.
        Xhat = W.T @ H.T
        # Metrics use float64 to improve numerical stability.
        X64 = np.asarray(X, dtype=np.float64)
        Xhat64 = Xhat.astype(np.float64, copy=False)
        residual = X64 - Xhat64
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        x_mean = float(np.mean(X64))
        total_ss = float(np.sum((X64 - x_mean) ** 2))
        residual_ss = float(np.sum(residual ** 2))
        variance_explained = float(1 - residual_ss / total_ss) if total_ss > 0 else float("nan")
        flattened_corr = corr_safe(X64, Xhat64)

        # Per-trial correlation across voxels, and per-voxel correlation across trials.
        trial_corrs = np.array([corr_safe(X64[:, t], Xhat64[:, t]) for t in range(n_trials)])
        voxel_corrs = np.array([corr_safe(X64[v, :], Xhat64[v, :]) for v in range(n_voxels)])
        trial_corrs = trial_corrs[np.isfinite(trial_corrs)]
        voxel_corrs = voxel_corrs[np.isfinite(voxel_corrs)]

        # Normalize each W row to unit L2 norm and transfer scale into H.
        # This leaves W.T @ H.T unchanged and makes component weight summaries comparable.
        W_norm = W.astype(np.float64, copy=True)
        H_norm = H.astype(np.float64, copy=True)
        norms = np.linalg.norm(W_norm, axis=1)
        norms[norms == 0] = 1.0
        W_norm /= norms[:, None]
        H_norm *= norms[None, :]

        component_rows = []
        for k in range(rank):
            row = {
                "rank": rank,
                "component": k,
                "mean_activation": float(np.mean(H_norm[:, k])),
                "median_activation": float(np.median(H_norm[:, k])),
                "std_activation": float(np.std(H_norm[:, k])),
                "max_activation": float(np.max(H_norm[:, k])),
                "mean_normalized_voxel_weight": float(np.mean(W_norm[k])),
                "max_normalized_voxel_weight": float(np.max(W_norm[k])),
                "top_1pct_activation_trial_count": int(
                    np.sum(H_norm[:, k] >= np.quantile(H_norm[:, k], 0.99))
                ),
            }
            if score_indices is not None:
                x = H_norm[score_indices, k]
                y = score_values
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 5 and np.unique(y[valid]).size > 1:
                    pval, rho = permutation_pvalue(
                        x[valid], y[valid], n_perm=args.permutations, seed=42 + k
                    )
                    row["food_score_spearman_rho"] = float(rho)
                    row["food_score_permutation_p"] = float(pval)
                    row["n_scored_trials"] = int(valid.sum())
                else:
                    row["food_score_spearman_rho"] = float("nan")
                    row["food_score_permutation_p"] = float("nan")
                    row["n_scored_trials"] = int(valid.sum())
            component_rows.append(row)
            all_summaries.append(row)

        rank_summary = {
            "rank": rank,
            "n_voxels": n_voxels,
            "n_trials": n_trials,
            "rmse": rmse,
            "flattened_pearson_r_original_vs_reconstruction": flattened_corr,
            "variance_explained_1_minus_sse_over_total_ss": variance_explained,
            "median_per_trial_pearson_r_across_voxels": float(np.median(trial_corrs)) if len(trial_corrs) else None,
            "mean_per_trial_pearson_r_across_voxels": float(np.mean(trial_corrs)) if len(trial_corrs) else None,
            "median_per_voxel_pearson_r_across_trials": float(np.median(voxel_corrs)) if len(voxel_corrs) else None,
            "mean_per_voxel_pearson_r_across_trials": float(np.mean(voxel_corrs)) if len(voxel_corrs) else None,
            "score_file_used": args.scores,
            "n_scored_trials": int(len(score_indices)) if score_indices is not None else 0,
        }

        with open(out_dir / f"rank_{rank}_metrics.json", "w") as f:
            json.dump(rank_summary, f, indent=2)
        with open(out_dir / f"rank_{rank}_component_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
            writer.writeheader()
            writer.writerows(component_rows)

        # Plot 1: observed vs reconstructed sampled values.
        rng = np.random.default_rng(42)
        total = n_voxels * n_trials
        sample_idx = rng.choice(total, size=min(30000, total), replace=False)
        # Flattened indexing corresponds to a row-major view of the original X.
        obs_sample = np.asarray(X).ravel()[sample_idx]
        recon_sample = Xhat.ravel()[sample_idx]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(obs_sample, recon_sample, s=5, alpha=0.18, rasterized=True)
        lim_max = float(max(np.max(obs_sample), np.max(recon_sample)))
        ax.plot([0, lim_max], [0, lim_max], linestyle="--", linewidth=1)
        ax.set_xlabel("Observed shifted voxel response")
        ax.set_ylabel("NMF reconstructed response")
        ax.set_title(f"Rank {rank}: observed vs reconstructed\nr = {flattened_corr:.3f}; RMSE = {rmse:.3f}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_reconstruction_scatter.png", dpi=180)
        plt.close(fig)

        # Plot 2: distribution of per-trial correlations.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(trial_corrs, bins=35, edgecolor="black", linewidth=0.4)
        ax.axvline(np.median(trial_corrs), linestyle="--",
                   label=f"Median = {np.median(trial_corrs):.3f}")
        ax.set_xlabel("Pearson r across voxels (observed vs reconstructed)")
        ax.set_ylabel("Number of trials")
        ax.set_title(f"Rank {rank}: reconstruction correlation per trial")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_trial_reconstruction_correlations.png", dpi=180)
        plt.close(fig)

        # Plot 3: normalized activation distributions per component.
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.boxplot([H_norm[:, k] for k in range(rank)], showfliers=False)
        ax.set_xlabel("NMF component")
        ax.set_ylabel("Activation (W rows L2-normalized)")
        ax.set_title(f"Rank {rank}: normalized component activation distributions")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_normalized_activations.png", dpi=180)
        plt.close(fig)

        # Plot 4: activation heatmap in original trial order.
        fig, ax = plt.subplots(figsize=(12, max(4, rank * 0.24)))
        im = ax.imshow(H_norm.T, aspect="auto", interpolation="nearest")
        ax.set_xlabel("Trial index (original order)")
        ax.set_ylabel("NMF component")
        ax.set_title(f"Rank {rank}: normalized component activations by trial")
        fig.colorbar(im, ax=ax, label="Activation")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_activation_heatmap.png", dpi=180)
        plt.close(fig)

        # Plot 5: component-to-component activation correlation.
        corr_matrix = np.corrcoef(H_norm, rowvar=False)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xlabel("Component")
        ax.set_ylabel("Component")
        ax.set_title(f"Rank {rank}: correlation between component activations")
        fig.colorbar(im, ax=ax, label="Pearson r")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_component_activation_correlations.png", dpi=180)
        plt.close(fig)

        # If scores are supplied, plot activation vs food score for every component.
        if score_indices is not None:
            fig, axes = plt.subplots(
                nrows=int(np.ceil(rank / 4)), ncols=4,
                figsize=(14, max(3, int(np.ceil(rank / 4)) * 2.7)),
                squeeze=False
            )
            for k, ax in enumerate(axes.ravel()):
                if k >= rank:
                    ax.axis("off")
                    continue
                ax.scatter(score_values, H_norm[score_indices, k], s=12, alpha=0.5)
                ax.set_title(f"Comp {k}")
                ax.set_xlabel("Food score")
                ax.set_ylabel("Activation")
            fig.suptitle(f"Rank {rank}: component activation vs food score", y=1.01)
            fig.tight_layout()
            fig.savefig(plot_dir / f"rank_{rank}_food_score_scatterplots.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        print(json.dumps(rank_summary, indent=2))
        print(f"Rank {rank} plots and summaries saved under {out_dir}")

    if all_summaries:
        fields = list(all_summaries[0].keys())
        with open(out_dir / "all_component_summaries.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_summaries)

    print(f"\nDone. Outputs: {out_dir}")
    print("Interpretation: reconstruction metrics evaluate how well NMF approximates X; they do not establish food selectivity.")
    if args.scores is None:
        print("No food-score file supplied, so food-selectivity tests were not run.")


if __name__ == "__main__":
    main()
'''

plot_script = r'''#!/usr/bin/env python3
"""
Create slide-ready plots from the analysis outputs and saved NMF factors.

Run after analyze_nmf.py. It creates a compact comparison figure across ranks
and a component activation correlation plot for each rank.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=os.path.expanduser(
            "~/85426-sude-berin-local-data/nmf/sub-01/ses-01/nmf_results"
        ),
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()

    base = Path(args.results)
    analysis_dir = base / "analysis"
    plot_dir = analysis_dir / "slide_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for rank in args.ranks:
        p = analysis_dir / f"rank_{rank}_metrics.json"
        if p.exists():
            with open(p) as f:
                metrics.append(json.load(f))
    if metrics:
        ranks = [m["rank"] for m in metrics]
        corr = [m["flattened_pearson_r_original_vs_reconstruction"] for m in metrics]
        rmse = [m["rmse"] for m in metrics]
        variance = [m["variance_explained_1_minus_sse_over_total_ss"] for m in metrics]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(ranks, corr, marker="o")
        axes[0].set_title("Original vs reconstructed")
        axes[0].set_xlabel("NMF rank")
        axes[0].set_ylabel("Pearson r")

        axes[1].plot(ranks, rmse, marker="o")
        axes[1].set_title("Reconstruction error")
        axes[1].set_xlabel("NMF rank")
        axes[1].set_ylabel("RMSE (lower is better)")

        axes[2].plot(ranks, variance, marker="o")
        axes[2].set_title("Variance explained")
        axes[2].set_xlabel("NMF rank")
        axes[2].set_ylabel("1 − SSE / total SS")

        fig.suptitle("NMF reconstruction diagnostics (not a food-selectivity test)")
        fig.tight_layout()
        fig.savefig(plot_dir / "rank_comparison_reconstruction.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plot the activation heatmap for the highest requested rank, if available.
    for rank in args.ranks:
        p = base / f"rank_{rank}_trial_activations.npy"
        if not p.exists():
            continue
        H = np.load(p)
        # Normalize each component by its standard deviation for visibility only.
        # This does not change saved data and is not used for statistical testing.
        mean = H.mean(axis=0, keepdims=True)
        std = H.std(axis=0, keepdims=True)
        std[std == 0] = 1
        Z = (H - mean) / std
        fig, ax = plt.subplots(figsize=(12, max(4, rank * 0.23)))
        im = ax.imshow(Z.T, aspect="auto", interpolation="nearest",
                       cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_xlabel("Trial index (original order)")
        ax.set_ylabel("Component")
        ax.set_title(f"Rank {rank}: component activation z-scores by trial")
        fig.colorbar(im, ax=ax, label="Activation z-score")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_activation_zscore_heatmap.png", dpi=200)
        plt.close(fig)

        # Component correlation matrix
        corrmat = np.corrcoef(H, rowvar=False)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corrmat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"Rank {rank}: correlation among component activations")
        ax.set_xlabel("Component")
        ax.set_ylabel("Component")
        fig.colorbar(im, ax=ax, label="Pearson r")
        fig.tight_layout()
        fig.savefig(plot_dir / f"rank_{rank}_activation_correlation_matrix.png", dpi=200)
        plt.close(fig)

    print(f"Slide-ready plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
'''

p1 = Path("/mnt/data/analyze_nmf.py")
p2 = Path("/mnt/data/plot_nmf_results.py")
p1.write_text(analysis_script)
p2.write_text(plot_script)
print(f"Created:\n{p1} ({p1.stat().st_size:,} bytes)\n{p2} ({p2.stat().st_size:,} bytes)")
