#!/usr/bin/env python3

"""Memory-conscious NMF baseline for LAION-fMRI responses."""

from pathlib import Path
import json
import time

import numpy as np
from sklearn.decomposition import NMF

ROOT = Path.home() / "85426-sude-berin-local-data"
INPUT = ROOT / "nmf/sub-01/ses-01/responses_shifted.npy"
OUT = ROOT / "nmf/sub-01/ses-01/nmf_results"
OUT.mkdir(parents=True, exist_ok=True)

# Start small to get a baseline before trying larger decompositions.
RANKS = [5, 10, 20]
RANDOM_STATE = 42
MAX_ITER = 500

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT}. Check preprocessing output path."
        )

    # Input is voxels x trials, matching the current preprocessing script.
    X = np.load(INPUT, mmap_mode="r")
    print(f"Input shape: {X.shape}; dtype: {X.dtype}")

    if X.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values.")
    if X.min() < 0:
        raise ValueError("NMF input must be nonnegative.")

    summary = {
        "input_shape_voxels_trials": list(X.shape),
        "input_dtype": str(X.dtype),
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
        "results": [],
        "interpretation_warning": (
            "NMF components are not automatically food-selective. "
            "Food-selectivity requires comparison with image-level food labels "
            "or food scores."
        ),
    }

    for rank in RANKS:
        print(f"\nFitting NMF rank={rank}...")
        start = time.time()

        model = NMF(
            n_components=rank,
            init="nndsvda",
            solver="cd",
            max_iter=MAX_ITER,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )

        # fit_transform returns trial activations because rows are voxels.
        # Transpose the matrix so observations are trials for sklearn.
        W_trials = model.fit_transform(np.asarray(X.T, dtype=np.float32))
        H_components = model.components_

        elapsed = time.time() - start
        prefix = OUT / f"rank_{rank}"

        np.save(f"{prefix}_trial_activations.npy", W_trials.astype(np.float32))
        np.save(f"{prefix}_voxel_components.npy", H_components.astype(np.float32))

        result = {
            "rank": rank,
            "reconstruction_error": float(model.reconstruction_err_),
            "n_iter": int(model.n_iter_),
            "elapsed_seconds": round(elapsed, 2),
            "trial_activations_shape": list(W_trials.shape),
            "voxel_components_shape": list(H_components.shape),
        }
        summary["results"].append(result)
        print(json.dumps(result, indent=2))

        del model, W_trials, H_components

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {OUT}")
    print("Important: reconstruction error measures fit, not food selectivity.")

if __name__ == "__main__":
    main()
