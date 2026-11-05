#!/usr/bin/env python3

"""
Memory-safe single-trial fMRI response extraction.

Reads a compressed 4D NIfTI sequentially, one volume at a time.
Extracts voxels inside an ROI and saves a trials x voxels matrix.

Outputs are saved OUTSIDE the Git repository.
"""

from pathlib import Path
import gzip

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SUBJECT = "sub-01"
SESSION = "ses-01"

DATA_ROOT = Path.home() / "shared" / "laion-fmri"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = (
    PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}-local-data"
)

BETA_FILE = (
    DATA_ROOT
    / "derivatives"
    / "glmsingle-tedana"
    / SUBJECT
    / SESSION
    / "func"
    / (
        f"{SUBJECT}_{SESSION}_task-images_space-T1w_stat-effect_"
        "desc-SingletrialBetas_statmap.nii.gz"
    )
)

TRIALS_FILE = (
    DATA_ROOT
    / "derivatives"
    / "glmsingle-tedana"
    / SUBJECT
    / SESSION
    / "func"
    / f"{SUBJECT}_{SESSION}_task-images_desc-SingletrialBetas_trials.tsv"
)

MASK_FILE = (
    DATA_ROOT
    / "derivatives"
    / "rois"
    / SUBJECT
    / "laion"
    / f"{SUBJECT}_space-T1w_res-1pt8_label-laiongeneral_mask.nii.gz"
)

OUT_DIR = (
    LOCAL_DATA_ROOT
    / "processed"
    / SUBJECT
    / SESSION
)


# ============================================================
# Helpers
# ============================================================

def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def main() -> None:
    print("=" * 72)
    print("Memory-safe fMRI response extraction")
    print("=" * 72)
    print(f"Subject: {SUBJECT}")
    print(f"Session: {SESSION}")
    print(f"Output:  {OUT_DIR}")
    print()

    # --------------------------------------------------------
    # 1. Check inputs
    # --------------------------------------------------------

    require_file(BETA_FILE, "Beta NIfTI")
    require_file(TRIALS_FILE, "Trial metadata TSV")
    require_file(MASK_FILE, "ROI mask")

    # --------------------------------------------------------
    # 2. Load headers and metadata
    # --------------------------------------------------------

    print("Reading beta image header...")
    beta_img = nib.load(str(BETA_FILE))

    print("Reading ROI mask...")
    mask_img = nib.load(str(MASK_FILE))

    print("Reading trial metadata...")
    trials = pd.read_csv(TRIALS_FILE, sep="\t")

    if len(beta_img.shape) != 4:
        raise ValueError(
            f"Expected a 4D beta image; got {beta_img.shape}"
        )

    if len(mask_img.shape) != 3:
        raise ValueError(
            f"Expected a 3D ROI mask; got {mask_img.shape}"
        )

    spatial_shape = beta_img.shape[:3]
    n_volumes = beta_img.shape[3]
    n_spatial_voxels = int(np.prod(spatial_shape))

    print(f"Beta image shape: {beta_img.shape}")
    print(f"Mask shape:       {mask_img.shape}")
    print(f"Metadata rows:    {len(trials)}")

    # --------------------------------------------------------
    # 3. Validate metadata and volume ordering
    # --------------------------------------------------------

    if len(trials) != n_volumes:
        raise ValueError(
            f"Metadata contains {len(trials)} rows but beta image "
            f"contains {n_volumes} volumes."
        )

    required_columns = {"run", "beta_index"}
    missing_columns = required_columns - set(trials.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_columns)}"
        )

    if trials["run"].isna().any():
        raise ValueError("Missing run values.")

    if trials["beta_index"].isna().any():
        raise ValueError("Missing beta_index values.")

    beta_indices = pd.to_numeric(
        trials["beta_index"], errors="raise"
    )

    if not np.all(beta_indices == np.floor(beta_indices)):
        raise ValueError("beta_index contains non-integer values.")

    trials["beta_index"] = beta_indices.astype(int)

    # beta_index resets within each run, so validate the pair.
    if trials.duplicated(["run", "beta_index"]).any():
        duplicates = trials.loc[
            trials.duplicated(
                ["run", "beta_index"], keep=False
            ),
            ["run", "beta_index"],
        ]
        raise ValueError(
            "Duplicate (run, beta_index) pairs:\n"
            f"{duplicates.head(20)}"
        )

    # Each run should have indices 0, 1, ..., N-1.
    for run, group in trials.groupby("run", sort=False):
        indices = sorted(group["beta_index"].tolist())
        expected = list(range(len(group)))

        if indices != expected:
            raise ValueError(
                f"Unexpected beta_index sequence for run {run}. "
                f"Expected 0..{len(group) - 1}, "
                f"got {indices[:10]}..."
            )

    # Do not sort the table: its existing row order maps to the
    # NIfTI volume order according to the dataset sidecar.
    trials = trials.reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Validate mask alignment
    # --------------------------------------------------------

    if spatial_shape != mask_img.shape:
        raise ValueError(
            f"Spatial shape mismatch: beta={spatial_shape}, "
            f"mask={mask_img.shape}"
        )

    if not np.allclose(
        beta_img.affine, mask_img.affine, atol=1e-4
    ):
        raise ValueError(
            "Beta image and ROI mask have different affines."
        )

    mask_data = np.asarray(mask_img.dataobj)
    mask = np.isfinite(mask_data) & (mask_data != 0)
    del mask_data

    n_voxels = int(mask.sum())

    if n_voxels == 0:
        raise ValueError("ROI mask contains no valid voxels.")

    print(f"Number of trials: {n_volumes}")
    print(f"Number of runs:   {trials['run'].nunique()}")
    print(f"Number of voxels: {n_voxels}")
    print()

    # --------------------------------------------------------
    # 5. Prepare sequential reading of compressed NIfTI
    # --------------------------------------------------------
    #
    # NIfTI stores each 3D volume contiguously, with the first
    # spatial axis varying fastest. We read the gzip stream
    # sequentially to avoid repeatedly decompressing earlier data.
    #
    # Nibabel's proxy supplies the on-disk dtype, data offset,
    # and scaling parameters so values are converted consistently.
    # --------------------------------------------------------

    proxy = beta_img.dataobj
    disk_dtype = np.dtype(beta_img.get_data_dtype())
    data_offset = int(proxy.offset)

    slope = float(proxy.slope)
    intercept = float(proxy.inter)

    bytes_per_volume = (
        n_spatial_voxels * disk_dtype.itemsize
    )

    print(f"Stored data type: {disk_dtype}")
    print(f"Data offset:      {data_offset} bytes")
    print(f"Scaling slope:    {slope}")
    print(f"Scaling intercept:{intercept}")
    print()
    print("Reading compressed beta data sequentially...")

    # The response matrix is relatively small compared with the
    # complete 4D image: about 48 MiB for 1044 x 12020 float32.
    X = np.empty(
        (n_volumes, n_voxels),
        dtype=np.float32,
    )

    with gzip.open(BETA_FILE, "rb") as f:
        # Skip the NIfTI header and extension area.
        header_bytes = f.read(data_offset)

        if len(header_bytes) != data_offset:
            raise IOError(
                "Could not read the complete NIfTI header/offset."
            )

        for volume_index in range(n_volumes):
            raw = f.read(bytes_per_volume)

            if len(raw) != bytes_per_volume:
                raise IOError(
                    f"Incomplete data at volume {volume_index}: "
                    f"expected {bytes_per_volume} bytes, "
                    f"received {len(raw)}."
                )

            # NIfTI stores the first array dimension fastest.
            volume = np.frombuffer(
                raw,
                dtype=disk_dtype,
                count=n_spatial_voxels,
            ).reshape(spatial_shape, order="F")

            # Convert stored values to scaled beta values.
            # Copy into float32 so later operations do not modify
            # the buffer owned by the gzip read.
            volume = volume.astype(np.float32)

            if slope != 1.0:
                volume *= slope

            if intercept != 0.0:
                volume += intercept

            # Boolean indexing extracts only ROI voxels.
            X[volume_index, :] = volume[mask]

            if (
                (volume_index + 1) % 25 == 0
                or volume_index + 1 == n_volumes
            ):
                print(
                    f"Processed {volume_index + 1}/"
                    f"{n_volumes} trials"
                )

            del raw, volume

    expected_shape = (n_volumes, n_voxels)

    if X.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected response shape {X.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # 6. Build aligned metadata
    # --------------------------------------------------------

    metadata = trials.copy()
    metadata.insert(0, "volume_index", np.arange(n_volumes))

    # --------------------------------------------------------
    # 7. Quality-control statistics
    # --------------------------------------------------------

    finite = np.isfinite(X)
    n_total = X.size
    n_finite = int(finite.sum())
    n_nonfinite = n_total - n_finite

    percentile_levels = [0, 1, 5, 25, 50, 75, 95, 99, 100]

    if n_finite:
        finite_values = X[finite]

        percentiles = np.percentile(
            finite_values, percentile_levels
        )

        negative_fraction = float(
            np.mean(finite_values < 0)
        )

        zero_fraction = float(
            np.mean(finite_values == 0)
        )
    else:
        percentiles = np.full(
            len(percentile_levels), np.nan
        )
        negative_fraction = float("nan")
        zero_fraction = float("nan")

    print()
    print("-" * 60)
    print("RESPONSE MATRIX QUALITY CHECK")
    print("-" * 60)
    print(f"Shape:                   {X.shape}")
    print(f"Dtype:                   {X.dtype}")
    print(f"Finite values:           {n_finite:,}/{n_total:,}")
    print(f"NaN/Inf values:          {n_nonfinite:,}")
    print(f"Negative-value fraction: {negative_fraction:.6f}")
    print(f"Zero-value fraction:     {zero_fraction:.6f}")
    print()
    print("Percentiles of finite beta values:")

    for level, value in zip(
        percentile_levels, percentiles
    ):
        print(f"  {level:>3}%: {value:.8g}")

    # --------------------------------------------------------
    # 8. Save outside the Git repository
    # --------------------------------------------------------

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    responses_path = OUT_DIR / "responses.npy"
    metadata_path = OUT_DIR / "trial_metadata.tsv"
    summary_path = OUT_DIR / "extraction_summary.txt"

    print()
    print(f"Saving responses: {responses_path}")
    np.save(responses_path, X)

    print(f"Saving metadata:  {metadata_path}")
    metadata.to_csv(
        metadata_path, sep="\t", index=False
    )

    print(f"Saving summary:   {summary_path}")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Subject: {SUBJECT}\n")
        f.write(f"Session: {SESSION}\n")
        f.write(f"Beta image: {BETA_FILE}\n")
        f.write(f"Trial metadata: {TRIALS_FILE}\n")
        f.write(f"ROI mask: {MASK_FILE}\n")
        f.write(f"Output directory: {OUT_DIR}\n")
        f.write(f"Response matrix shape: {X.shape}\n")
        f.write(f"Response matrix dtype: {X.dtype}\n")
        f.write(f"Number of trials: {n_volumes}\n")
        f.write(f"Number of runs: {trials['run'].nunique()}\n")
        f.write(f"Number of ROI voxels: {n_voxels}\n")
        f.write(f"Finite values: {n_finite}/{n_total}\n")
        f.write(f"NaN/Inf values: {n_nonfinite}\n")
        f.write(
            f"Negative-value fraction: "
            f"{negative_fraction:.8g}\n"
        )
        f.write(f"Zero-value fraction: {zero_fraction:.8g}\n")
        f.write("\nPercentiles of finite beta values:\n")

        for level, value in zip(
            percentile_levels, percentiles
        ):
            f.write(f"{level}%: {value:.8g}\n")

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Responses: {responses_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
