
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}-local-data"

SUBJECT = "sub-01"
SESSION = "ses-01"

INPUT_DIR = (
    LOCAL_DATA_ROOT
    / "processed"
    / SUBJECT
    / SESSION
)

OUTPUT_DIR = (
    LOCAL_DATA_ROOT
    / "nmf"
    / SUBJECT
    / SESSION
)


def main():
    input_path = INPUT_DIR / "responses.npy"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Response matrix not found: {input_path}"
        )

    # Memory-map the original extracted response matrix.
    X = np.load(input_path, mmap_mode="r")

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D response matrix, got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Response matrix contains NaN or Inf values."
        )

    print("Input shape (trials, voxels):", X.shape)
    print("Input dtype:", X.dtype)

    # Match the original implementation:
    # subtract each voxel's minimum across trials.
    voxel_minima = np.min(X, axis=0).astype(np.float32)

    # Explicitly allocate a writable array.
    # Output orientation: voxels x trials.
    V = np.asarray(X, dtype=np.float32).T.copy()

    # Apply the voxel-wise baseline shift.
    V -= voxel_minima[:, None]

    # Check nonnegativity.
    min_value = float(V.min())
    if min_value < -1e-5:
        raise ValueError(
            f"Unexpected negative value after shifting: {min_value}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "responses_shifted.npy"
    np.save(output_path, V)

    print("Output shape (voxels, trials):", V.shape)
    print("Output dtype:", V.dtype)
    print("Minimum shifted value:", float(V.min()))
    print("Maximum shifted value:", float(V.max()))
    print("Fraction equal to zero:", float((V == 0).mean()))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
