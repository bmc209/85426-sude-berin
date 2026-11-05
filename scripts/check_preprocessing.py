import numpy as np

path = "/home/jovyan/85426-sude-berin-local-data/nmf/sub-01/ses-01/responses_shifted.npy"
X = np.load(path)

print("Shape:", X.shape)
print("Dtype:", X.dtype)
print("All finite:", np.isfinite(X).all())
print("Minimum:", X.min())
print("Any negative values:", np.any(X < 0))
print("Zero fraction:", np.mean(X == 0))