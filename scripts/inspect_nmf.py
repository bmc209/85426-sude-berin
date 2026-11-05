import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser(
    "~/85426-sude-berin-local-data/nmf/sub-01/ses-01/nmf_results"
)
OUT = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)

for rank in (5, 10, 20):
    activations = np.load(
        os.path.join(BASE, f"rank_{rank}_trial_activations.npy")
    )
    components = np.load(
        os.path.join(BASE, f"rank_{rank}_voxel_components.npy")
    )

    print(f"\nRank {rank}")
    print("Activations:", activations.shape)
    print("Components:", components.shape)

    # How strongly each component is activated across trials
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].boxplot(
        [activations[:, k] for k in range(rank)],
        showfliers=False
    )
    axes[0].set_title(f"Rank {rank}: Trial activation distributions")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Activation")

    # Trial activation patterns, with trials in original order
    im = axes[1].imshow(
        activations.T,
        aspect="auto",
        interpolation="nearest"
    )
    axes[1].set_title(f"Rank {rank}: Component activation across trials")
    axes[1].set_xlabel("Trial index")
    axes[1].set_ylabel("Component")
    fig.colorbar(im, ax=axes[1], label="Activation")

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, f"rank_{rank}_activations.png"),
        dpi=150
    )
    plt.close(fig)

    # Inspect the distribution of voxel weights for each component
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        [components[k] for k in range(rank)],
        showfliers=False
    )
    ax.set_title(f"Rank {rank}: Voxel-weight distributions")
    ax.set_xlabel("Component")
    ax.set_ylabel("Voxel weight")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT, f"rank_{rank}_voxel_weights.png"),
        dpi=150
    )
    plt.close(fig)

print(f"\nPlots saved to: {OUT}")
