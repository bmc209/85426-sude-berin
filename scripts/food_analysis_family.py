"""Stimulus-family sensitivity analysis for NMF/food associations.

Family-adjusted correlations remove between-family mean differences, then
permute food labels *within* stimulus families at the unique-image level.
This is exploratory, not a substitute for independent replication or run QC.
"""
import numpy as np
import pandas as pd


def stimulus_family(image_name):
    s = str(image_name).lower()
    for needle, name in (("_laion_", "LAION"), ("_things+_", "THINGS+"),
                         ("_nsd_", "NSD"), ("_ood_", "OOD")):
        if needle in s:
            return name
    return "OTHER"


def within_family_permutation(X, y, families, n_permutations=5000, seed=42):
    """Family-adjusted partial r and two-sided permutation p-values.

    Returns (r, p_uncorrected, p_FWER, number_images, families_used).
    Families without both food/nonfood contribute no within-family contrast;
    exclude them to prevent a large all-nonfood family from dominating scale.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    families = np.asarray(families, dtype=str)
    if X.shape[0] != len(y) or len(y) != len(families):
        raise ValueError("X, y, families must have aligned image rows")
    unique = np.unique(families)
    eligible = [f for f in unique if np.unique(y[families == f]).size == 2]
    keep = np.isin(families, eligible)
    if not eligible or (y[keep] == 1).sum() < 5 or (y[keep] == 0).sum() < 5:
        return None
    X, y, families = X[keep], y[keep], families[keep]
    subgroups = [np.flatnonzero(families == f) for f in eligible]
    Xc = X.copy()
    yc = y.copy()
    for ix in subgroups:
        Xc[ix] -= Xc[ix].mean(axis=0)
        yc[ix] -= yc[ix].mean()
    denx = np.sqrt(np.sum(Xc * Xc, axis=0))
    deny = float(np.sqrt(np.sum(yc * yc)))
    valid = denx > 1e-12
    ZX = np.zeros_like(Xc)
    ZX[:, valid] = Xc[:, valid] / denx[valid]
    Zy = yc / deny
    obs = ZX.T @ Zy
    counts = np.ones(X.shape[1], dtype=int)
    family_counts = np.ones(X.shape[1], dtype=int)
    magnitudes = np.abs(obs)
    rng = np.random.default_rng(seed)
    for _ in range(n_permutations):
        yp = Zy.copy()
        for ix in subgroups:
            yp[ix] = rng.permutation(yp[ix])
        rp = np.abs(ZX.T @ yp)
        counts += rp >= magnitudes - 1e-12
        family_counts += np.max(rp) >= magnitudes - 1e-12
    obs[~valid] = 0.0
    p = counts / (n_permutations + 1)
    pf = family_counts / (n_permutations + 1)
    p[~valid] = 1.0
    pf[~valid] = 1.0
    return obs, p, pf, len(y), eligible
