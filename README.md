# Voxel-wise regression: real-world food size in LAION-fMRI

## Research question
Among food-containing images, do individual ventral-visual-cortex voxels change their fMRI responses with the typical **physical size in real life** of the depicted food (1 = grain of rice; 10 = watermelon)? This is NOT the area occupied within the photograph.

## Inputs
- `analysis/khosla_replication/sub-01/ses-01/X_image_by_voxel.npy`: 769-image x 3767-voxel preprocessing matrix.
- `analysis/khosla_replication/sub-01/ses-01/image_order.csv`: the exact image ordering for the matrix.
- `analysis/food_selectivity/fast_labels/05_food_realworld_size.csv`: output of script `12_clip_realworld_food_size.py`, with `image_key`, `food_present`, `size_1to10`, `size_source`.
- sub-01 native-T1w `laionventral` ROI NIfTI mask (same one used to extract the original voxels).

## Run

```bash
cd ~/85426-sude-berin
cp 15_voxelwise_food_size_regression.py scripts/
OUT=analysis/khosla_replication/sub-01/ses-01
FAST=analysis/food_selectivity/fast_labels
MASK=~/shared/laion-fmri/derivatives/rois/sub-01/laion/sub-01_space-T1w_res-1pt8_label-laionventral_mask.nii.gz
python scripts/15_voxelwise_food_size_regression.py \
  --input-dir "$OUT" \
  --labels "$FAST/05_food_realworld_size.csv" \
  --mask "$MASK" \
  --permutations 5000
```

For a very quick exploratory first pass, use `--permutations 200`; for a more stable final max-T adjusted p-value use `--permutations 5000` or `10000`. To restrict to manually confirmed ratings, use `--reviewed-only` (requires at least 12 manually rated food images). `--no-family-adjustment` fits only intercept+size, but the default controls for categorical stimulus family. If you only want CSV results without brain volumes, omit `--mask`.

## Model
For each voxel v and food image i:

`response[i,v] = intercept[v] + beta_size[v] * real_world_food_size[i] + sum_f gamma_f[v] * family_indicator[i,f] + error[i,v]`

The full model includes stimulus-family dummy variables by default. OLS beta quantifies change in voxel response per one unit increase in rated food size. The t/p values use the usual OLS assumptions. To mitigate testing ~3767 voxels, it computes Benjamini-Hochberg FDR and optionally Freedman-Lane nuisance-adjusted max-|t| permutation FWER across the ROI. The permutation scheme is restricted within stimulus family when that adjustment is active. This remains exploratory and rests on image-level exchangeability assumptions.

## Outputs
Under `OUT/voxelwise_food_size_regression/`:

- `voxelwise_food_size_regression.csv` (one row per voxel: beta, SE, t, raw p, BH-FDR q, max-T FWER p, partial r)
- `included_food_images.csv` (the exact food images included)
- `summary.json` (sample counts, label status, significant voxel counts)
- `voxel_size_regression_distributions.png` (histograms)
- Native-T1w NIfTI maps: `size_beta_per_point_nativeT1w.nii.gz`, `size_t_statistic_nativeT1w.nii.gz`, `size_fdr_significant_beta_nativeT1w.nii.gz`, `size_fwer_significant_beta_nativeT1w.nii.gz` and corrected-p maps.
- `size_tstat_brain_slices.png` when nilearn is available.

This is a **native volume** map, not a cortical mesh. The map **must not** be overlaid directly on MNI152 or fsaverage. The script checks the number of mask voxels; still confirm that the original extraction flattened voxels in NumPy C order using `mask > 0`. A matching count alone cannot guarantee correct ordering.

## Interpretation / limitations
- This is a food-only size analysis; it does *not* measure food-vs-nonfood selectivity.
- The image-size ratings are derived from CLIP food-item identity and an approximate object-size dictionary, *not* direct physical measurements, and must be manually verified where possible.
- Very few currently food-labeled images (~27) makes the regression noisy; corrected significance may be absent even if real effects exist.
- Neural response matrix was minimum-shifted to nonnegative for NMF: per-voxel constant offsets do not change slopes or t statistics, but its units are not raw BOLD percent signal change.
- Each unique image appears only once after averaging repeated presentations; this does not model participant/session variability.
- Visual prominence, viewpoint, scene content, food identity and low-level image statistics could confound physical-size associations. Results are preliminary and noncausal.
