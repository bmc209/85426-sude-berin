#!/usr/bin/env python3
"""Map Khosla-style consensus NMF voxel weights to native-T1w ventral ROI.

Outputs native-space NIfTI maps, slice PNGs, and optional interactive 3D HTML
point clouds. Uses the EXACT volumetric mask and C-order boolean voxel indexing
used by extract_responses.py. Verify this assumption in the extraction script.

Dependencies: numpy, nibabel, nilearn, matplotlib; plotly for --interactive-3d.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True,
                   help='analysis/khosla_replication/sub-01/ses-01')
    p.add_argument('--mask', type=Path, required=True,
                   help='The original volumetric laionventral NIfTI ROI mask')
    p.add_argument('--weights', type=Path,
                   help='Override consensus weights .npy (components x voxels)')
    p.add_argument('--method', default='bayesian')
    p.add_argument('--rank', type=int, default=20)
    p.add_argument('--component', type=int, default=None,
                   help='1-based component number; default: export all')
    p.add_argument('--bg-img', type=Path, default=None,
                   help='Optional native-space T1w anatomical NIfTI matching mask space')
    p.add_argument('--top-percent', type=float, default=5.0,
                   help='Top percentile of positive weights to highlight')
    p.add_argument('--interactive-3d', action='store_true',
                   help='Create HTML native-voxel-coordinate point cloud using plotly')
    a = p.parse_args()
    if not 0 < a.top_percent <= 100:
        p.error('--top-percent must be in (0,100]')
    import nibabel as nib
    from nilearn import plotting
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    input_dir = a.input_dir.expanduser().resolve()
    weights_file = (a.weights or input_dir / f'consensus_{a.method}_rank{a.rank}' /
                    'consensus_voxel_weights.npy').expanduser().resolve()
    W = np.load(weights_file)
    if W.ndim != 2 or not np.all(np.isfinite(W)):
        raise SystemExit(f'Bad voxel weight matrix: {W.shape}')
    mask_nii = nib.load(str(a.mask.expanduser()))
    mask_data = np.asarray(mask_nii.dataobj) > 0
    if mask_data.ndim != 3:
        raise SystemExit('Mask must be 3D')
    positions = np.flatnonzero(mask_data.ravel(order='C'))
    if W.shape[1] != len(positions):
        raise SystemExit(f'Voxel order/mask mismatch! W has {W.shape[1]} voxels, '
                         f'mask contains {len(positions)}. STOP; do not plot.')
    if a.bg_img:
        bg_nii = nib.load(str(a.bg_img.expanduser()))
        if bg_nii.shape[:3] != mask_nii.shape or not np.allclose(bg_nii.affine, mask_nii.affine, atol=1e-3):
            raise SystemExit('Anatomical background does not match native mask geometry.')
        bg = str(a.bg_img.expanduser())
    else:
        bg = None
    comps = [a.component] if a.component is not None else list(range(1, W.shape[0] + 1))
    if any(c < 1 or c > W.shape[0] for c in comps):
        raise SystemExit(f'Component numbers must be between 1 and {W.shape[0]}')
    out = input_dir / f'consensus_{a.method}_rank{a.rank}' / 'brain_maps'
    out.mkdir(parents=True, exist_ok=True)
    import csv
    rows = []
    for c in comps:
        w = np.asarray(W[c - 1], dtype=np.float32)
        volume = np.zeros(mask_data.size, dtype=np.float32)
        volume[positions] = w
        volume = volume.reshape(mask_data.shape, order='C')
        img = nib.Nifti1Image(volume, mask_nii.affine, header=mask_nii.header.copy())
        img.set_data_dtype(np.float32)
        fname = out / f'component_{c:02d}_weights_nativeT1w.nii.gz'
        nib.save(img, str(fname))
        positive = w[w > 0]
        if len(positive) == 0:
            print(f'Component {c}: no positive weights')
            continue
        cutoff = float(np.percentile(positive, 100 - a.top_percent))
        # Explicit threshold map so top X% means the same in all visualizers.
        top = np.where(volume >= cutoff, volume, 0.0).astype(np.float32)
        top_img = nib.Nifti1Image(top, mask_nii.affine, header=mask_nii.header.copy())
        top_img.set_data_dtype(np.float32)
        nib.save(top_img, str(out / f'component_{c:02d}_top{a.top_percent:g}pct_nativeT1w.nii.gz'))
        cut_voxel = np.unravel_index(positions[int(np.argmax(w))], mask_data.shape)
        cut_world = nib.affines.apply_affine(mask_nii.affine, cut_voxel)
        fig = plt.figure(figsize=(13, 4.5))
        display = plotting.plot_stat_map(img, bg_img=bg, display_mode='ortho',
                    cut_coords=tuple(float(q) for q in cut_world),
                    threshold=cutoff, cmap='magma', colorbar=True,
                    black_bg=False,
                    title=f'Consensus component {c} | top {a.top_percent:g}% positive weights',
                    figure=fig)
        display.savefig(str(out / f'component_{c:02d}_slices.png'), dpi=160)
        display.close()
        if a.interactive_3d:
            import plotly.graph_objects as go
            ix = np.flatnonzero(w >= cutoff)
            # Upper bound for responsive interactive HTML.
            if len(ix) > 5000:
                ix = ix[np.argsort(w[ix])[-5000:]]
            xyz = np.array(np.unravel_index(positions[ix], mask_data.shape)).T
            # This is a point cloud of ROI voxels, not a full cortical reconstruction.
            fig3 = go.Figure(data=[go.Scatter3d(
                x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode='markers',
                marker=dict(size=2.5, color=w[ix], colorscale='Magma', opacity=.8,
                            colorbar=dict(title='Voxel weight')),
                text=[f'Weight: {v:.4g}' for v in w[ix]], hoverinfo='text')])
            fig3.update_layout(title=f'Component {c}: high-weight voxels (native voxel coordinates)',
                scene=dict(xaxis_title='Voxel i', yaxis_title='Voxel j', zaxis_title='Voxel k',
                           aspectmode='data'), margin=dict(l=0,r=0,b=0,t=50))
            fig3.write_html(out / f'component_{c:02d}_3d_voxels.html', include_plotlyjs=True)
        rows.append(dict(component=c, max_weight=float(w.max()),
                         threshold_top_percent=cutoff,
                         highlighted_voxels=int(np.count_nonzero(w >= cutoff)),
                         total_voxels=len(w)))
        print(f'Component {c:02d}: {len(w)} voxels, top cutoff {cutoff:.6g}; saved maps/PNG')
    with open(out / 'map_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['component','max_weight',
                  'threshold_top_percent','highlighted_voxels','total_voxels'])
        writer.writeheader(); writer.writerows(rows)
    print(f'All outputs: {out}')
    print('NOTE: Native-T1w coordinates; do not overlay directly on MNI/fsaverage.')
    print('VERIFY original extraction uses mask>0 with NumPy C-order flattening.')


if __name__ == '__main__':
    main()
