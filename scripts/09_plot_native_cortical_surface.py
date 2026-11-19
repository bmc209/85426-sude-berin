#!/usr/bin/env python3
"""Project LAION-fMRI native-T1w component maps onto native FreeSurfer cortical surfaces.

Requirements: numpy nibabel nilearn matplotlib scipy

The source NIfTI is in native T1w scanner RAS. FreeSurfer lh/rh.{white,pial,inflated}
coordinates are tkRAS; transform using subject's mri/T1.mgz header.
Sampling is performed along the white-to-pial cortical ribbon, then visualized
on the inflated mesh. No MNI/fsaverage registration is assumed.

Surface maps are descriptive component weights, NOT statistical food contrasts.
"""
import argparse
from pathlib import Path
import numpy as np


def tkr_to_scanner(fs_dir):
    import nibabel as nib
    orig = fs_dir / 'mri' / 'T1.mgz'
    if not orig.is_file():
        orig = fs_dir / 'mri' / 'orig.mgz'
    if not orig.is_file():
        raise SystemExit('Missing FreeSurfer mri/T1.mgz or orig.mgz in ' + str(fs_dir) + '\n'
                         'Cannot safely convert surface tkRAS to scanner RAS.')
    image = nib.load(str(orig))
    hdr = image.header
    return np.asarray(hdr.get_vox2ras()) @ np.linalg.inv(np.asarray(hdr.get_vox2ras_tkr()))


def read_mesh(path, transform):
    from nibabel.freesurfer.io import read_geometry
    from nibabel.affines import apply_affine
    xyz, triangles = read_geometry(str(path))
    xyz = apply_affine(transform, xyz).astype(np.float32)
    return (xyz, triangles.astype(np.int32))


def overlap_diagnostics(mesh, img, label):
    import nibabel as nib
    xyz, _ = mesh
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)
    inside = np.all((vox >= 0) & (vox <= np.asarray(img.shape[:3]) - 1), axis=1)
    portion = inside.mean()
    print(f'{label}: {inside.sum():,}/{len(inside):,} vertices in NIfTI bounding box ({portion:.1%})')
    if portion < .90:
        raise SystemExit(
            f'Coordinate-space mismatch for {label}: only {portion:.1%} vertices inside '
            'the NIfTI volume. Do NOT render misleading surface maps. '
            'Check that FreeSurfer subject and native T1w volume belong to the same space.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir', type=Path, required=True, help='Khosla replication subject/session directory')
    p.add_argument('--freesurfer-dir', type=Path, required=True,
                   help='Subject FreeSurfer directory containing surf/ and mri/T1.mgz')
    p.add_argument('--method', default='bayesian')
    p.add_argument('--rank', type=int, default=20)
    p.add_argument('--component', type=int, default=None, help='1-based component; default all')
    p.add_argument('--top-percent', type=float, default=10.0,
                   help='Highlight top percentage of positive sampled cortical values PER component')
    p.add_argument('--n-samples', type=int, default=5, help='Depth samples from white to pial')
    p.add_argument('--no-html', action='store_true', help='Skip interactive plotly HTML surface viewer')
    a = p.parse_args()
    if not 0 < a.top_percent <= 100:
        p.error('--top-percent must be >0 and <=100')
    if a.n_samples < 1:
        p.error('--n-samples must be >=1')

    import nibabel as nib
    from nilearn import surface, plotting
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    input_dir = a.input_dir.expanduser().resolve()
    fs_dir = a.freesurfer_dir.expanduser().resolve()
    for rel in ('surf/lh.pial', 'surf/rh.pial', 'surf/lh.white', 'surf/rh.white',
                'surf/lh.inflated', 'surf/rh.inflated'):
        if not (fs_dir / rel).is_file():
            raise SystemExit('Missing required native FreeSurfer surface: ' + str(fs_dir / rel) + '\n'
                             'Use `find ~/shared/laion-fmri -name lh.inflated -o -name rh.inflated` '
                             'to find actual FreeSurfer subject directory.')

    maps_dir = input_dir / f'consensus_{a.method}_rank{a.rank}' / 'brain_maps'
    if not maps_dir.is_dir():
        raise SystemExit('No brain_maps folder: ' + str(maps_dir) + '\nRun 08_plot_component_brain.py first.')
    comps = [a.component] if a.component is not None else list(range(1, a.rank + 1))
    if any(c < 1 or c > a.rank for c in comps):
        p.error('Invalid component number')

    transform = tkr_to_scanner(fs_dir)
    print('FreeSurfer tkRAS -> scannerRAS affine:\n', np.array2string(transform, precision=3))
    meshes = {}
    for hemi in ('lh', 'rh'):
        pial = read_mesh(fs_dir / 'surf' / (hemi + '.pial'), transform)
        white = read_mesh(fs_dir / 'surf' / (hemi + '.white'), transform)
        inflated = read_mesh(fs_dir / 'surf' / (hemi + '.inflated'), transform)
        if len(pial[0]) != len(white[0]) or len(pial[0]) != len(inflated[0]):
            raise SystemExit(f'Vertex count mismatch among {hemi} FreeSurfer meshes')
        meshes[hemi] = {'pial': pial, 'white': white, 'inflated': inflated}

    out = maps_dir / 'cortical_surface'
    out.mkdir(parents=True, exist_ok=True)
    first = True
    for c in comps:
        vol_file = maps_dir / f'component_{c:02d}_weights_nativeT1w.nii.gz'
        if not vol_file.is_file():
            raise SystemExit('Missing component volume: ' + str(vol_file))
        nii = nib.load(str(vol_file))
        if first:
            for hemi in ('lh', 'rh'):
                overlap_diagnostics(meshes[hemi]['pial'], nii, hemi+' pial')
                overlap_diagnostics(meshes[hemi]['white'], nii, hemi+' white')
            first = False
        cortex = {}
        for hemi in ('lh', 'rh'):
            pial = meshes[hemi]['pial']
            white = meshes[hemi]['white']
            # Volumetric interpolation within native cortical ribbon, not on inflated coords.
            depths = np.linspace(0.1, 0.9, a.n_samples).tolist()
            vals = surface.vol_to_surf(nii, surf_mesh=pial, inner_mesh=white,
                                       kind='depth', depth=depths, interpolation='linear')
            vals = np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0., posinf=0., neginf=0.)
            if not np.any(vals > 0):
                print(f'WARNING: component {c:02d} {hemi}: no positive samples; check alignment!')
            cortex[hemi] = vals
            nib.freesurfer.io.write_morph_data(str(out / f'component_{c:02d}_{hemi}_weights.curv'), vals)
            np.save(out / f'component_{c:02d}_{hemi}_weights.npy', vals)

        positive = np.concatenate([v[v > 0] for v in cortex.values()])
        if positive.size == 0:
            raise SystemExit(f'Component {c} has no positive surface values in either hemisphere. '
                             'Likely voxel-to-surface coordinate mismatch.')
        cut = float(np.percentile(positive, 100 - a.top_percent))
        top_max = max(np.max(v) for v in cortex.values())
        print(f'Component {c:02d}: top {a.top_percent:g}% cutoff={cut:.6g}, max={top_max:.6g}')

        # Nilearn generates static inflated surface figures with actual cortical mesh topology.
        fig = plt.figure(figsize=(16, 9), facecolor='white')
        for i, (hemi, view) in enumerate((('lh','lateral'),('lh','medial'),('rh','lateral'),('rh','medial'))):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            vals = np.where(cortex[hemi] >= cut, cortex[hemi], 0.)
            plotting.plot_surf_stat_map(
                meshes[hemi]['inflated'], vals, hemi='left' if hemi=='lh' else 'right',
                view=view, threshold=cut, cmap='magma', colorbar=False,
                bg_on_data=False, axes=ax, figure=fig)
            ax.set_title(f'{hemi.upper()} {view}', fontsize=11)
        fig.suptitle(f'LAION-fMRI consensus NMF component {c:02d} | top {a.top_percent:g}% surface weights', fontsize=16)
        fig.savefig(out / f'component_{c:02d}_inflated_surface.png', dpi=170, bbox_inches='tight')
        plt.close(fig)

        if not a.no_html:
            # For each hemisphere, optional interactive brain display (requires plotly).
            for hemi in ('lh','rh'):
                vals = np.where(cortex[hemi] >= cut, cortex[hemi], 0.)
                try:
                    view = plotting.view_surf(meshes[hemi]['inflated'], surf_map=vals, cmap='magma',
                                             threshold=cut, symmetric_cmap=False, colorbar=True)
                    view.save_as_html(str(out / f'component_{c:02d}_{hemi}_interactive.html'))
                except Exception as ex:
                    print(f'Interactive view skipped for {hemi}: {ex}')
        print(f'  wrote {out / f"component_{c:02d}_inflated_surface.png"}')

    print('\nFinished. Cortical surface figures:', out)
    print('Maps are native-space NMF WEIGHTS, not statistically significant food contrasts.')


if __name__ == '__main__':
    main()
