#!/usr/bin/env python3
"""Map food size-vs-fMRI correlations onto native-T1w ventral ROI brain slices.

Requires outputs of 13_analyze_food_realworld_size.py. Exports voxelwise
Spearman r maps, separately FWER-significant map, and the candidate consensus
component's spatial weights. These are NOT interchangeable: voxel r maps are
food-size associations, while component weights are an unsupervised anatomy map.

Plot native volume slices (mask silhouette or optional native-T1w MRI) and an
optional 3D point-cloud in native mm. No fsaverage/MNI/inflated surface projection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def volume_from_vector(vec, mask):
    positions=np.flatnonzero(mask.ravel(order='C'))
    if len(positions)!=len(vec):
        raise SystemExit(f'FATAL: mask contains {len(positions)} voxels, '
                         f'vector contains {len(vec)}. Refusing to mis-map voxels.')
    flat=np.zeros(mask.size,dtype=np.float32)
    flat[positions]=vec.astype(np.float32)
    return flat.reshape(mask.shape,order='C')


def write_nii(nib, ref, data, filename):
    im=nib.Nifti1Image(data.astype(np.float32),ref.affine,header=ref.header.copy())
    im.set_data_dtype(np.float32)
    nib.save(im,str(filename))
    print('Wrote',filename)


def show_slices(volume, mask, background, outfile, heading, threshold,
                symmetric=True, cmap='coolwarm', sig=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Prefer ROI median coordinate; signal maximum can be just one unrepresentative voxel.
    coords=np.array(np.where(mask))
    pivot=np.median(coords,axis=1).astype(int)
    lo=np.maximum(coords.min(axis=1)-9,0)
    hi=np.minimum(coords.max(axis=1)+10,np.array(mask.shape))
    if symmetric:
        mx=max(float(np.nanpercentile(np.abs(volume[mask]), 99)),1e-5)
        vmin,vmax=-mx,mx
    else:
        vmin=0;vmax=max(float(np.nanpercentile(volume[mask],99)),1e-6)
    fig,axs=plt.subplots(1,3,figsize=(15,5.1))
    # All indices are in native voxel grid; axis ordering/orientation deliberately not
    # labeled L/R because that depends on this particular scan's affine.
    for ax,dim,title in zip(axs,range(3),('i fixed (sagittal-like)',
                                        'j fixed (coronal-like)',
                                        'k fixed (axial-like)')):
        index=int(pivot[dim])
        sel=[slice(int(lo[q]),int(hi[q])) for q in range(3)]
        sel[dim]=index
        v=np.asarray(volume[tuple(sel)]).T
        m=np.asarray(mask[tuple(sel)]).T
        bg=np.asarray(background[tuple(sel)]).T
        if np.any(np.isfinite(bg)):
            low=float(np.nanpercentile(bg,5)); high=float(np.nanpercentile(bg,95))
            if high<=low: high=low+1
            ax.imshow(bg,cmap='gray',origin='lower',vmin=low,vmax=high,interpolation='nearest')
        v_plot=np.ma.masked_where(~m | (np.abs(v)<threshold),v)
        if np.ma.count(v_plot):
            im=ax.imshow(v_plot,cmap=cmap,origin='lower',interpolation='nearest',
                         vmin=vmin,vmax=vmax,alpha=.92)
        # Mark ROI boundary very faintly (makes masks visible without a background).
        if m.any() and min(m.shape)>2:
            ax.contour(m.astype(float),levels=[.5],colors=['#8ecae6'],linewidths=.55,alpha=.65)
        if sig is not None:
            s=np.asarray(sig[tuple(sel)]).T
            if np.any(s):
                ax.contour(s.astype(float),levels=[.5],colors=['yellow'],linewidths=1.5)
        ax.set_title(f'{title} | slice {index}',fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(heading,fontsize=12)
    fig.text(.5,.005,'Native-T1w voxel-grid slices; orientation is not normalized to MNI or fsaverage.',
             ha='center',fontsize=9)
    # Use a separate ScalarMappable so an all-subthreshold map still gets its colorbar.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cb=fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin,vmax=vmax),cmap=cmap),
                    ax=axs,shrink=.72,pad=.02)
    cb.set_label('Spearman r' if symmetric else 'Consensus NMF voxel weight')
    fig.subplots_adjust(left=.02,right=.89,top=.84,bottom=.11,wspace=.04)
    fig.savefig(outfile,dpi=190,bbox_inches='tight')
    plt.close(fig)
    print('Wrote',outfile)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',required=True,type=Path,help='analysis/khosla_replication/sub-01/ses-01')
    p.add_argument('--mask',required=True,type=Path,help='The exact native laionventral mask used in extraction')
    p.add_argument('--method',choices=['bayesian','sklearn'],default='bayesian')
    p.add_argument('--rank',type=int,default=20)
    p.add_argument('--analysis-dir',type=Path,default=None)
    p.add_argument('--bg-img',type=Path,default=None,help='Optional native T1w structural MRI: must match mask affine and shape')
    p.add_argument('--display-threshold',type=float,default=.20,help='ABS(r) visualization threshold, NOT significance')
    p.add_argument('--interactive-3d',action='store_true',help='3D voxels point-cloud native mm (not cortical mesh)')
    p.add_argument('--top-voxel-percent',type=float,default=5.)
    a=p.parse_args()
    if not 0<=a.display_threshold<1: p.error('display threshold must be in [0,1)')
    if not 0<a.top_voxel_percent<=100: p.error('top-voxel-percent in (0,100]')
    try:
        import nibabel as nib
    except ImportError as e:raise SystemExit('Install nibabel') from e
    root=a.input_dir.expanduser()
    d=root/f'consensus_{a.method}_rank{a.rank}'
    ana=(a.analysis_dir or d/'realworld_food_size').expanduser()
    data=np.load(ana/'voxel_food_size_stats.npz')
    table=pd.read_csv(ana/'component_food_realworld_size.csv')
    best=int(table.sort_values('size_spearman_r',ascending=False).iloc[0].component)
    mask_nii=nib.load(str(a.mask.expanduser()))
    mask=np.asarray(mask_nii.dataobj)>0
    if mask.ndim!=3:raise SystemExit('ROI mask must be 3D')
    v=np.asarray(data['spearman_r'])
    famv=np.asarray(data['within_family_r'])
    p_fwer=np.asarray(data['p_FWER_voxels'])
    w=np.load(d/'consensus_voxel_weights.npy')
    if w.shape!=(a.rank,int(mask.sum())):
        raise SystemExit(f'CONSENSUS MASK MISMATCH. W={w.shape}, mask voxels={int(mask.sum())}. STOP.')
    raw=volume_from_vector(v,mask)
    famraw=volume_from_vector(famv,mask)
    significant=volume_from_vector(np.where(p_fwer<.05,v,0),mask)
    significance_boolean=volume_from_vector((p_fwer<.05).astype(np.float32),mask)>0
    component=volume_from_vector(w[best-1],mask)
    out=ana/'brain_maps';out.mkdir(parents=True,exist_ok=True)
    write_nii(nib,mask_nii,raw,out/'food_size_voxel_spearman_r_nativeT1w.nii.gz')
    write_nii(nib,mask_nii,famraw,out/'food_size_voxel_within_family_r_nativeT1w.nii.gz')
    write_nii(nib,mask_nii,significant,out/'food_size_FWER_significant_voxels_nativeT1w.nii.gz')
    write_nii(nib,mask_nii,component,out/f'candidate_component_{best:02d}_weights_nativeT1w.nii.gz')
    if a.bg_img:
        bg_nii=nib.load(str(a.bg_img.expanduser()))
        if bg_nii.shape != mask_nii.shape or not np.allclose(bg_nii.affine,mask_nii.affine,atol=1e-3):
            raise SystemExit('Background anatomical image does not share native mask shape and affine. STOP.')
        background=np.asarray(bg_nii.dataobj,dtype=np.float32)
        print('Using matching native anatomical MRI:',a.bg_img)
    else:
        background=mask.astype(np.float32)
        print('No structural MRI supplied: using ventral ROI silhouette as background.')
    show_slices(raw,mask,background,out/'food_size_voxel_correlations_native_slices.png',
                f'Real-world food size: voxelwise Spearman r | top ROI candidates (n={int(mask.sum())} voxels)\n'
                'Exploratory map; yellow contours are FWER p<0.05; colors are NOT significance',
                a.display_threshold,sig=significance_boolean)
    show_slices(famraw,mask,background,out/'food_size_within_family_native_slices.png',
                'Food physical size vs voxel response | stimulus-family-adjusted r (exploratory)',
                a.display_threshold)
    weight_positive=np.asarray(w[best-1]);cutoff=float(np.percentile(weight_positive,100-a.top_voxel_percent))
    show_slices(component,mask,background,out/f'candidate_component_{best:02d}_native_slices.png',
                f'Consensus component {best}: spatial NMF weights (NOT food-size significance)',
                max(cutoff,1e-12),symmetric=False,cmap='magma')
    if a.interactive_3d:
        try:import plotly.graph_objects as go
        except ImportError as e:raise SystemExit('Install plotly') from e
        inds=np.array(np.where(mask)).T
        keep=np.abs(v)>=max(a.display_threshold,float(np.quantile(np.abs(v),.9)))
        xyz=nib.affines.apply_affine(mask_nii.affine,inds[keep])
        if not len(xyz):
            print('No voxels above 3D display threshold; skipping interactive plot')
        else:
            sz=v[keep]
            lim=max(np.max(np.abs(sz)),.01)
            fig=go.Figure(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
                  marker=dict(size=3,color=sz,colorscale='RdBu',cmin=-lim,cmax=lim,
                              colorbar=dict(title='Spearman r'),opacity=.78),
                  text=[f'Spearman r={v:.3f}' for v in sz],hoverinfo='text'))
            fig.update_layout(title='Real-world food size: native T1w ROI voxels, exploratory correlations',
                 scene=dict(xaxis_title='Native X (mm)',yaxis_title='Native Y (mm)',
                            zaxis_title='Native Z (mm)',aspectmode='data'))
            f=out/'food_size_native_voxels_interactive.html'
            fig.write_html(f,include_plotlyjs=True)
            print('Wrote',f)
    (out/'map_metadata.json').write_text(json.dumps({
        'mask':str(a.mask),'input_dir':str(root),'analysis_dir':str(ana),
        'candidate_component':best,'mask_voxel_count':int(mask.sum()),
        'fwer_significant_voxels':int(np.count_nonzero(p_fwer<.05)),
        'visual_threshold_absolute_r':a.display_threshold,
        'notes':['Image-size voxel map is voxelwise rank correlation on food images only.',
                 'These are native T1w volume maps, NOT cortical surface maps.',
                 'Candidate component weights are spatial weights only and NOT tested voxel associations.',
                 'FWER correction spans all ROI voxels, separate from component-level correction.',
                 'Anatomical interpretation depends on exact voxel-mask order and correct native geometry.']},indent=2))
    print('Finished. Brain maps:',out)

if __name__=='__main__':main()
