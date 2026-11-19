#!/usr/bin/env python3
"""Extract LAION ROI single-trial betas with bounded memory and aligned metadata.

Official LAION package required. A memmapped output avoids concatenating all
sessions in RAM. Crucially, this never mixes voxels from a general/ventral ROI.
"""
import argparse
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd
from khosla_common import DEFAULT_OUTPUT, image_key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', default='sub-01')
    p.add_argument('--sessions', nargs='+', default=['ses-01'], help='ses-01 ... or all')
    p.add_argument('--roi', default='laionventral')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--verify-against', type=Path, default=None,
                   help='Compare order and image labels with original single-session metadata')
    a = p.parse_args()
    try:
        from laion_fmri import load_subject
    except ImportError:
        try:
            from laion_fmri.subject import load_subject
        except ImportError as ex:
            raise SystemExit('Install official LAION-fMRI: python -m pip install "laion-fmri @ git+https://github.com/ViCCo-Group/LAION-fMRI.git@main"') from ex
    sub = load_subject(a.subject)
    rois = sub.get_available_rois()
    if a.roi not in rois:
        raise SystemExit(f'ROI {a.roi!r} unavailable. Options include: {rois}')
    sessions = list(sub.get_sessions() if a.sessions == ['all'] else a.sessions)
    if not sessions or len(sessions) != len(set(sessions)):
        raise SystemExit('Invalid/duplicate sessions')
    # Read just the trial tables first, to know exact memmap dimensions.
    tables = []
    for ses in sessions:
        trials = sub.get_trial_info(session=ses).copy()
        if 'label' not in trials or not len(trials):
            raise SystemExit(f'Empty or label-free trial table for {ses}')
        trials['session'] = ses
        trials['session_trial'] = np.arange(len(trials))
        tables.append(trials)
    df = pd.concat(tables, ignore_index=True)
    df['volume_index'] = np.arange(len(df))
    if a.verify_against:
        reference = a.verify_against.expanduser()
        old = pd.read_csv(reference, sep='\t' if reference.suffix.lower() == '.tsv' else ',')
        if 'label' not in old:
            raise SystemExit('Reference trial table lacks label')
        if 'volume_index' in old:
            old = old.sort_values('volume_index').reset_index(drop=True)
        if len(old) != len(df) or not np.array_equal(old.label.map(image_key), df.label.map(image_key)):
            raise SystemExit('Image names / trial order differ from reference: STOP')
        print('Verified full image order against', reference, flush=True)
    out = a.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    final = out / 'ventral_raw_betas.npy'
    temp = out / 'ventral_raw_betas.partial.npy'
    if temp.exists():
        temp.unlink()
    mm = None
    start = 0
    try:
        for ses, trials in zip(sessions, tables):
            # Only one session of betas in memory at a time.
            beta = sub.get_betas(session=ses, roi=a.roi, streaming=True)
            if not isinstance(beta, np.ndarray) or beta.ndim != 2 or beta.shape[0] != len(trials):
                raise SystemExit(f'Bad beta shape for {ses}: {getattr(beta, "shape", None)} expected rows={len(trials)}')
            if not np.isfinite(beta).all():
                raise SystemExit(f'Nonfinite betas in {ses}')
            if mm is None:
                mm = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32,
                                                shape=(len(df), beta.shape[1]))
            if beta.shape[1] != mm.shape[1]:
                raise SystemExit('Inconsistent voxel axis across sessions; STOP')
            stop = start + len(trials)
            mm[start:stop] = beta
            mm.flush()
            print(f'{a.subject} / {ses}: {beta.shape}, rows {start}:{stop}', flush=True)
            start = stop
            del beta
        if mm is None or start != len(df):
            raise SystemExit('Data extraction incomplete')
        mm.flush()
        del mm
        mm = None
        os.replace(temp, final)
        df.to_csv(out / 'ventral_trial_metadata.tsv', sep='\t', index=False)
        (out / 'roi_extraction_summary.json').write_text(json.dumps({
            'subject': a.subject, 'sessions': sessions, 'roi': a.roi,
            'trials': len(df), 'voxels': int(np.load(final, mmap_mode='r').shape[1]),
            'verified_against': str(a.verify_against) if a.verify_against else None,
            'caveat': 'LAION laionventral is an anatomical subdivision, not the exact NSD higher-level ventral ROI'
        }, indent=2))
        print('Saved', final, '\nand', out / 'ventral_trial_metadata.tsv')
    except BaseException:
        if mm is not None:
            del mm
        if temp.exists():
            temp.unlink()
        raise


if __name__ == '__main__':
    main()
