from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import json

import nibabel as nib

import anise.process.fusi_glm_fit2

root = Path('fUS_data') / 'fUSI_Rat_BIDS'

reg_path = root / 'derivatives' / 'registration'

df = pd.read_csv(reg_path / 'experiment_data.csv')

glm_path = root / 'derivatives' / 'glm'
glm_path.mkdir(parents=True, exist_ok=True)


def get_pwd_basename(pwd_path):
    return '_'.join('.'.join(pwd_path.stem.split('.')[:-1]
                             ).split('_')[:-1])


def do_one_glm(pwd_path):
    pwd_path = Path(pwd_path)
    sub_ses_dir = (pwd_path.relative_to(root /
                   'sourcedata')).parent.parent
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    this_glm_path = glm_path / sub_ses_dir / 'fus'
    basename = get_pwd_basename(pwd_path)
    basename_events = '_'.join([group for group in basename.split('_')
                                if 'proc' not in group and 'acq' not in group])

    pwd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
    pwd_data = np.array(pwd_img.dataobj)
    with open(root / 'sourcedata' / sub_ses_dir / 'fus' /
                f'{basename}_pwdt.json', 'r') as fid:
        time_stamps = np.array(json.load(fid)['VolumeTiming'])
    
    # get and apply nan mask
    pwd_img_orig = nib.load(root / 'sourcedata' / sub_ses_dir /
                            'fus' / f'{basename}_pwdt.nii.gz')
    pwd_data_orig = np.array(pwd_img_orig.dataobj)
    mask = ~np.isnan(pwd_data_orig).any(axis=tuple(range(pwd_img.ndim - 1)))
    time_stamps = time_stamps[mask]

    transformations = np.loadtxt(this_reg_path / f'{basename}_reg.txt')
    events = pd.read_csv(root / 'sourcedata' / sub_ses_dir /
                         'beh' / f'{basename_events}_events.tsv', sep='\t')
    event = events.loc[0, 'trial_type']

    shifts, max_zmaps, best_offset = \
        anise.process.fusi_glm_fit2.fit_glm_time_shift(
            pwd_img, time_stamps, transformations, event, events,
            out_dir=this_glm_path / basename, shift=20
        )
    return shifts, max_zmaps, best_offset


output = Parallel(n_jobs=20)(
    delayed(do_one_glm)(pwd_path)
    for pwd_path in tqdm(df['power_doppler_fname'], total=len(df)))
"""
for pwd_path in tqdm(df['power_doppler_fname'], total=len(df)):
    do_one_glm(pwd_path)
"""

df['shifts'] = [list(out[0]) for out in output]
df['max_zmaps'] = [list(out[1]) for out in output]
df['shift'] = [out[2] for out in output]
df['max_zmap'] = [max_zmaps[shifts.index(best_shift)]
                  for max_zmaps, shifts, best_shift in
                  zip(df['max_zmaps'], df['shifts'],
                      df['shift'])]
df.to_csv(glm_path / 'experiment_data.csv', index=False)
