from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from shutil import copyfile

import nibabel as nib

import anise.utils

import matplotlib.pyplot as plt

root = Path('fUS_data') / 'fUSI_Rat_BIDS'

df = pd.read_csv(root / 'experiment_data.csv')

reg_path = root / 'derivatives' / 'registration'
reg_plots = reg_path / 'plots'
reg_plots.mkdir(parents=True, exist_ok=True)

# %%
# Exclude mismatched bmodes
drop = list()
for i, row in tqdm(df.iterrows(), total=len(df)):
    # get power doppler image
    power_doppler_path = Path(row['power_doppler_fname'])
    pd_img = nib.load(power_doppler_path)
    sub_ses_dir = (power_doppler_path.relative_to(root /
                   'sourcedata')).parent.parent
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
    # get matching bmodes
    basename_bmode = '_'.join([group for group in basename.split('_')
                                if 'proc' not in group])
    bmode_fnames = list((root / 'rawdata' / sub_ses_dir / 'fus').glob(
                         f'{basename_bmode}*.h5'))
    bmode_df = pd.read_csv(root / 'rawdata' / sub_ses_dir / 'fus' /
                           f'{basename_bmode}_bmode.tsv', sep='\t')
    if pd_img.shape[-1] != len(bmode_fnames):
        drop.append(i)

perc = np.round(100 * len(drop) / len(df), 3)
print(f'{len(drop)} runs with mismatching bmode count dropped ({perc}%)')
df = df.drop(drop, axis='index').reset_index()

# %%
# Do registration
for i, row in tqdm(df.iterrows()):
    power_doppler_path = Path(row['power_doppler_fname'])
    this_reg_path = reg_path / \
        (power_doppler_path.relative_to(root / 'sourcedata')).parent
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])

    pd_img = nib.load(power_doppler_path)
    
    reg_img_path = this_reg_path / f'{basename}_pwdt.nii.gz'
    fusi_data_3d_reg, transformations = anise.utils.register_image_stack(
        np.array(pd_img.dataobj)[:, 0])
    reg_img_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(fusi_data_3d_reg[:, None], pd_img.affine),
             reg_img_path)
    reg_trans_path = this_reg_path / f'{basename}_reg.txt'
    np.savetxt(reg_trans_path, transformations)


# %%
# Exclude large movement
df_reg = pd.DataFrame(columns=['power_doppler_fname',
                               'power_doppler_path',
                               'max_displacement',
                               'delta_activity_over'])
for i, row in tqdm(df.iterrows(), total=len(df)):
    power_doppler_path = Path(row['power_doppler_fname'])
    this_reg_path = reg_path / \
        (power_doppler_path.relative_to(root / 'sourcedata')).parent
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
    # difference in activation
    pd_img = nib.load(power_doppler_path)
    pd_data = np.array(pd_img.dataobj)
    baseline = pd_data[..., :10].mean(axis=-1)
    axes = tuple(np.arange(pd_data.ndim - 1))
    delta_activity = \
        (pd_data - baseline[..., None]).mean(axis=axes) / baseline.mean()

    fig, ax = plt.subplots()
    ax.plot(delta_activity)
    ax.set_ylim([-5, 5])
    ax.set_xlabel('Power Doppler Frame')
    ax.set_ylabel('Change in Activity vs 10 Frame Baseline')
    fig.savefig(reg_plots / f'{basename}_da.png')
    plt.close(fig)

    reg_trans_path = this_reg_path / f'{basename}_reg.txt'
    transformations = np.loadtxt(reg_trans_path)

    df_reg.loc[len(df_reg.index)] = \
        (row['power_doppler_fname'],
         row['power_doppler_path'],
         np.abs(transformations).max(),
         (np.abs(delta_activity) > 0.5).sum())
    
    fig, ax = plt.subplots()
    ax.plot(transformations.T)
    ax.set_ylim([-5, 5])
    ax.set_xlabel('Power Doppler Frame')
    ax.set_ylabel('Displacement (mm)')
    fig.savefig(reg_plots / f'{basename}_reg.png')
    plt.close(fig)

fig, ax = plt.subplots()
ax.hist(df_reg['delta_activity_over'], np.linspace(0, 20, 51))
ax.set_xlabel(r'#frames over 50% baseline')
fig.savefig(reg_plots / 'activity.png')
plt.close(fig)

fig, ax = plt.subplots()
ax.hist(df_reg['max_displacement'], np.linspace(0, 20, 51))
ax.set_xlabel('max displacement')
fig.savefig(reg_plots / 'max_displacement.png')
plt.close(fig)

drop = np.where(df_reg['max_displacement'] > 3)[0]
drop2 = np.where(df_reg['delta_activity_over'] > 3)[0]
# print(df_reg.loc[drop].to_string())
# print(df_reg.loc[drop2].to_string())
perc = np.round(100 * len(drop) / len(df), 3)
print(f'{len(drop)} runs with large movement dropped ({perc}%)')

df = df.drop(drop, axis='index').reset_index()

df.to_csv(reg_path / 'experiment_data.csv', index=False)
