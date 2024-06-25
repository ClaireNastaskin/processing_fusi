from pathlib import Path
from shutil import copyfile, rmtree
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import json

import nibabel as nib
from skimage.measure import label
from skimage.morphology import closing


import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting, image
from nilearn.image import mean_img

import anise.process.fusi_glm_fit2
import anise.process.image_metrics

root = Path('fUS_data') / 'fUSI_Rat_BIDS'

reg_path = root / 'derivatives' / 'registration'

df = pd.read_csv(reg_path / 'experiment_data.csv')

glm_path = root / 'derivatives' / 'glm'
glm_path.mkdir(parents=True, exist_ok=True)


def get_pwd_basename(power_doppler_path):
    return '_'.join('.'.join(power_doppler_path.stem.split('.')[:-1]
                             ).split('_')[:-1])


def do_one_glm(power_doppler_path):
    power_doppler_path = Path(power_doppler_path)
    sub_ses_dir = (power_doppler_path.relative_to(root /
                   'sourcedata')).parent.parent
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    this_glm_path = glm_path / sub_ses_dir / 'fus'
    basename = get_pwd_basename(power_doppler_path)
    basename_events = '_'.join([group for group in basename.split('_')
                                if 'proc' not in group and 'acq' not in group])

    pwd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
    with open(root / 'sourcedata' / sub_ses_dir / 'fus' /
              f'{basename}_pwdt.json', 'r') as fid:
        time_stamps = np.array(json.load(fid)['VolumeTiming'])
    transformations = np.loadtxt(this_reg_path / f'{basename}_reg.txt')
    events = pd.read_csv(root / 'sourcedata' / sub_ses_dir /
                         'beh' / f'{basename_events}_events.tsv', sep='\t')
    event = events.loc[0, 'trial_type']

    shifts, max_tstats, best_offset = \
        anise.process.fusi_glm_fit2.fit_glm_time_shift(
            pwd_img, time_stamps, transformations, event, events,
            out_dir=this_glm_path / basename, shift=20
        )
    return shifts, max_tstats, best_offset


output = Parallel(n_jobs=20)(
    delayed(do_one_glm)(power_doppler_path)
    for power_doppler_path in df['power_doppler_fname'])

df['shifts'] = [list(out[0]) for out in output]
df['max_tstats'] = [list(out[1]) for out in output]
df['shift'] = [out[2] for out in output]
df['max_tstat'] = [max_tstats[shifts.index(best_shift)]
                   for max_tstats, shifts, best_shift in
                   zip(df['max_tstats'], df['shifts'],
                       df['best_shift'])]
"""
df['shift'] = [row['shifts'][np.argmax(row['max_tstats'])]
               for _, row in df.iterrows()]
df['max_tstat'] = [np.max(max_tstats) for max_tstats in df['max_tstats']]
"""

# %%
# Exclude bad zmap
n_clusters_keep = 2
thresh = 3
min_cluster_size = 15
df_zmap = pd.DataFrame(
    columns=['power_doppler_path', 'n_clusters',
             'largest_cluster', 'average_cluster_size',
             'largest_clusters_max_z']
)
largest_cluster_idxs = dict()
for i, row in tqdm(df.iterrows(), total=len(df)):
    power_doppler_path = Path(row['power_doppler_fname'])
    sub_ses_dir = (power_doppler_path.relative_to(root /
                   'sourcedata')).parent.parent
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    this_glm_path = glm_path / sub_ses_dir / 'fus'
    zmap = nib.load(this_glm_path / basename / 'best_zmap.nii.gz')
    zmap_data = np.array(zmap.dataobj)
    zmap_data_proc = closing(np.abs(zmap_data) > thresh)
    labels = label(zmap_data_proc)
    idxs = np.unique(labels)
    idxs = idxs[idxs > 0]  # 0 is background
    if idxs.size == 0:
        df_zmap.loc[len(df_zmap.index)] = power_doppler_path, 0, 0, 0, 0
        continue

    sizes = np.array([np.sum(labels == idx) for idx in idxs])
    n_clusters, largest_cluster, average_cluster_size = \
        sizes.size, np.max(sizes), np.mean(sizes)
    i_sorted = np.argsort(sizes)
    largest_clusters = [
        np.array(np.where(labels == idxs[i_sorted[-j]])).T
        for j in range(1, n_clusters_keep + 1)
    ]

    largest_clusters = np.concatenate(largest_clusters, axis=0)
    largest_clusters_max_z = \
        np.max(np.abs([zmap_data[tuple(idx)] for idx in
                       largest_clusters]))
    largest_cluster_idxs[power_doppler_path] = largest_clusters
        
    df_zmap.loc[len(df_zmap.index)] = \
        (power_doppler_path, n_clusters, largest_cluster,
         average_cluster_size, largest_clusters_max_z)

df['max_tstat2'] = df_zmap['largest_clusters_max_z']
df['largest_cluster'] = df_zmap['largest_cluster']
np.savez_compressed(
    glm_path / 'cluster_idxs.npz',
    **{str(k): v for k, v in largest_cluster_idxs.items()}
)

keep_dir = glm_path / "plots" / "keep"
exclude_dir = glm_path / "plots" / "exclude"
if keep_dir.exists():
    rmtree(keep_dir)
if exclude_dir.exists():
    rmtree(exclude_dir)
keep_dir.mkdir(parents=True, exist_ok=True)
exclude_dir.mkdir(parents=True, exist_ok=True)

rule_func = lambda row: row['largest_cluster'] > min_cluster_size
for i, row in tqdm(df_zmap.iterrows(), total=len(df)):
    power_doppler_path = Path(row['power_doppler_path'])
    sub_ses_dir = (power_doppler_path.relative_to(root /
                   'sourcedata')).parent.parent
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    this_glm_path = glm_path / sub_ses_dir / 'fus'
    out_dir = keep_dir if rule_func(row) else exclude_dir
    copyfile(this_glm_path / basename / 'best_zmap.png',
             out_dir / f'{basename}_zmap.png')

for col in df_zmap.columns[1:]:
    ax = sns.displot(df_zmap, x=col)
    ax.figure.savefig(glm_path / "plots" / f"{col}.png")
    plt.close(ax.figure)

keep = [p.stem for p in (glm_path / 'plots' / 'keep').glob('*')]
print(f'{np.round(100 * len(keep) / len(df), 2)}% active zmaps kept '
      f'({len(keep)} / {len(df)})')

df.to_csv(glm_path / 'experiment_data.csv', index=False)

# manually fix okay maps in keep
# bash:
# keep () { mv $1 ${1/"exclude"/"keep"}; }
# keep FILENAME

# %%
# Verify kept zmaps clusters are reasonable
(glm_path / 'plots' / 'keep_qc').mkdir(exist_ok=True)
for i, row in tqdm(df.iterrows(), total=len(df)):
    power_doppler_path = Path(row['power_doppler_fname'])
    sub_ses_dir = (power_doppler_path.relative_to(root /
                    'sourcedata')).parent.parent
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
    if f'{basename}_zmap' not in keep:
        continue
    pwd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
    zmap = nib.load(glm_path / sub_ses_dir / 'fus' / basename / 'zmap.nii.gz')
    zmap_data_orig = np.array(zmap.dataobj)
    zmap_data = np.zeros_like(zmap_data_orig)
    for idx in largest_cluster_idxs[power_doppler_path]:
        zmap_data[tuple(idx)] = zmap_data_orig[tuple(idx)]
    zmap_cluster = nib.Nifti1Image(zmap_data, zmap.affine)
    display = plotting.plot_stat_map(
        zmap_cluster,
        bg_img=mean_img(pwd_img),
        cut_coords=[0],
        threshold=3,
        display_mode="y",
        black_bg=True,
        vmin=-10,
        vmax=10
    )
    display.savefig(glm_path / 'plots' / 'keep_qc' /
                    f"{basename}_zmap.png")
    display.close()
    nib.save(
        zmap_cluster,
        glm_path / sub_ses_dir / 'fus' / basename / 'zmap_keep.nii.gz'
    )

# manually fix put bad maps in exclude
# bash:
# exclude () { rm $1; mv ${1/"keep_qc"/"keep"} ${1/"keep_qc"/"exclude"}; }
# exclude FILENAME

# %%
# Plot attributes of excluded datasets
exclude = [p.stem for p in (glm_path / 'plots' / 'exclude').glob('*')]
df['empty_zmap'] = [
    get_pwd_basename(Path(power_doppler_path)) + '_zmap' in exclude
    for power_doppler_path in df['power_doppler_fname']
]
(glm_path / 'plots' / 'empty_zmap').mkdir(parents=True, exist_ok=True)
for col in df.columns[7:-10]:
    ax = sns.countplot(df[df['empty_zmap']], x=col)
    ax.figure.savefig(glm_path / 'plots' / 'empty_zmap' / f'{col}.png')
    plt.close(ax.figure)
