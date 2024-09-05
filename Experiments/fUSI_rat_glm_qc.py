from pathlib import Path
from shutil import copyfile, rmtree
from tqdm import tqdm
import numpy as np
import pandas as pd
import json

import nibabel as nib
from skimage.measure import label
from skimage.morphology import binary_dilation


import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting, image
from nilearn.image import mean_img

import anise.process.fusi_glm_fit2
import anise.process.image_metrics

root = Path('fUS_data') / 'fUSI_Rat_BIDS'

reg_path = root / 'derivatives' / 'registration'

df = pd.read_csv(glm_path / 'experiment_data.csv')

glm_path = root / 'derivatives' / 'glm'
glm_path.mkdir(parents=True, exist_ok=True)

# %%
# Cluster zmaps
thresh = 3
n_clusters_keep = 2
min_cluster_size = 15
df_zmap = pd.DataFrame(
    columns=['power_doppler_path', 'n_clusters',
             'largest_cluster', 'average_cluster_size',
             'largest_clusters_max_z']
)
largest_cluster_idxs = dict()
for i, row in tqdm(df.iterrows(), total=len(df)):
    pwd_path = Path(row['power_doppler_fname'])
    sub_ses_dir = pwd_path.relative_to(root / 'sourcedata').parent.parent
    basename = '_'.join('.'.join(
        pwd_path.stem.split('.')[:-1]).split('_')[:-1])
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    this_glm_path = glm_path / sub_ses_dir / 'fus'
    zmap = nib.load(this_glm_path / basename / 'best_zmap.nii.gz')
    zmap_data = np.array(zmap.dataobj)
    zmap_data_proc = np.abs(zmap_data) > thresh
    
    # cluster
    labels = label(zmap_data_proc)
    idxs = np.unique(labels)
    idxs = idxs[idxs > 0]  # 0 is background
    if idxs.size == 0:
        df_zmap.loc[len(df_zmap.index)] = pwd_path, 0, 0, 0, 0
        continue

    sizes = np.array([np.sum(labels == idx) for idx in idxs])
    n_clusters, largest_cluster, average_cluster_size = \
        sizes.size, np.max(sizes), np.mean(sizes)
    i_sorted = np.argsort(sizes)
    largest_clusters = [
        np.array(
            np.where(
                binary_dilation(  # close holes
                    labels == idxs[i_sorted[-j]],
                    np.ones((3,) * zmap_data.ndim)
                )
            )
        ).T
        for j in range(1, n_clusters_keep + 1)
    ]

    largest_clusters_combo = np.concatenate(largest_clusters, axis=0)
    largest_clusters_max_z = \
        np.max(np.abs([zmap_data[tuple(idx)] for idx in
                       largest_clusters_combo]))
    largest_cluster_idxs[pwd_path] = largest_clusters
        
    df_zmap.loc[len(df_zmap.index)] = \
        (pwd_path, n_clusters, largest_cluster,
         average_cluster_size, largest_clusters_max_z)

df['max_zmap2'] = df_zmap['largest_clusters_max_z']
df['largest_cluster'] = df_zmap['largest_cluster']
np.savez_compressed(glm_path / 'cluster_idxs.npz', largest_cluster_idxs)

# %%
# Plot histogram distributionns
(glm_path / "plots").mkdir(exist_ok=True)
for col in df_zmap.columns[1:]:
    ax = sns.displot(df_zmap, x=col)
    ax.figure.savefig(glm_path / "plots" / f"{col}.png")
    plt.close(ax.figure)

# %%
# Copy over zmaps, make cluster zmaps

def mask_zmap_by_clusters(pwd_img, zmap, clusters, n=None):
    zmap_data_orig = np.array(zmap.dataobj)
    zmap_data = np.zeros_like(zmap_data_orig)
    for i, cluster in enumerate(clusters):
        if n is None or i < n:
            for idx in cluster:
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
    display.savefig(zmap_dir / f"{basename}_zmap_keep.png")
    display.close()
    nib.save(
        zmap_cluster,
        glm_path / sub_ses_dir / 'fus' / basename / 'zmap_keep.nii.gz'
    )


zmap_dir = glm_path / "plots" / "zmaps"
zmap_dir.mkdir(exist_ok=True)
for i, row in tqdm(df_zmap.iterrows(), total=len(df)):
    pwd_path = Path(row['power_doppler_path'])
    sub_ses_dir = pwd_path.relative_to(root /  'sourcedata').parent.parent
    basename = '_'.join('.'.join(
        pwd_path.stem.split('.')[:-1]).split('_')[:-1])
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    this_glm_path = glm_path / sub_ses_dir / 'fus'
    # copy over zmaps
    copyfile(this_glm_path / basename / 'best_zmap.png',
             zmap_dir / f'{basename}_zmap.png')
    # make clustered zmaps
    pwd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
    zmap = nib.load(this_glm_path / basename / 'best_zmap.nii.gz')
    mask_zmap_by_clusters(pwd_img, zmap, largest_cluster_idxs[pwd_path])

# %%
# Manually QC zmaps
rule_func = lambda row: row['largest_cluster'] > min_cluster_size
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
fig.show()

empty_zmap = list()
modified = list()
progress_bar = tqdm(range(len(df_zmap)))


def show_current_zmap(i, axs):
    row = df_zmap.loc[i]
    for ax in axs:
        ax.cla()
    pwd_path = Path(row['power_doppler_path'])
    sub_ses_dir = (pwd_path.relative_to(root /
                    'sourcedata')).parent.parent
    basename = '_'.join('.'.join(
        pwd_path.stem.split('.')[:-1]).split('_')[:-1])
    fig.suptitle('Auto=' + ('keep' if rule_func(row) else 'exclude'))
    axs[0].imshow(plt.imread(zmap_dir / f"{basename}_zmap.png"))
    axs[1].imshow(plt.imread(zmap_dir / f"{basename}_zmap_keep.png"))
    fig.canvas.draw()


def key_press_event(event):
    if len(empty_zmap) == len(df_zmap)
        plt.close(fig)
        return
    empty_zmap.append(event.key == 'e')
    if event.key == 1:  # remove extraneous cluster
        modified.append(progress_bar.n)
        row = df_zmap.loc[progress_bar.n]
        pwd_path = Path(row['power_doppler_path'])
        sub_ses_dir = (pwd_path.relative_to(root /
                        'sourcedata')).parent.parent
        basename = '_'.join('.'.join(
            pwd_path.stem.split('.')[:-1]).split('_')[:-1])
        this_reg_path = reg_path / sub_ses_dir / 'fus'
        this_glm_path = glm_path / sub_ses_dir / 'fus'
        pwd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
        zmap = nib.load(this_glm_path / basename / 'best_zmap.nii.gz')
        mask_zmap_by_clusters(pwd_img, zmap, largest_cluster_idxs[pwd_path], n=1)
    progress_bar.update(1)
    progress_bar.refresh()
    show_current_zmap(progress_bar.n, axs)

fig.canvas.mpl_connect('key_press_event', key_press_event)
show_current_zmap(0, axs)

fig, axs = plt.subplots(1, 2, figsize=(8, 6))
fig.show()
for i in modified:
    show_current_zmap(i, axs)
    if input('Exclude?\t') == 'e':
        empty_zmap[i] = True

n_kept = len(df) - np.sum(empty_zmap)
print(f'{np.round(100 * n_kept / len(df), 2)}% active zmaps kept '
      f'({n_kept} / {len(df)})')

# %%
# Plot attributes of excluded datasets
df['empty_zmap'] = empty_zmap
df.to_csv(glm_path / 'experiment_data_proc.csv', index=False)

(glm_path / 'plots' / 'empty_zmap').mkdir(parents=True, exist_ok=True)
for col in df.columns[7:-10]:
    ax = sns.countplot(df[df['empty_zmap']], x=col)
    ax.figure.savefig(glm_path / 'plots' / 'empty_zmap' / f'{col}.png')
    plt.close(ax.figure)
