from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed

import nibabel as nib
from scipy import stats
import bwsample as bws

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.filters import threshold_otsu
from skimage.transform import rescale
import anise.utils
import anise.process.image_metrics

root = Path('fUS_data') / 'fUSI_Rat_BIDS'

reg_path = root / 'derivatives' / 'registration'
glm_path = root / 'derivatives' / 'glm'
metric_path = root / 'derivatives' / 'imgmetrics'
metric_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(glm_path / 'experiment_data_proc.csv')

# %%
# Manually label regions of interest
coords = list()
last_click = None
def on_click(event):
    global fig, ax, coords, last_click
    if last_click is None and event.xdata is not None and \
            event.ydata is not None:
        last_click = (event.xdata, event.ydata)
    else:
        coord = last_click + (event.xdata, event.ydata)
        coords.append(coord)
        last_click = None
        ax.plot([coord[0], coord[0], coord[2], coord[2], coord[0]],
                [coord[1], coord[3], coord[3], coord[1], coord[1]])
        fig.canvas.draw()
        print(coords)

# manual ROI
overwrite = False
coord_path = metric_path / 'coords'
coord_path.mkdir(exist_ok=True)
coord_lut = dict()
for i, row in tqdm(df.iterrows(), total=len(df)):
    if row['empty_zmap']:
        continue
    power_doppler_path = Path(row['power_doppler_fname'])
    sub_ses_dir = (power_doppler_path.relative_to(root /
                    'sourcedata')).parent.parent
    pose = anise.utils.get_bids_value(power_doppler_path, 'pose')
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
    out_fname = coord_path / \
        ('_'.join(sub_ses_dir.parts) + f'_pose-{pose}_coords.txt')
    if out_fname not in coord_lut:
        coord_lut[out_fname] = list()
    coord_lut[out_fname].append(this_reg_path / f'{basename}_pwdt.nii.gz')
    if out_fname.exists() and not overwrite:
        continue

    pwd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
    pwd_img_data = np.array(pwd_img.dataobj)

    fig, ax = plt.subplots()
    ax.imshow(pwd_img_data.mean(axis=-1)[:, 0].T, aspect='auto', cmap='hot')
    ax.invert_yaxis()
    fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    np.savetxt(out_fname, coords)
    coords = list()

# %%
# Register to the first image for each setup
for coord_fname, power_doppler_paths in \
        tqdm(coord_lut.items(), total=len(coord_lut)):
    coords = np.loadtxt(coord_fname)
    pwd_img0 = nib.load(power_doppler_paths[0])
    pwd_img_frame0 = np.array(pwd_img0.dataobj)[..., 0]
    np.savetxt(
        coord_path / (power_doppler_paths[0].name.replace(
            '_pwdt.nii.gz', '_coords.txt')), coords)
    for power_doppler_path in power_doppler_paths[1:]:
        coords2 = coords.copy()
        pwd_img = nib.load(power_doppler_path)
        pwd_img_data = np.array(pwd_img.dataobj)
        scale = np.array(pwd_img.shape[:-1]) / np.array(pwd_img0.shape[:-1])
        pwd_img_stack = np.concatenate([
            rescale(pwd_img_frame0, scale)[..., None], pwd_img_data[..., :1]
        ], axis=-1)[:, 0]
        _, trans = anise.utils.register_image_stack(pwd_img_stack)
        coords2[:, 0] = (coords2[:, 0] * scale[0]) + trans[0][1]
        coords2[:, 2] = (coords2[:, 2] * scale[0]) + trans[0][1]
        coords2[:, 1] = (coords2[:, 1] * scale[2]) + trans[1][1]
        coords2[:, 3] = (coords2[:, 3] * scale[2]) + trans[1][1]
        np.savetxt(
            coord_path / (power_doppler_path.name.replace(
                '_pwdt.nii.gz', '_coords.txt')), coords2)


# %%
# Check properly labelled
for coord_fname in tqdm(coord_lut):
    n = len(coord_lut[coord_fname])
    nr = int(round(np.sqrt(n)))
    nc = int(np.ceil(n / nr))
    fig, axs = plt.subplots(nr, nc, figsize=(8, 8))
    axs = axs.flatten()
    for ax, power_doppler_path in zip(axs, coord_lut[coord_fname]):
        pwd_img = nib.load(power_doppler_path)
        pwd_img_data = np.array(pwd_img.dataobj)
        coords = np.loadtxt(
            coord_path / (power_doppler_path.name.replace(
                '_pwdt.nii.gz', '_coords.txt')))
        ax.imshow(pwd_img_data.mean(axis=-1)[:, 0].T,
                  aspect='auto', cmap='hot')
        for x0, y0, x1, y1 in coords:
            ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0])
        ax.invert_yaxis()
    fig.savefig(coord_fname.with_suffix('.png'))
    plt.close(fig)

# %%
# Hyperparameter optimization for metrics
cluster_idxs = np.load(
    glm_path / 'cluster_idxs.npz', allow_pickle=True)['arr_0'].item()
thresh_roi_check = np.arange(5, 210, 5)
thresh_noise_check = np.arange(10, 4000, 100)

def tstat_cor(df, rois, thresh_roi, thresh_noise, n_frames=None):
    vals = list()
    for i, row in df.iterrows():
        power_doppler_path = Path(row['power_doppler_fname'])
        sub_ses_dir = (power_doppler_path.relative_to(root /
                        'sourcedata')).parent.parent
        this_reg_path = reg_path / sub_ses_dir / 'fus'
        basename = '_'.join('.'.join(
            power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
        pd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
        pd_img_data = np.array(pd_img.dataobj)
        if n_frames is not None:
            pd_img_data = pd_img_data[..., :n_frames]
        vals.append(anise.process.image_metrics.cnr(
            pd_img_data,
            rois[power_doppler_path],
            thresh_roi=thresh_roi,
            thresh_noise=thresh_noise
        ))
    r, p = stats.pearsonr(vals, df['max_tstat_rel'])
    return r, p


rois = {Path(power_doppler_fname):
        cluster_idxs[Path(power_doppler_fname)]
        for power_doppler_fname in df['power_doppler_fname']}
out = [Parallel(n_jobs=20)(delayed(tstat_cor)(
       df[~np.isnan(df['max_tstat_rel'])], rois,
       thresh_roi, thresh_noise)
       for thresh_roi in thresh_roi_check)
       for thresh_noise in tqdm(thresh_noise_check)]
rs, ps = np.array(out).transpose(2, 0, 1)


def plot_grid_search(rs, ps, thresh_roi_check,
                     thresh_noise_check):
    best_idx = np.unravel_index(np.argmin(ps), ps.shape)
    print(f'Best params: # ROI voxels: {thresh_roi_check[best_idx[1]]} '
        f'# noise voxels: {thresh_noise_check[best_idx[0]]}, '
        f'r={rs[best_idx].round(4)}')
    fig, ax = plt.subplots()
    im = ax.imshow(rs, aspect='auto')
    ax.invert_yaxis()
    cax = fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(0, thresh_roi_check.size, 5))
    ax.set_xticklabels(thresh_roi_check[::5])
    ax.set_yticks(np.arange(0, thresh_noise_check.size, 5))
    ax.set_yticklabels(thresh_noise_check[::5])
    ax.contour(rs)
    ax.scatter(*best_idx[::-1], s=3, color='red')
    ax.text(*np.array(best_idx[::-1]) * 1.1, f'p={ps[best_idx]:.3e}')
    ax.set_xlabel('# voxels ROI in CNR')
    ax.set_ylabel('# voxels noise in CNR')
    cax.set_label('Pearson r')
    fig.tight_layout()
    return fig


fig = plot_grid_search(rs, ps, thresh_roi_check, thresh_noise_check)
fig.savefig(metric_path / 'plots' / 'cnr_glm_n_voxels.png')
plt.close(fig)

best_idx = np.unravel_index(np.argmin(ps), ps.shape)
thresh_roi = thresh_roi_check[best_idx[1]]
thresh_noise = thresh_noise_check[best_idx[0]]

# %%
# Plot number of frame dependency
n_frames_check = np.arange(1, 101)
out = Parallel(n_jobs=20)(delayed(tstat_cor)(
      df[~np.isnan(df['max_tstat_rel'])], rois,
      thresh_roi, thresh_noise, n_frames)
      for n_frames in n_frames_check)
rs, ps = np.array(out).T

fig, ax = plt.subplots()
ax.plot(n_frames_check, rs)
ax.set_xlabel('# Power Doppler Frames Used')
ax.set_ylabel('Pearson\'s r')
fig.savefig(metric_path / 'plots' / 'cnr_glm_n_frames.png')
plt.close(fig)

# %%
# Show pixels used
best_idx = df.loc[~np.isnan(df['max_tstat_rel']),
                  'max_tstat_rel'].idxmax()
worst_idx = df.loc[~np.isnan(df['max_tstat_rel']),
                   'max_tstat_rel'].idxmin()
for name, idx in dict(best=best_idx, worst=worst_idx).items():
    row = df.loc[idx]
    power_doppler_path = Path(row['power_doppler_fname'])
    sub_ses_dir = (power_doppler_path.relative_to(root /
                    'sourcedata')).parent.parent
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])
    pd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
    pd_img_data = np.array(pd_img.dataobj)
    roi = rois[power_doppler_path]
    img = pd_img_data

    # get indices of ROI and noise
    dmin, dmax = roi[:, -1].min(), roi[:, -1].max()
    roi_idxs = [tuple(idx) for idx in roi]
    roi_pixels = np.array([img[idx] for idx in roi_idxs])
    noise_idxs = np.indices(img.shape[:-1]).reshape(3, -1).T
    noise_idxs = noise_idxs[(noise_idxs[:, -1] >= dmin) &
                            (noise_idxs[:, -1] <= dmax)]
    noise_idxs = [tuple(idx) for idx in noise_idxs
                if tuple(idx) not in roi_idxs]
    noise_pixels = np.array([img[tuple(idx)] for idx in noise_idxs])
    roi_idxs = [roi_idxs[i] for i in
                np.argsort(roi_pixels.mean(axis=1))[-thresh_roi:]]
    noise_idxs = [noise_idxs[i] for i in
                np.argsort(noise_pixels.mean(axis=1))[:thresh_noise]]
    roi_noise_img = np.zeros(img.shape[:-1]) * np.nan
    for roi_idx in roi_idxs:
        roi_noise_img[roi_idx] = 100
    for noise_idx in noise_idxs:
        roi_noise_img[noise_idx] = -100

    fig, ax = plt.subplots()
    ax.imshow(pd_img_data[:, 0].mean(axis=-1).T, aspect='auto', cmap='hot')
    ax.imshow(roi_noise_img[:, 0].T, aspect='auto', cmap='brg_r')
    ax.invert_yaxis()
    fig.savefig(metric_path / 'plots' / f'{name}_cnr_voxels.png')
    plt.close(fig)

# %%
# Manual ROI
coords = np.loadtxt(list(coord_lut.keys())[0])
rois = {i: dict() for i in range(coords.shape[0])}
for _, row in tqdm(df.iterrows(), total=len(df)):
    coords = np.loadtxt(
        coord_path / (Path(row['power_doppler_fname']).name.replace(
            '_pwdt.nii.gz', '_coords.txt')))
    pwd_img = nib.load(row['power_doppler_fname'])
    img_idxs = np.indices(pwd_img.shape[:-1]).transpose((1, 2, 3, 0))
    for i, coord in enumerate(coords):
        coord = coord.round().astype(int)
        x0, x1 = min([coord[0], coord[2]]), max([coord[0], coord[2]])
        y0, y1 = min([coord[1], coord[3]]), max([coord[1], coord[3]])
        rois[i][Path(row['power_doppler_fname'])] = \
            img_idxs[x0:x1, :, y0:y1].reshape(-1, 3)


thresh_roi_check = np.arange(5, 210, 5)
thresh_noise_check = np.arange(10, 4000, 100)
rs_ps_all = dict()
for i, _ in enumerate(tqdm(coords)):
    out = [Parallel(n_jobs=20)(delayed(tstat_cor)(
       df[~np.isnan(df['max_tstat_rel'])], rois[i],
       thresh_roi, thresh_check)
       for thresh_roi in thresh_roi_check)
       for thresh_check in tqdm(thresh_noise_check)]
    rs, ps = np.array(out).transpose(2, 0, 1)
    rs_ps_all[i] = rs, ps
    fig = plot_grid_search(rs, ps, thresh_roi_check, thresh_noise_check)
    fig.savefig(metric_path / 'plots' / f'cnr_roi-{i}_n_voxels.png')
    plt.close(fig)


# %%
# Summary plot
row = df.loc[0]
coords = np.loadtxt(
    coord_path / (Path(row['power_doppler_fname']).name.replace(
        '_pwdt.nii.gz', '_coords.txt')))
pwd_img = nib.load(row['power_doppler_fname'])
pwd_img_data = np.array(pwd_img.dataobj)

fig, ax = plt.subplots()
ax.imshow(pwd_img_data.mean(axis=-1)[:, 0].T, aspect='auto', cmap='hot')
ax.invert_yaxis()
for i, coord in enumerate(coords):
    ax.plot([coord[0], coord[0], coord[2], coord[2], coord[0]],
            [coord[1], coord[3], coord[3], coord[1], coord[1]])
    rs, ps = rs_ps_all[i]
    best_idx = np.unravel_index(np.argmin(ps), ps.shape)
    max_r = rs[best_idx]
    ax.text((coord[0] + coord[2]) / 2, (coord[1] + coord[3]) / 2,
            f'{i} max r=\n{max_r.round(3)}',
            color='white', ha='center', va='center')
fig.savefig(metric_path / 'plots' / f'cnr_roi_summary.png')
plt.close(fig)

# %%
# Whole image
thresh_roi_check = np.arange(5, 210, 5)
thresh_noise_check = np.arange(10, 50000, 500)

rois = {Path(pwd_fname): None for pwd_fname in
        df['power_doppler_fname']}
out = [Parallel(n_jobs=20)(delayed(tstat_cor)(
       df[~np.isnan(df['max_tstat_rel'])], rois,
       thresh_roi, thresh_noise)
       for thresh_roi in thresh_roi_check)
       for thresh_noise in tqdm(thresh_noise_check)]
rs, ps = np.array(out).transpose(2, 0, 1)

fig = plot_grid_search(rs, ps, thresh_roi_check, thresh_noise_check)
fig.savefig(metric_path / 'plots' / 'cnr_full_n_voxels.png')
plt.close(fig)

best_idx = np.unravel_index(np.argmin(ps), ps.shape)
thresh_roi_full = thresh_roi_check[best_idx[1]]
thresh_noise_full = thresh_noise_check[best_idx[0]]

# %%
# Compute metrics
for i, row in tqdm(df.iterrows(), total=len(df)):
    power_doppler_path = Path(row['power_doppler_fname'])
    sub_ses_dir = (power_doppler_path.relative_to(root /
                    'sourcedata')).parent.parent
    this_reg_path = reg_path / sub_ses_dir / 'fus'
    basename = '_'.join('.'.join(
        power_doppler_path.stem.split('.')[:-1]).split('_')[:-1])

    pd_img = nib.load(this_reg_path / f'{basename}_pwdt.nii.gz')
    pd_img_data = np.array(pd_img.dataobj)

    cluster_idx = cluster_idxs[power_doppler_path]
    df.loc[i, 'cnr_glm'] = anise.process.image_metrics.cnr(
        pd_img_data, cluster_idx, thresh_roi=thresh_roi,
        thresh_noise=thresh_noise
    )
    df.loc[i, 'cnr_full'] = anise.process.image_metrics.cnr(
        pd_img_data, cluster_idx, thresh_roi=thresh_roi_full,
        thresh_noise=thresh_noise_full
    )
    df.loc[i, 'neighbor_coherence'] = \
        anise.process.image_metrics.neighbor_coherence(
            pd_img_data[:, 0]
        )
    df.loc[i, 'resolution'] = pd_img_data[..., 0].size

df.to_csv(metric_path / 'experiment_data.csv', index=False)

# %%
# Plot regression with max t-statistic

(metric_path / 'plots').mkdir(exist_ok=True)

def annotate(data, **kws):
    r, p = stats.pearsonr(
        data[kws['kwargs']['x']], data[kws['kwargs']['y']])
    ax = plt.gca()
    ax.text(0.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)
    
for col in df.columns[-4:]:
    ax = sns.lmplot(df[~np.isnan(df['max_tstat_rel'])], x=col, y='max_tstat_rel')
    ax.map_dataframe(annotate, kwargs=dict(x=col, y='max_tstat_rel'))
    for ext in ('png', 'eps'):
        ax.figure.savefig(metric_path / 'plots' / 
                          f'{col}_maxtstat_reg.{ext}', dpi=300)
    plt.close(ax.figure)

# %%
# Compare with expert ratings

for mode in ('bmode', 'power_doppler'):
    mode_str = mode.replace('_', '')
    rating_fnames = (metric_path / 'ratings').glob(
        f'*mode-{mode_str}*.tsv')
    for rating_fname in rating_fnames:
        df_raw = pd.read_csv(rating_fname, sep='\t')
        evaluations = list()
        for _, row in df_raw.iterrows():
            images = row['images'].split(',')
            ratings = [0 for _ in range(len(images))]
            ratings[int(row['best'])] = 1
            ratings[int(row['worst'])] = 2
            evaluations.append((ratings, images))
        agg_dok, direct_dok, direct_detail, logical_dok, logical_detail = bws.count(
            evaluations)
        ranked, ordids, metrics, scores, info = bws.rank(
            agg_dok, method='ratio', adjust='quantile')

        ordids = list(ordids)
        for i, row in df.iterrows():
            img_fname = Path(row['power_doppler_img_fname']).name
            df.loc[i, 'imrate_score'] = scores[ordids.index(img_fname)]

        ax = sns.lmplot(df.dropna(),
                        x="imrate_score", y='max_tstat_rel')
        ax.map_dataframe(annotate, kwargs=dict(x="imrate_score",
                                               y='max_tstat_rel'))
        for ext in ('png', 'eps'):
            ax.figure.savefig(
                metric_path / 'plots' / 
                f'{rating_fname.stem}_maxtstat_reg.{ext}', dpi=300)
        plt.close(ax.figure)

        print(rating_fname, ordids, scores)

# %%
# Display table as html
df = pd.read_csv(metric_path / 'experiment_data.csv')
df = df.drop([col for col in df.columns if '0' in col or col == 'index'],
             axis='columns')  # cruft
df = df.drop(['metadata', 'shifts', 'max_tstats'], axis='columns')
for col in [col for col in df.columns if 'img_fname' in col]:
    df[col] = [
        f'<img src="{Path(img_fname).relative_to(root)}" width="400">'
        for img_fname in df[col]
    ]
df.to_html(root / 'index.html', escape=False)
