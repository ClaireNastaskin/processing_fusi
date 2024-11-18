from pathlib import Path
import numpy as np
import io
from contextlib import redirect_stdout
from tqdm import tqdm
import json
import jax
import re
from datetime import datetime
import h5py
import pandas as pd
from joblib import Parallel, delayed

# raw to schema
from mangrove.io.metadata import PoseidonMetadata
from mangrove.io.convert.raw_data_to_schema import metadata_to_schema, preprocess_raw_data
from mangrove.beamform.preprocess import preprocess_iq_data_to_rf

# beamforming
from mangrove.beamform.vbeam_ import das_beamformer
from mangrove.io.convert.vbeam.mangrove_to_vbeam import \
    import_space_time_to_vbeam_setup, _time_beamform
from mangrove.schema.wrapper.bmode_wrapper import BModeFile

# power doppler
from mangrove.power_doppler.pca_metal import PCAMetalPowerDoppler

# registration
from dipy.align import affine_registration
# from quaternion import from_rotation_matrix, as_float_array
from mne.transforms import _affine_to_quat

# GLM
from anise.utils import register_image_stack
from anise.process.fusi_glm_fit2 import fit_glm_time_shift
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.image import mean_img

# plotting
from nilearn import plotting
import skimage.measure
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import BlendedGenericTransform

event_offset = 0
smoothing_fwhm = 0.3
overwrite = False

base_paths = [
    Path('fUS_data/UCLA/data/UCLA_008/10-24-2024/Functional_runs/run-02'),
    Path('fUS_data/UCLA/data/UCLA_008/10-24-2024/Functional_runs/run-03'),
    Path('fUS_data/UCLA/data/UCLA_008/10-24-2024/Functional_runs/run-04')
]


def decode_event_item(item):
    if isinstance(item, bytes):
        item = item.decode('utf-8')
    if isinstance(item, str) and '{' in item and '}' in item:
        item = json.loads(item)
    return item


events_all = {}
pwd_all = {}
time_stamps_all = {}
for base_path in base_paths:
    events_all[base_path.stem] = {}
    pwd_all[base_path.stem] = {}
    time_stamps_all[base_path.stem] = {}
    for experiment_folder in (base_path / 'acquisitions').glob('seq-*'):
        # %%
        # Load metadata
        metadata = PoseidonMetadata.from_folder(experiment_folder / 'metadata')
        acquisition_sequence_metadata, transducer_metadata = metadata_to_schema(metadata)
        events_raw = pd.DataFrame(columns=['name', 'stimulus', 'time'])
        for event in h5py.File(base_path / 'streams' / 'task-event_stream.h5')['data'][:]:
            items = list()
            for item, col in zip(event, events_raw.columns):
                item = decode_event_item(item)
                if isinstance(item, dict) and col in item:
                    if 'task' in item:
                        item = item['task']  # weird structure for movement
                    else:
                        item = item[col]
                items.append(item)
            events_raw.loc[len(events_raw.index)] = items

        events = pd.DataFrame(columns=['onset', 'trial_type', 'duration'])
        for i, row in events_raw.iterrows():
            if i % 2:
                continue
            row2 = events_raw.iloc[i + 1]
            assert 'start' in row['name'] or 'on' in row['name']
            assert 'stop' in row2['name'] or 'off' in row2['name']
            assert row['stimulus'] == row2['stimulus']
            duration = row2['time'] - row['time']
            trial_type = row['stimulus'].split('.')[0]
            events.loc[len(events.index)] = row['time'], trial_type, duration

        events_all[base_path.stem][experiment_folder.stem] = events

        pwd = nib.load(experiment_folder / 'power_doppler' / 'pwd.nii.gz')
        pwd_all[base_path.stem][experiment_folder.stem] = pwd
        time_stamps = np.loadtxt(experiment_folder / 'power_doppler' / 'pwd_timestamps.txt')
        time_stamps_all[base_path.stem][experiment_folder.stem] = time_stamps


# concatenate per sequence
pwds_seq = {}
events_seq = {}
time_stamps_seq = {}
for sequence in list(list(pwd_all.values())[0]):
    pwds_seq[sequence] = nib.concat_images(
        [pwd_all[base_path][sequence] for base_path in pwd_all],
        axis=-1)
    events_seq[sequence] = []
    time_stamps_seq[sequence] = []
    start_time = 0
    for base_path in time_stamps_all:
        time_stamps_seq[sequence].append(
            time_stamps_all[base_path][sequence] + start_time)
        events_all[base_path][sequence]['onset'] += start_time
        events_seq[sequence].append(events_all[base_path][sequence])
        # for probe crashes, behavior events happen with no data so
        # shift time stamps so by whichever is longer (behavior)
        # so that there is no overlap
        start_time += np.max([
            time_stamps_all[base_path][sequence][-1],
            events_all[base_path][sequence]['onset'].iloc[-1] +
            events_all[base_path][sequence]['duration'].iloc[-1]
        ]) + np.diff(time_stamps_all[base_path][sequence]).mean()
    events_seq[sequence] = pd.concat(
        events_seq[sequence]
    ).reset_index().drop(['index'], axis='columns')
    time_stamps_seq[sequence] = np.concatenate(time_stamps_seq[sequence])
    # events_seq[sequence] = events_seq[sequence][events_seq[sequence]['onset']]


for sequence in pwds_seq:
    pwd = pwds_seq[sequence]
    events = events_seq[sequence]
    time_stamps = time_stamps_seq[sequence]
    experiment_folder = base_paths[0].parent / \
        '+'.join([base_path.stem for base_path in base_paths]) / sequence
    experiment_folder.mkdir(parents=True, exist_ok=True)
    # registration
    (experiment_folder / 'reg').mkdir(parents=True, exist_ok=True)
    if (experiment_folder / 'reg' / 'pwd.nii.gz').exists() and not overwrite:
        pwd_reg = nib.load(experiment_folder / 'reg' / 'pwd.nii.gz')
        transformations = np.loadtxt(
            experiment_folder / 'reg' / 'transformations.txt'
        )
    else:
        if pwd.shape[1] < 15:
            pwd_reg, transformations = register_image_stack(
                np.array(pwd.dataobj)[:, pwd.shape[1] // 2]
            )
            pwd_reg = nib.Nifti1Image(pwd_reg[:, None], pwd.affine)
            transformations = transformations.T
            trans_labels = ['tx', 'ty', 'rot']
        else:
            pwd_data = np.array(pwd.dataobj).transpose(3, 0, 1, 2)
            static = pwd_data[0]
            pwd_reg = np.zeros_like(pwd_data)
            pwd_reg[0] = static
            reg_affines = np.zeros((pwd_data.shape[0], 4, 4))
            reg_affines[0] = np.eye(4)

            def register_image(moving):
                log = io.StringIO()
                with redirect_stdout(log):
                    moved, reg_affine = affine_registration(
                        moving,
                        static,
                        moving_affine=pwd.affine,
                        static_affine=pwd.affine,
                        pipeline=['rigid'],
                    )
                return moved, reg_affine

            out = Parallel(n_jobs=-5)(
                delayed(register_image)(moving) for moving in tqdm(pwd_data[1:])
            )
            for i, (moved, reg_affine) in enumerate(out):
                pwd_reg[i + 1] = moved
                reg_affines[i + 1] = reg_affine
            """
            transformations = np.concatenate(
                [as_float_array(from_rotation_matrix(reg_affines[..., :3, :3])),
                 reg_affines[..., :3, 3]],
                axis=-1,
            )  # convert to quaternions
            """
            transformations = _affine_to_quat(reg_affines)
            pwd_reg = nib.Nifti1Image(pwd_reg.transpose(1, 2, 3, 0), pwd.affine)
            trans_labels = ["rx", "ry", "rz", "tx", "ty", "tz"]
        nib.save(pwd_reg, experiment_folder / 'reg' / 'pwd.nii.gz')
        np.savetxt(experiment_folder / 'reg' / 'transformations.txt', transformations)

        # plot
        fig, ax = plt.subplots()
        trans_labels_trans = [label for label in trans_labels if label.startswith('t')]
        # TO DO: this scales elevation by depth for 2D since elevation is skipped
        ax.plot(np.array([transformations[:, trans_labels.index(label)] *
                          pwd_reg.affine[trans_labels_trans.index(label),
                                         trans_labels_trans.index(label)]
                          for label in trans_labels_trans]).T,
                label=trans_labels_trans)
        ax.set_xlabel('Frame #')
        ax.set_ylabel('Displacement (mm)')
        ax.set_ylim([-10, 10])
        ax2 = ax.twinx()
        for _ in range(len(trans_labels_trans)):
            ax2.plot([])  # get on same color cycle
        trans_labels_rot = [label for label in trans_labels if label.startswith('r')]
        ax2.plot(np.array([np.rad2deg(transformations[:, trans_labels.index(label)])
                           for label in trans_labels_rot]).T,
                 label=trans_labels_rot)
        ax2.set_ylim([-10, 10])
        ax2.set_ylabel('Displacement (degrees)')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(experiment_folder / 'reg' / 'movement.png')
        plt.close(fig)

    # define events of interest
    for comp in ('main', 'both'):
        if comp == 'main':
            event = min(set(events.trial_type))
            events2 = events[events['trial_type'] == event]
        else:
            event = 'both'
            events2 = events.copy()
            events2['trial_type'] = 'both'

        # glm
        (experiment_folder / 'glm' / comp).mkdir(parents=True, exist_ok=True)
        if not (experiment_folder / 'glm' / comp / 'shifts.png').exists() or overwrite:
            shifts, max_tstats, best_offset = fit_glm_time_shift(
                pwd_reg, time_stamps,
                event, events2, shift=20,
                smoothing_fwhm=smoothing_fwhm, hrf='glover',
                transformations=transformations,
                out_dir=experiment_folder / 'glm' / comp, n_jobs=-5
            )
            fig, ax = plt.subplots()
            ax.plot(shifts, max_tstats)
            ax.set_xlabel('shifts')
            ax.set_ylabel('max_z')
            fig.savefig(experiment_folder / 'glm' / comp / 'shifts.png')
            plt.close(fig)

        if (experiment_folder / 'glm' / comp / 'pwd_zmap.nii.gz').exists() and not overwrite:
            zmap = nib.load(experiment_folder / 'glm' / comp / 'pwd_zmap.nii.gz')
            mean_pwd = nib.load(experiment_folder / 'glm' / comp / 'pwd_mean.nii.gz')
        else:
            mean_pwd = mean_img(pwd_reg)
            nib.save(mean_pwd, experiment_folder / 'glm' / comp / 'pwd_mean.nii.gz')
            events_shifted = events2.copy()
            events_shifted['onset'] += event_offset
            design_matrix = make_first_level_design_matrix(
                time_stamps,
                events_shifted,
                drift_model="polynomial",
                drift_order=1,
                hrf_model='glover'
            )
            glm = FirstLevelModel(minimize_memory=False, mask_img=False,
                                  smoothing_fwhm=smoothing_fwhm, standardize=True).fit(
                pwd_reg, design_matrices=design_matrix
            )
            zmap = glm.compute_contrast(
                event,
                output_type="stat"
            )
            nib.save(zmap, experiment_folder / 'glm' / comp / 'pwd_zmap.nii.gz')
            nib.save(nib.Nifti1Image(
                (np.abs(np.array(zmap.dataobj)) > 3).astype(np.float32),
                zmap.affine),
                experiment_folder / 'glm' / comp / 'pwd_zmap_mask.nii.gz'
            )

            display = plotting.plot_stat_map(
                zmap,
                bg_img=mean_pwd,
                cut_coords=[zmap.shape[1] // 2],
                threshold=3,
                display_mode="y",
                black_bg=True,
                title='audio',
            )
            display.savefig(experiment_folder / 'glm' / comp / 'pwd_zmap.png')

        # time course
        slice_idxs = [1] if min(pwd_reg.shape) == 1 else [0, 1, 2]
        zmap_data = np.array(zmap.dataobj)
        zmap_data[np.abs(zmap_data) < 3] = np.nan
        clusters = skimage.measure.label(~np.isnan(zmap_data))
        pwd_data = np.array(pwd_reg.dataobj).copy()
        pwd_data -= pwd_data.mean(axis=-1, keepdims=True)
        pwd_data = np.divide(pwd_data, pwd_data.std(axis=-1, keepdims=True),
                             where=pwd_data.std(axis=-1, keepdims=True) != 0)

        roi_idx = 0
        while roi_idx < np.nanmax(clusters):
            ext_idx = tuple(np.unravel_index(np.nanargmax(zmap_data), zmap_data.shape))
            ext_idx_show = tuple(np.mean(np.array(np.where(
                clusters == clusters[ext_idx])), axis=1
            ).round().astype(int))
            fig = plt.figure(figsize=(6, 6))
            fig.suptitle(ext_idx)
            gs = GridSpec(3, 2, figure=fig)
            if len(slice_idxs) == 1:
                axs = [fig.add_subplot(gs[:2, :2])]
            else:
                axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
                axs[-1].axis('off')
            for slice_idx, i in enumerate(slice_idxs):
                ax = axs[slice_idx]
                idx = tuple([slice(None)] * i + [ext_idx_show[i]])
                ax.imshow(np.array(pwd_reg.dataobj).mean(axis=-1)[idx].T, cmap='gray', aspect='auto')
                ax.imshow(zmap_data[idx].T, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
                ext_idx_2d = np.delete(ext_idx, i).T
                ax.axvline(ext_idx_2d[0], color='red', linewidth=0.25)
                ax.axhline(ext_idx_2d[1], color='red', linewidth=0.25)
                ax.axis('off')
            ax = fig.add_subplot(gs[-1, :])
            thresh = 10
            pwd_data_nans = np.insert(
                pwd_data[ext_idx], np.where(np.diff(time_stamps) > thresh)[0] + 1, np.nan)
            time_stamps_nans = np.insert(
                time_stamps, np.where(np.diff(time_stamps) > thresh)[0] + 1, np.nan)
            ax.plot(time_stamps_nans, pwd_data_nans)
            ymax = np.min([10, np.nanmax(np.abs(pwd_data[tuple(ext_idx)]))])
            ax.set_ylim([-ymax, ymax])
            for _, (onset, trial_type, duration) in events.iterrows():
                ax.axvspan(onset, onset + duration, color='gray', alpha=0.25)
                ax.text(onset + duration / 2, 0.95, trial_type.replace('_', '\n'),
                        ha='center', va='center',
                        transform=BlendedGenericTransform(ax.transData, ax.transAxes))
            fig.savefig(experiment_folder / 'glm' / comp / f'pwd_tc{roi_idx}.png')
            plt.close(fig)
            zmap_data[clusters == clusters[ext_idx]] = np.nan
            roi_idx += 1
