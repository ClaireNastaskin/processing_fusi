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

experiment_folders = list(Path('fUS_data/UCLA/data/UCLA_008').rglob('*raw_frame_data')) + \
    list(Path('fUS_data/UCLA/data/UCLA_009').rglob('*raw_frame_data'))
experiment_folders = [experiment_folder.parent for experiment_folder
                      in experiment_folders
                      if 'functional' in str(experiment_folder).lower()]

"""
base_path = Path('fUS_data/UCLA/data/UCLA_009/Functional_runs/run-01')
sequence = 'seq-IQ_3D_2cmto3cm_Depth_5MHz_1a_3c_200loops_-1Gain_2rp'  # 'seq-2d_plane_wave'
"""


def decode_event_item(item):
    if isinstance(item, bytes):
        item = item.decode('utf-8')
    if isinstance(item, str) and '{' in item and '}' in item:
        item = json.loads(item)
    return item


def raw_to_power_doppler(experiment_folder, metadata, transducer_metadata):
    """Convert raw, bit-packed data to power doppler."""
    pwd_files = list((experiment_folder / "power_doppler").glob("[!.]*.h5"))
    if pwd_files:   # pre-computed
        pwd_frames = dict()
        for pwd_file in tqdm(pwd_files):
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}',
                              pwd_file.stem)
            timestamp = datetime.strptime(
                match.group(0).replace('-', ':'), '%Y:%m:%dT%H:%M:%S:%f')
            pwd_h5 = h5py.File(pwd_file)
            pwd_frames[timestamp] = pwd_h5['power_doppler'][:]
        affine = np.diag([
            1000 * (pwd_h5['lateral'][1] - pwd_h5['lateral'][0]),
            1000 * (pwd_h5['elevation'][1] - pwd_h5['elevation'][0]) if
            pwd_h5['elevation'].size > 1 else
            (pwd_h5['lateral'][1] - pwd_h5['lateral'][0]),
            1000 * (pwd_h5['depth'][1] - pwd_h5['depth'][0]),
            1
        ])
        elevation_idx = 0 if pwd_frames[timestamp].shape[1] < 15 else \
            pwd_frames[timestamp].shape[1] // 2
        affine[:3, 3] = [
            pwd_h5['lateral'][0],
            pwd_h5['elevation'][elevation_idx],
            pwd_h5['depth'][0]
        ]

    """ # partially pre-computed, don't use too slow
    bmode_files = list((experiment_folder / "beamformed").glob("[!.]*.h5"))
    elif bmode_files:
        pwd_frames = dict()
        for bmode_file in tqdm(bmode_files):
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}',
                              bmode_file.stem)
            timestamp = datetime.strptime(
                match.group(0).replace('-', ':'), '%Y:%m:%dT%H:%M:%S:%f')
            beamformed = h5py.File(bmode_file)['beamformed'][:]
            pwd_frame = PCAMetalPowerDoppler(
                num_tissue_components=50,
                num_blood_and_tissue_components=None
            ).fit_transform(
                beamformed.transpose(1, 2, 3, 0)
            )
            del beamformed
            pwd_frames[timestamp] = pwd_frame
        bmode_h5 = h5py.File(bmode_file)
        affine = np.diag([
            1000 * (bmode_h5['lateral'][1] - bmode_h5['lateral'][0]),
            1000 * (bmode_h5['elevation'][1] - bmode_h5['elevation'][0]) if
            bmode_h5['elevation'].size > 1 else
            (bmode_h5['lateral'][1] - bmode_h5['lateral'][0]),
            1000 * (bmode_h5['depth'][1] - bmode_h5['depth'][0]),
            1
        ])
        elevation_idx = 0 if pwd_frames[timestamp].shape[1] < 15 else \
            pwd_frames[timestamp].shape[1] // 2
        affine[:3, 3] = [
            bmode_h5['lateral'][0],
            bmode_h5['elevation'][elevation_idx],
            bmode_h5['depth'][0]
        ]
    """
    if not pwd_files:
        raw_data_files = list((experiment_folder / "raw_frame_data").glob("[!.]*.h5"))
        pwd_frames = dict()
        for raw_data_file in tqdm(raw_data_files):
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', raw_data_file.stem)
            timestamp = datetime.strptime(match.group(0).replace('-', ':'), '%Y:%m:%dT%H:%M:%S:%f')
            # %%
            # Preprocess
            data, scan_timing_metadata = preprocess_raw_data(
                metadata, raw_data_file, transducer_metadata=transducer_metadata
            )
            # %%
            # vbeam
            vbeam_setup = import_space_time_to_vbeam_setup(
                rf_or_iq_data=data,
                acquisition_sequence_metadata=acquisition_sequence_metadata,
                speed_of_sound_m_s=1540.0,
            )
            del data
            vbeam_setup_slice = vbeam_setup.slice["frames", 0]
            beamformer = das_beamformer(
                setup=vbeam_setup_slice,
                ram_limit_gb=8.0,
            )
            beamformer = jax.jit(beamformer)
            beamformed = np.array(
                _time_beamform(vbeam_setup=vbeam_setup, beamformer=beamformer)
            )

            # %%
            # Power doppler
            pwd_frame = PCAMetalPowerDoppler(
                num_tissue_components=50,
                num_blood_and_tissue_components=None
            ).fit_transform(
                beamformed.transpose(1, 2, 3, 0)
            )
            del beamformed

            pwd_frames[timestamp] = pwd_frame
        affine = np.diag([
            1000 * (vbeam_setup.scan.x[1] - vbeam_setup.scan.x[0]),
            1000 * (vbeam_setup.scan.y[1] - vbeam_setup.scan.y[0]),
            1000 * (vbeam_setup.scan.z[1] - vbeam_setup.scan.z[0]),
            1
        ])
        elevation_idx = 0 if pwd_frames[timestamp].shape[1] < 15 else \
            pwd_frames[timestamp].shape[1] // 2
        affine[:3, 3] = [
            vbeam_setup.scan.x[0],
            vbeam_setup.scan.y[elevation_idx],
            vbeam_setup.scan.z[0]
        ]

    time_stamps = sorted(pwd_frames)

    pwd = nib.Nifti1Image(
        np.array([pwd_frames.get(timestamp)
                  for timestamp in time_stamps]).transpose(1, 2, 3, 0),
        affine
    )
    del pwd_frames
    time_stamps = np.array([(timestamp - time_stamps[0]).total_seconds()
                            for timestamp in time_stamps])
    return pwd, time_stamps


for experiment_folder in tqdm(experiment_folders):
    base_path = experiment_folder.parent.parent
    sequence = experiment_folder.stem

    try:
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

        # %%
        # Pipeline 1: raw -> power doppler

        (experiment_folder / 'power_doppler').mkdir(parents=True, exist_ok=True)
        if not (experiment_folder / 'power_doppler' / 'pwd.nii.gz').exists() or overwrite:
            pwd, time_stamps = raw_to_power_doppler(
                experiment_folder,
                metadata,
                transducer_metadata
            )
            nib.save(pwd, experiment_folder / 'power_doppler' / 'pwd.nii.gz')
            np.savetxt(experiment_folder / 'power_doppler' / 'pwd_timestamps.txt', time_stamps)

        # %%
        # Pipeline 2: power doppler -> zmap

        pwd = nib.load(experiment_folder / 'power_doppler' / 'pwd.nii.gz')
        time_stamps = np.loadtxt(experiment_folder / 'power_doppler' / 'pwd_timestamps.txt')

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
        event = min(set(events.trial_type))
        events2 = events[events['trial_type'] == event]

        # glm
        (experiment_folder / 'glm').mkdir(parents=True, exist_ok=True)
        if not (experiment_folder / 'glm' / 'shifts.png').exists() or overwrite:
            shifts, max_tstats, best_offset = fit_glm_time_shift(
                pwd_reg, time_stamps,
                event, events2, shift=20,
                smoothing_fwhm=smoothing_fwhm, hrf='glover',
                transformations=transformations,
                out_dir=experiment_folder / 'glm', n_jobs=-5
            )
            fig, ax = plt.subplots()
            ax.plot(shifts, max_tstats)
            ax.set_xlabel('shifts')
            ax.set_ylabel('max_z')
            fig.savefig(experiment_folder / 'glm' / 'shifts.png')
            plt.close(fig)

        if (experiment_folder / 'glm' / 'pwd_zmap.nii.gz').exists() and not overwrite:
            zmap = nib.load(experiment_folder / 'glm' / 'pwd_zmap.nii.gz')
            mean_pwd = nib.load(experiment_folder / 'glm' / 'pwd_mean.nii.gz')
        else:
            mean_pwd = mean_img(pwd_reg)
            nib.save(mean_pwd, experiment_folder / 'glm' / 'pwd_mean.nii.gz')
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
            nib.save(zmap, experiment_folder / 'glm' / 'pwd_zmap.nii.gz')
            nib.save(nib.Nifti1Image(
                (np.abs(np.array(zmap.dataobj)) > 3).astype(np.float32),
                zmap.affine),
                experiment_folder / 'glm' / 'pwd_zmap_mask.nii.gz'
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
            display.savefig(experiment_folder / 'glm' / 'pwd_zmap.png')

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
            ax.plot(time_stamps, pwd_data[ext_idx])
            ymax = np.min([10, np.nanmax(np.abs(pwd_data[tuple(ext_idx)]))])
            ax.set_ylim([-ymax, ymax])
            for _, (onset, trial_type, duration) in events.iterrows():
                ax.axvspan(onset, onset + duration, color='gray', alpha=0.25)
                ax.text(onset + duration / 2, 0.95, trial_type.replace('_', '\n'),
                        ha='center', va='center',
                        transform=BlendedGenericTransform(ax.transData, ax.transAxes))
            fig.savefig(experiment_folder / 'glm' / f'pwd_tc{roi_idx}.png')
            plt.close(fig)
            zmap_data[clusters == clusters[ext_idx]] = np.nan
            roi_idx += 1

        # 3D rendering
        """
        import pyvista as pv
        zmap_data = np.array(zmap.dataobj)
        zmap_data[np.abs(zmap_data) < 3] = np.nan
        pwd_mean_data = np.array(mean_pwd.dataobj)#
        # pwd_mean_data[np.isinf(pwd_mean_data)] = np.nan
        # pwd_mean_data[np.abs(pwd_mean_data) < np.quantile(np.abs(pwd_mean_data), 0.5)] = np.nan
        # pwd_mean_vol = pv.ImageData(dimensions=pwd_mean_data.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
        # pwd_mean_vol.point_data['data'] = pwd_mean_data.ravel(order='F')
        pl = pv.Plotter()
        # pl.add_mesh(pwd_mean_vol, nan_opacity=0, cmap="gray_r", opacity=0.25)
        pl.add_volume(pwd_mean_data, cmap='magma')
        pl.add_volume(zmap_data, clim=(-5, 5), cmap="coolwarm")
        pl.show()
        """
    except Exception as err:
        with open('log.txt', 'a') as fid:
            fid.write(f'{base_path} {sequence} {err}\n')
