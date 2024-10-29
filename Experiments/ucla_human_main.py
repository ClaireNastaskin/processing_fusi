from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import jax
import re
from datetime import datetime
import h5py
import pandas as pd

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

# GLM
from anise.utils import get_power_doppler_nii, register_image_stack
from anise.process.fusi_glm_fit2 import rat_hrf, fit_glm_time_shift
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.image import mean_img

# plotting
from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt

base_path = Path('fUS_data/UCLA/data/UCLA_009/Functional_runs/run-01')
sequence = 'seq-2d_plane_wave'
experiment_folder = base_path / 'acquisitions' / sequence

event_offset = 0
smoothing_fwhm = 0.3


def decode_event_item(item):
    if isinstance(item, bytes):
        item = item.decode('utf-8')
    if isinstance(item, str) and '{' in item and '}' in item:
        item = json.loads(item)
    return item


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

raw_data_files = (experiment_folder / "raw_frame_data").glob("[!.]*.h5")
pwd_frames = dict()
for raw_data_file in tqdm(list(raw_data_files)):
    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', raw_data_file.stem)
    timestamp = datetime.strptime(match.group(0).replace('-', ':'), '%Y:%m:%dT%H:%M:%S:%f')
    # %%
    # Preprocess
    data, scan_timing_metadata = preprocess_raw_data(
        metadata, raw_data_file, transducer_metadata=transducer_metadata
    )
    # file = h5py.File(str(raw_data_file).replace('raw_frame_data', 'schema'))
    # np.testing.assert_array_equal(data, file['data'][:])
    """
    rf_data = preprocess_iq_data_to_rf(
        data, iq_afe_artifact_data=None, interp_factor=1,
        freq_demodulation=metadata.poseidon_config.tx_freq_hz
    )
    """
    # %%
    # vbeam
    vbeam_setup = import_space_time_to_vbeam_setup(
        rf_or_iq_data=data,
        acquisition_sequence_metadata=acquisition_sequence_metadata,
        speed_of_sound_m_s=1540.0,
    )
    del data
    # Naive way to reduce GPU-RAM usage
    # Slice by:
    # - frame
    # then beamform each separately
    vbeam_setup_slice = vbeam_setup.slice["frames", 0]
    beamformer = das_beamformer(
        setup=vbeam_setup_slice,
        ram_limit_gb=8.0,
    )
    beamformer = jax.jit(beamformer)
    beamformed = np.array(
        _time_beamform(vbeam_setup=vbeam_setup, beamformer=beamformer)
    )
    # fails: settings changed in between runs
    # file = h5py.File(str(raw_data_file).replace('raw_frame_data', 'beamformed'))
    # np.testing.assert_array_equal(beamformed, file['beamformed'][:])

    # %%
    # Power doppler
    pwd_frame = PCAMetalPowerDoppler(
        num_tissue_components=50,
        num_blood_and_tissue_components=None
    ).fit_transform(
        beamformed.transpose(1, 2, 3, 0)
    )
    del beamformed
    # Also fails: because of backend? Stochastic?
    # fname = list((raw_data_file.parent.parent / 'power_doppler').glob(f'{raw_data_file.stem}*.h5'))[0]
    # assert 'num_blood_and_tissue_components=None' in fname.stem, 'num_tissue_components=50' in fname.stem
    # np.testing.assert_array_equal(pwd_frame, h5py.File(fname)['power_doppler'][:])

    pwd_frames[timestamp] = pwd_frame


"""
pwd_files = (experiment_folder / "power_doppler").glob("[!.]*.h5")
pwd_frames = dict()
for pwd_file in tqdm(list(pwd_files)):
    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', pwd_file.stem)
    timestamp = datetime.strptime(match.group(0).replace('-', ':'), '%Y:%m:%dT%H:%M:%S:%f')
    pwd_frames[timestamp] = h5py.File(pwd_file)['power_doppler'][:]
"""

time_stamps = sorted(pwd_frames)
pwd = nib.Nifti1Image(
    np.array([pwd_frames.get(timestamp)
              for timestamp in time_stamps]).transpose(1, 2, 3, 0),
    np.diag([0.15, 0.15, 0.15, 1])
)
(experiment_folder / 'power_doppler').mkdir(parents=True, exist_ok=True)
nib.save(pwd, experiment_folder / 'power_doppler' / 'pwd.nii.gz')
del pwd_frames
time_stamps = np.array([(timestamp - time_stamps[0]).total_seconds()
                        for timestamp in time_stamps])
np.savetxt(experiment_folder / 'power_doppler' / 'pwd_timestamps.txt', time_stamps)

# %%
# Pipeline 2: power doppler -> zmap

pwd = nib.load(experiment_folder / 'power_doppler' / 'pwd.nii.gz')
time_stamps = np.loadtxt(experiment_folder / 'power_doppler' / 'pwd_timestamps.txt')

pwd_reg, transformations = register_image_stack(
    np.array(pwd.dataobj)[:, 0]
)
pwd_reg = nib.Nifti1Image(pwd_reg[:, None], pwd.affine)
# power_doppler_path = experiment_folder / 'power_doppler'

"""
pwd, time_stamps = get_power_doppler_nii(
    power_doppler_path, num_tissue_components='50'
)
"""
event = 'right_hand_squeeze'
events2 = events[events['trial_type'] == event]
shifts, max_tstats, best_offset = fit_glm_time_shift(
    pwd_reg, time_stamps, event, events2, shift=20,
    smoothing_fwhm=smoothing_fwhm, hrf='glover',
    transformations=transformations
)
fig, ax = plt.subplots()
ax.plot(shifts, max_tstats)
ax.set_xlabel('shifts')
ax.set_ylabel('max_z')
fig.savefig(experiment_folder / 'power_doppler' / 'shifts.png')
plt.close(fig)


"""pwd, time_stamps = get_power_doppler_nii(
    power_doppler_path, num_tissue_components='50'
)"""
mean_pwd = mean_img(pwd_reg)
nib.save(mean_pwd, experiment_folder / 'power_doppler' / 'pwd_mean.nii.gz')
events_shifted = events.copy()
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
nib.save(zmap, experiment_folder / 'power_doppler' / 'pwd_zmap.nii.gz')

display = plotting.plot_stat_map(
    zmap,
    bg_img=mean_pwd,
    cut_coords=[0],
    threshold=3,
    display_mode="y",
    black_bg=True,
    title=event,
)
display.savefig(experiment_folder / 'power_doppler' / 'pwd_zmap.png')
