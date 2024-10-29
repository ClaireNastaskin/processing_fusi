from pathlib import Path
import numpy as np
from tqdm import tqdm
import jax
import re
from datetime import datetime
import time
import pandas as pd
# import h5py

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
from anise.io.behavior_loader import behavior_loader
from anise.utils import get_power_doppler_nii
from anise.process.fusi_glm_fit2 import rat_hrf, fit_glm_time_shift
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.image import mean_img

# plotting
import nibabel as nib
import matplotlib.pyplot as plt

base_path = Path('fUS_data/fUSI_Rat_acquisition/2024-09-10/Rat_3/Functional_recording/run-06/')
sequence = 'seq-RF_reducedTX_ReducedRX_6500000Hz_1a_3c_200loops_-1Gain_3AFE_3rp'
experiment_folder = base_path / 'acquisitions' / sequence

event_offset = -5
smoothing_fwhm = 0.3

df = pd.DataFrame(columns=['step', 'time', 'ensemble_duration', 'iteration'])

# %%
# Pipeline 1: raw -> power doppler
metadata = PoseidonMetadata.from_folder(experiment_folder / 'metadata')
acquisition_sequence_metadata, transducer_metadata = metadata_to_schema(metadata)
raw_data_files = list((experiment_folder / "raw_frame_data").glob("[!.]*.h5"))
events = behavior_loader(
    base_path / 'streams' / 'task-event_stream.h5'
)

pwd_frames = dict()
for i, raw_data_file in tqdm(enumerate(raw_data_files)):
    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', raw_data_file.stem)
    timestamp = datetime.strptime(match.group(0).replace('-', ':'), '%Y:%m:%dT%H:%M:%S:%f')
    # %%
    # Preprocess
    start = time.time()
    data, scan_timing_metadata = preprocess_raw_data(
        metadata, raw_data_file, transducer_metadata=transducer_metadata
    )
    ensemble_duration = scan_timing_metadata.trigger_timestamps_s[-1]
    df.loc[len(df.index)] = ('preprocess', time.time() - start, ensemble_duration, i)
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
    start = time.time()
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
    df.loc[len(df.index)] = ('beamform', time.time() - start, ensemble_duration, i)
    # fails: settings changed in between runs
    # file = h5py.File(str(raw_data_file).replace('raw_frame_data', 'beamformed'))
    # np.testing.assert_array_equal(beamformed, file['beamformed'][:])

    # %%
    # Power doppler
    start = time.time()
    pwd_frame = PCAMetalPowerDoppler(
        num_tissue_components=50,
        num_blood_and_tissue_components=None
    ).fit_transform(
        beamformed.transpose(1, 2, 3, 0)
    )
    df.loc[len(df.index)] = ('power doppler', time.time() - start, ensemble_duration, i)
    del beamformed
    # Also fails: because of backend? Stochastic?
    # fname = list((raw_data_file.parent.parent / 'power_doppler').glob(f'{raw_data_file.stem}*.h5'))[0]
    # assert 'num_blood_and_tissue_components=None' in fname.stem, 'num_tissue_components=50' in fname.stem
    # np.testing.assert_array_equal(pwd_frame, h5py.File(fname)['power_doppler'][:])

    pwd_frames[timestamp] = pwd_frame

time_stamps = sorted(pwd_frames)
pwd = nib.Nifti1Image(
    np.array([pwd_frames.get(timestamp)
              for timestamp in time_stamps]).transpose(1, 2, 3, 0),
    np.diag([0.15, 0.15, 0.15, 1])
)
del pwd_frames
time_stamps = np.array([(timestamp - time_stamps[0]).total_seconds()
                        for timestamp in time_stamps])

# Plot times
df['time'] /= df['ensemble_duration']
gb = df.groupby('step').mean()
autopct_fun = lambda p: f'{p:.2f}%\n({p / 100 * gb["time"].sum():.2f})'
ax = gb.plot.pie(
    y='time', autopct=autopct_fun, figsize=(6, 6)
)
ax.figure.savefig('time_pie.png', dpi=300)

# %%
# Pipeline 2: power doppler -> zmap

# power_doppler_path = experiment_folder / 'power_doppler'

"""
shifts, max_tstats, best_offset = fit_glm_time_shift(
    pwd, time_stamps, 'pwm_enable', events, shift=20,
    smoothing_fwhm=smoothing_fwhm,
)
fig, ax = plt.subplots()
ax.plot(shifts, max_tstats)
ax.set_xlabel('shifts')
ax.set_ylabel('max_z')
fig.savefig('tmp.png')
"""

mean_pwd = mean_img(pwd)
nib.save(mean_pwd, 'tmp_mean.nii.gz')
events_shifted = events.copy()
events_shifted['onset'] += event_offset
start = time.time()
design_matrix = make_first_level_design_matrix(
    time_stamps,
    events_shifted,
    drift_model="polynomial",
    drift_order=1,
    hrf_model='glover'  # rat_hrf
)
glm  = FirstLevelModel(minimize_memory=False, mask_img=False,
                       smoothing_fwhm=smoothing_fwhm, standardize=True).fit(
    pwd, design_matrices=design_matrix
)
zmap = glm.compute_contrast(
    'pwm_enable',
    output_type="stat"
)
print(f'GLM took {time.time() - start} seconds')
nib.save(zmap, 'tmp_zmap.nii.gz')
