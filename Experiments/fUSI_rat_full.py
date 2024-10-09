from pathlib import Path
import numpy as np
import jax

# raw to schema
from mangrove.io.metadata import PoseidonMetadata
from mangrove.io.convert.raw_data_to_schema import metadata_to_schema, preprocess_raw_data
from mangrove.beamform.preprocess import preprocess_iq_data_to_rf

# beamforming
from mangrove.beamform.vbeam_ import das_beamformer
from mangrove.io.convert.vbeam.mangrove_to_vbeam import \
    import_space_time_to_vbeam_setup, _time_beamform

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

# %%
# Pipeline 1: raw -> power doppler
metadata = PoseidonMetadata.from_folder(experiment_folder / 'metadata')
acquisition_sequence_metadata, transducer_metadata = metadata_to_schema(metadata)
raw_data_files = (experiment_folder / "raw_frame_data").glob("[!.]*.h5")
pwd = list()
for raw_data_file in raw_data_files:
    # %%
    # Preprocess
    data, scan_timing_metadata = preprocess_raw_data(
        metadata, raw_data_file, transducer_metadata=transducer_metadata
    )
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

    # %%
    # Power doppler
    pwd_frame = PCAMetalPowerDoppler(
        num_tissue_components=50,
        num_blood_and_tissue_components=None
    ).fit_transform(
        beamformed
    )

    pwd.append(pwd_frame)

# %%
# Pipeline 2: power doppler -> zmap

power_doppler_path = experiment_folder / 'power_doppler'
events = behavior_loader(
    base_path / 'streams' / 'task-event_stream.h5'
)

"""
pwd, time_stamps = get_power_doppler_nii(
    power_doppler_path, num_tissue_components='50'
)
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

pwd, time_stamps = get_power_doppler_nii(
    power_doppler_path, num_tissue_components='50'
)
mean_pwd = mean_img(pwd)
nib.save(mean_pwd, 'tmp_mean2.nii.gz')
events_shifted = events.copy()
events_shifted['onset'] += event_offset
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
nib.save(zmap, 'tmp_zmap2.nii.gz')

