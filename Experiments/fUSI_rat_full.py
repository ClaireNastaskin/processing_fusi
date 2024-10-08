from pathlib import Path

# raw to schema
from mangrove.io.metadata import PoseidonMetadata
from mangrove.io.convert.raw_data_to_schema import metadata_to_schema, preprocess_raw_data
from mangrove.beamform.preprocess import preprocess_iq_data_to_rf

# beamforming

# power doppler

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

# %%
# Pipeline 1: raw -> power doppler
metadata = PoseidonMetadata.from_folder(experiment_folder / 'metadata')
acquisition_sequence_metadata, transducer_metadata = metadata_to_schema(metadata)
raw_data_files = (experiment_folder / "raw_frame_data").glob("[!.]*.h5")
for raw_data_file in raw_data_files:
    data, scan_timing_metadata = preprocess_raw_data(
        metadata, raw_data_file, transducer_metadata=transducer_metadata
    )
    rf_data = preprocess_iq_data_to_rf(
        data, iq_afe_artifact_data=None, interp_factor=1,
        freq_demodulation=metadata.poseidon_config.tx_freq_hz
    )


# %%
# Pipeline 2: power doppler -> zmap
event_offset = -5
power_doppler_path = experiment_folder / 'power_doppler'
events = behavior_loader(
    base_path / 'streams' / 'task-event_stream.h5'
)
pwd, time_stamps = get_power_doppler_nii(
    power_doppler_path, num_tissue_components='50'
)
shifts, max_tstats, best_offset = fit_glm_time_shift(
    pwd, time_stamps, 'pwm_enable', events, shift=20
)
fig, ax = plt.subplots()
ax.plot(shifts, max_tstats)
ax.set_xlabel('shifts')
ax.set_ylabel('max_z')
fig.savefig('tmp.png')

mean_pwd = mean_img(pwd)
nib.save(mean_pwd, 'tmp_mean.nii.gz')
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
                       smoothing_fwhm=None, standardize=True).fit(
    pwd, design_matrices=design_matrix
)
zmap = glm.compute_contrast(
    'pwm_enable_rat_hrf',
    output_type="stat"
)

