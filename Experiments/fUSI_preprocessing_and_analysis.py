import argparse
import hashlib
import io
import json
import os
import pickle
import re
import time
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import xarray as xr
from dendrology.beamform_params import BeamformReconParams
from dipy.align import affine_registration
from joblib import Parallel, delayed
from loguru import logger
from mangrove.beamform.cuda_ import CudaBeamformer
from mangrove.io.convert.raw_data_to_schema import (
    metadata_to_schema,
    preprocess_raw_data,
)
from mangrove.io.metadata import PoseidonMetadata
from mangrove.power_doppler.pca import PCAPowerDoppler
from mangrove.utils.geometry import bfly_angles_to_cartesian
from mne.transforms import _affine_to_quat
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import _gamma_difference_hrf
from tqdm import tqdm

os.environ["JAX_PLATFORMS"] = "cpu"

# TO DO:
# - [ ] Add registration rejection

# %%
# Input parameters (modify these)

parser = argparse.ArgumentParser(description="Analyze a fusi dataset")
parser.add_argument(
    "--sourcedata",
    type=Path,
    default=Path("sourcedata"),
    help="Path to the sourcedata folder",
)
parser.add_argument(
    "--bids-root",
    type=Path,
    default=Path("BIDS"),
    help="Path to the BIDS root directory (default ./BIDS).",
)
parser.add_argument("--species", type=str, default="human", help="Species type")
parser.add_argument(
    "--run-folder",
    type=str,
    default="[!.]*run-*",
    help="Pattern to subselect run folders",
)
parser.add_argument(
    "--sequence-folder",
    type=str,
    default="[!.]*",
    help="Pattern to subselect sequence folders",
)
parser.add_argument(
    "--overwrite", type=bool, default=False, help="Whether to overwrite the data"
)

args = parser.parse_args()
sourcedata, bids_root, species = args.sourcedata, args.bids_root, args.species
run_folder_glob, sequence_folder_glob = args.run_folder, args.sequence_folder
overwrite = args.overwrite
if not sourcedata.exists():
    error_msg = "`sourcedata` not found use `--sourcedata /path/to/sourcedata`"
    raise RuntimeError(error_msg)

n_jobs = 25  # number of ensemble files that can fit in memory
fps = 12

task_descriptions = {
    "rest": "rest",
    "visual": "Blue LED, flashing at 5Hz",
    "joystick": "Playing connect the dots with a video game controller",
    "wrist+fingers": "Moving left and right wrist or fingers in time with a video",
    "fallingblocks": "Using four fingers on the hand opposite the recording side, "
    "catch blocks as they fall by pressing the corresponding button like guitar hero",
}

# TO DO: integrate into beamformer
speed_of_sound_lut = {
    "human": 1540,
    "rat": 1540,
    "ATS539": 1450,
    "water": 1500,
}

beamform_params = {
    "f_number": 2.0,
    "pixel_size_wavelengths": 0.5,
    "max_depth_factor": 1.0,
    "aperture_lateral_factor": 1.0,
    "aperture_elevation_factor": 1.0,
    "envelope": None,
    "use_apodization": True,
    "array_lib": "cuda",
    "ram_limit_gb": None,
}

pwd_params = {
    "num_tissue_components": 50,
    "num_blood_and_tissue_components": None,
}

reg_params = {}

glm_params = {
    "hrf_model": "rat" if species == "rat" else "glover",
    "smoothing_fwhm": 0.3,
    "drift_model": "cosine",
    "drift_order": 1,
    "n_best_vox": 20,
    "event_time_offsets": np.arange(-12, 12.25, 0.25).round(2),
}

# %%
# Internal parameters (don't modify)

bids_root.mkdir(exist_ok=True)
fusi_analysis_pipeline_folder = bids_root / "fusi-analysis-pipeline"
fusi_analysis_pipeline_folder.mkdir(exist_ok=True)
sequence_out_folder = fusi_analysis_pipeline_folder / "sequences"
sequence_out_folder.mkdir(parents=True, exist_ok=True)
sequence_hash_lut = {}

proc_folder = bids_root / "fusi-analysis-pipeline" / "processes"
proc_folder.mkdir(parents=True, exist_ok=True)
pipeline_params = [beamform_params, pwd_params, reg_params, glm_params]
param_hashes = [
    hashlib.sha1(str(params).encode()).hexdigest()[:6] for params in pipeline_params
]  # noqa: S324
for params, hash_code in zip(pipeline_params, param_hashes):
    with open(proc_folder / f"{hash_code}.pkl", "wb") as fid:
        pickle.dump(params, fid)
pipeline_hashes = ["+".join(param_hashes[: i + 1]) for i in range(len(param_hashes))]

pca_pwd = PCAPowerDoppler(**pwd_params)

clutter_filters = [
    {
        "FilterType": "Fixed-threshold SVD",
        "LowThreshold": pca_pwd.num_tissue_components,
        "HighThreshold": (
            "n/a"
            if pca_pwd.num_blood_and_tissue_components is None
            else pca_pwd.num_blood_and_tissue_components
        ),
    }
]


# %%
# Helper functions to be incorporated into packages in the future

log_mangrove = bids_root / "log_mangrove.log"
log_dipy = io.StringIO()
logger.remove()
logger.add(log_mangrove, level="DEBUG", enqueue=True)
logger.add(log_mangrove, level="INFO", enqueue=True)


def get_exp_start_time(run_folder):
    commands_fpath = run_folder / "streams" / "probe-commands_stream.h5"
    if not commands_fpath.exists():
        commands_fpath = run_folder / "streams" / "probe_1-commands_stream.h5"
    if not commands_fpath.exists():
        return
    with h5py.File(commands_fpath, "r") as file:
        timestamp = file.attrs["experiment_start_utc"]
    return datetime.fromisoformat(timestamp).astimezone(timezone.utc)


def metadata_to_json(metadata):
    """Convert metadata into a single json."""
    metadata_json = {"probe_version": metadata.probe_version}
    for param in (
        "clock_info",
        "poseidon_config",
        "sequence_timing_config",
        "transmit_steering_config",
    ):
        metadata_json.update(getattr(metadata, param))
    for key, val in metadata_json.items():
        try:
            json.dumps({key: val})
        except Exception:
            metadata_json[key] = str(val)
    return metadata_json


def get_sequence_folder_info(sequence_folder):
    if not (sequence_folder / "metadata" / "probe_version.txt").exists():
        return
    # from the base_name, first is the sequence name, then the folder
    # called acquisitions, then the BIDS base name
    base_name = sequence_folder.parts[-3]

    metadata = PoseidonMetadata.from_folder(sequence_folder / "metadata")

    metadata_json = metadata_to_json(metadata)

    acq_hash = hashlib.sha1(
        json.dumps(metadata_json, sort_keys=True).encode()
    ).hexdigest()  # noqa: S324
    acq = acq_hash[:6]
    if acq in sequence_hash_lut and sequence_hash_lut[acq] != acq_hash:
        error_msg = "Hash shortening mistake, increase the number of characters used"
        raise ValueError(error_msg)
    sequence_hash_lut[acq] = acq_hash

    with open(sequence_out_folder / f"{acq}.pkl", "wb") as fid:
        pickle.dump(metadata, fid)

    ids = dict(pair.split("-") for pair in base_name.split("_"))
    dtype = "angio" if ids["task"] == "rest" else "fus"
    out_dir = bids_root / "rawdata" / f"sub-{ids['sub']}" / f"ses-{ids['ses']}" / dtype
    fpath_upd = out_dir / f"{base_name}_acq-{acq}_upd.zarr"
    raw_data_files = list(sequence_folder.glob("raw_frame_data/*.h5"))
    raw_data_files = sorted(raw_data_files, key=get_legacy_timestamp)
    return fpath_upd, raw_data_files, metadata, acq_hash


def get_legacy_timestamp(raw_data_file):
    """Get the timestamps of a raw data file."""
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}", raw_data_file.stem)
    timestamp = datetime.strptime(
        match.group(0).replace("-", ":"), "%Y:%m:%dT%H:%M:%S:%f"
    )
    # bug where saved out in PDT
    if timestamp < datetime(2025, 2, 14):
        timestamp += timedelta(hours=8)
    return timestamp.timestamp()


def decode_raw(raw_data_file, metadata, transducer_metadata):
    """Bit unpack and reshape the raw data from butterfly format to sensors by time."""
    try:
        data, scan_timing_metadata = preprocess_raw_data(
            metadata, raw_data_file, transducer_metadata=transducer_metadata
        )
    except OSError:
        return None, None
    return data, scan_timing_metadata


def write_raw(fpath_upd, raw_data_files, metadata, exp_start_time, n_jobs=None):  # noqa: C901
    metadata_json = metadata_to_json(metadata)
    acquisition_sequence_metadata, transducer_metadata = metadata_to_schema(metadata)
    elevation = acquisition_sequence_metadata.receive_element_positions_m[0, :, 1]
    for el in acquisition_sequence_metadata.receive_element_positions_m[:, :, 1]:
        assert np.array_equal(el, elevation)  # noqa: S101
    lateral = acquisition_sequence_metadata.receive_element_positions_m[:, 0, 0]
    for lat in acquisition_sequence_metadata.receive_element_positions_m[:, :, 0].T:
        assert np.array_equal(lat, lateral)  # noqa: S101
    fast_time = np.linspace(
        *acquisition_sequence_metadata.imaging_depth_s,
        acquisition_sequence_metadata.num_samples,
    )
    fpath_upd.parent.mkdir(parents=True, exist_ok=True)
    scan_timing_metadata = []
    for i in tqdm(range(len(raw_data_files) // n_jobs + 1)):
        raw_data_files_pick = raw_data_files[i * n_jobs : (i + 1) * n_jobs]
        raw_batch = dict(
            zip(
                raw_data_files_pick,
                Parallel(n_jobs=n_jobs)(
                    delayed(decode_raw)(raw_data_file, metadata, transducer_metadata)
                    for raw_data_file in raw_data_files_pick
                ),
            )
        )

        if i == 0:
            raw_data, scan_timing = raw_batch[raw_data_files_pick[0]]
            if raw_data is None:
                return
            if scan_timing is None:
                timestamp = get_legacy_timestamp(raw_data_files[0])
            else:
                timestamp = scan_timing.ensemble_start_datetime.timestamp()
                scan_timing_metadata.append(scan_timing.model_dump())
            raw_data = raw_data.expand_dims(
                time=[timestamp - exp_start_time.timestamp()]
            )
            raw_data = raw_data.rename(
                {"rx_elevation_index": "elevation", "rx_lateral_index": "lateral"}
            )
            raw_data = raw_data.assign_coords(
                elevation=elevation, lateral=lateral, fast_time=fast_time
            )
            tx_angles = bfly_angles_to_cartesian(
                np.deg2rad(acquisition_sequence_metadata.pulse_configs.tx_az_angle_deg),
                np.deg2rad(acquisition_sequence_metadata.pulse_configs.tx_el_angle_deg),
            )
            raw_data.coords["transmit_angle_x"] = ("pulse", tx_angles[:, 0])
            raw_data.coords["transmit_angle_y"] = ("pulse", tx_angles[:, 1])
            raw_data.coords["transmit_angle_z"] = ("pulse", tx_angles[:, 2])
            raw_data.attrs["sample_rate_hz"] = (
                acquisition_sequence_metadata.sample_rate_hz
            )
            raw_data.attrs["transmit_freq_hz"] = (
                acquisition_sequence_metadata.transmit_freq_hz
            )
            raw_data.to_zarr(fpath_upd, mode="w")
            del raw_data, raw_batch[raw_data_files_pick[0]]
        for raw_data_file in tqdm(list(raw_batch)):
            raw_data, scan_timing = raw_batch[raw_data_file]
            if raw_data is None:
                continue
            if scan_timing is None:
                timestamp = get_legacy_timestamp(raw_data_file)
            else:
                timestamp = scan_timing.ensemble_start_datetime.timestamp()
                scan_timing_metadata.append(scan_timing.model_dump())
            raw_data = raw_data.expand_dims(
                time=[timestamp - exp_start_time.timestamp()]
            )
            raw_data = raw_data.rename(
                {"rx_elevation_index": "elevation", "rx_lateral_index": "lateral"}
            )
            raw_data.to_zarr(fpath_upd, mode="a", append_dim="time")
            del raw_data, raw_batch[raw_data_file]
    metadata_json["scan_timing"] = scan_timing_metadata
    for col in metadata_json:
        if col not in scans:
            scans[col] = None
    with open(fpath_upd.with_suffix(".json"), "w") as fid:
        json.dump(metadata_json, fid, indent=4)
    return metadata_json


def get_fpaths(base_path):
    sub_ses_folder = Path().joinpath(*base_path.parts[-4:-2])
    data_type = base_path.parts[-2]
    base_name = "_".join(base_path.name.split("_")[:-1])
    base_name_no_acq = "_".join(base_name.split("_")[:-1])
    return {
        "events": base_path.parent / f"{base_name_no_acq}_events.tsv",
        "events_pwd": (
            bids_root
            / "sourcedata"
            / sub_ses_folder
            / data_type
            / f"{base_name_no_acq}_events.tsv"
        ),
        "bmode": (
            bids_root
            / "sourcedata"
            / sub_ses_folder
            / data_type
            / f"{base_name}_proc-{pipeline_hashes[0]}_bmode.zarr"
        ),
        "pwd": (
            bids_root
            / "sourcedata"
            / sub_ses_folder
            / data_type
            / f"{base_name}_proc-{pipeline_hashes[1]}_pwd.nii.gz"
        ),
        "pwd_reg": (
            bids_root
            / "fusi-analysis-pipeline"
            / sub_ses_folder
            / "registration"
            / f"{base_name}_proc-{pipeline_hashes[2]}_pwd.nii.gz"
        ),
        "zmap": (
            bids_root
            / "fusi-analysis-pipeline"
            / sub_ses_folder
            / "glm"
            / f"{base_name}_proc-{pipeline_hashes[3]}_zmap.nii.gz"
        ),
        "plot": (
            bids_root / "fusi-analysis-pipeline" / sub_ses_folder / "plots" / base_name
        ),
    }


def load_log_events(log_fpath, exp_start_time):
    timestamps = []
    events = []
    with open(log_fpath) as fid:
        for line in fid:
            timestamps.append(datetime.fromisoformat(line.split()[0]))
            events.append(line.split("-")[-1].strip())
    onsets = []
    durations = []
    rest = True
    for timestamp, event in zip(timestamps, events):
        if rest:
            if "response" in event.lower():
                onsets.append(timestamp.timestamp())
                rest = False
        else:
            if "stimulus instructions onset" in event.lower():
                durations.append(timestamp.timestamp() - onsets[-1])
                rest = True
    return pd.DataFrame(
        {
            "trial_type": "fallingblock",
            "onset": np.array(onsets) - exp_start_time.timestamp(),
            "duration": durations,
        }
    )


def load_events(event_fpath):
    with h5py.File(event_fpath, "r") as file:
        dataset = file["data"][:]
        events = [event.decode("utf-8") for event in dataset["event"]]
        timestamps = dataset["timestamp"]
        payloads = [payload.decode("utf-8") for payload in dataset["payload"]]
    if all(event == "event" for event in events):
        df = pd.DataFrame([json.loads(payload) for payload in payloads])
    else:
        df = pd.DataFrame({"stimulus": events})
        df["start"] = [("enab" in event or "start" in event) for event in events]
        df["stop"] = [("disab" in event or "stop" in event) for event in events]
    df["timestamp"] = timestamps
    df_starts = df[df["start"]]
    df_stops = df[df["stop"]]
    df = df_starts.copy().drop(columns=["start", "stop"])
    df = df.rename(columns={"stimulus": "trial_type", "timestamp": "onset"})
    if len(df) > len(df_stops):
        df = df.iloc[:-1]
    df["duration"] = np.array(df_stops["timestamp"]) - np.array(df["onset"])
    return df.reset_index(drop=True)


def truncate_pca(data, n_components_start, n_components_end=None):
    data_2d = data.reshape(-1, data.shape[-1])

    # Compute the covariance matrix of the ensemble
    # i.e., mean-centered product
    data_2d -= data_2d.mean(axis=-1, keepdims=True)
    cov_matrix = (data_2d.conj().T @ data_2d) / (data.shape[-1] - 1)
    # PCA algorithm: calculate eigenvectors of the covariance matrix
    eig_val, eig_vect = np.linalg.eigh(cov_matrix)
    # sort eigenvectors by eigenvalues
    idx = np.argsort(eig_val)[::-1]
    # eig_val = eig_val[idx] keeping this here for later visualizations
    eig_vect = eig_vect[:, idx]

    # Sub-select the eigenvectors for blood components
    eig_vect_sliced = eig_vect[:, n_components_start:n_components_end]

    # Project original data onto the eigenvectors within the specified range
    M_A_subset = data_2d @ eig_vect_sliced

    # Project the data back into the original space, dropping non-blood components
    return M_A_subset @ eig_vect_sliced.T

    # In normal PCA, we would also add back in the mean of the data
    # but we are only interested in the changing part of the signal

    # Compute Power Doppler as temporal integration of the ensemble
    """
    return np.mean(
        np.abs(data_out).reshape(data.shape) ** 2,
        axis=-1,
    )
    """


def _raw_to_power_doppler(raw_data, timestamp, beamformer, save_bmodes, fpath):
    computation_times = {}

    # beamforming
    start = time.time()
    beamformed = beamformer.fit_transform(
        raw_data["data"]
        .sel(time=timestamp)
        .rename({"elevation": "rx_elevation_index", "lateral": "rx_lateral_index"})
    )
    computation_times["beamform"] = time.time() - start

    beamformed = beamformed.transpose("lateral", "elevation", "depth", "loop")
    if save_bmodes:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        kwargs = (
            {"mode": "w"}
            if timestamp == raw_data.time.values[0]
            else {"mode": "a", "append_dim": "time"}
        )
        beamformed.expand_dims(time=[timestamp]).to_zarr(fpath, **kwargs)

    # power doppler
    start = time.time()
    pwd_frame = pca_pwd.fit_transform(beamformed.values)
    computation_times["pwd"] = time.time() - start
    del beamformed
    return pwd_frame, computation_times


def raw_to_power_doppler(
    fpath_upd, metadata, fpath_bmode, fpath_pwd, save_bmodes=False, save_pwd=True
):
    acquisition_sequence_metadata, transducer_metadata = metadata_to_schema(metadata)
    raw_data = xr.open_dataset(fpath_upd, engine="zarr")
    timestamps = sorted(raw_data.time.values)
    ensemble_duration = (
        acquisition_sequence_metadata.compound_bmode_repetition_interval_s
        * acquisition_sequence_metadata.ensemble_length
    )
    if beamform_params["array_lib"] == "vbeam":
        # import as needed to avoid incompatibility with joblib
        from mangrove.beamform.vbeam_ import VBeamformer

        beamformer = VBeamformer(
            BeamformReconParams(
                **{**beamform_params, "speed_of_sound": speed_of_sound_lut[species]}
            ),
            acquisition_sequence_metadata,
        )
    else:
        assert beamform_params["array_lib"] == "cuda"  # noqa: S101
        beamformer = CudaBeamformer(
            BeamformReconParams(
                **{**beamform_params, "speed_of_sound": speed_of_sound_lut[species]}
            ),
            acquisition_sequence_metadata,
        )
    sidecar_pwd = get_sidecar(
        fpath_upd,
        acquisition_sequence_metadata,
        transducer_metadata,
        beamformer,
        timestamps,
    )

    pwd_frames = {}
    df_benchmark = pd.DataFrame(columns=["step", "time", "ensemble_duration"])
    for timestamp in tqdm(timestamps):
        pwd_frames[timestamp], computation_times = _raw_to_power_doppler(
            raw_data, timestamp, beamformer, save_bmodes, fpath_bmode
        )
        for name, computation_time in computation_times.items():
            df_benchmark.loc[len(df_benchmark.index)] = (
                name,
                computation_time,
                ensemble_duration,
            )

    if not pwd_frames:
        return

    # TO DO: better affine units, store as xarray, xr_to_nii
    pwd = nib.Nifti1Image(
        np.array(
            [pwd_frames.get(timestamp) for timestamp in raw_data.time.values]
        ).transpose(1, 2, 3, 0),
        np.diag([0.15, 0.15, 0.15, 1]),
    )
    del pwd_frames

    if save_pwd:
        fpath_pwd.parent.mkdir(parents=True, exist_ok=True)
        nib.save(pwd, fpath_pwd)
        with open(fpath_pwd.as_posix().split(".")[0] + ".json", "w") as fid:
            json.dump(sidecar_pwd, fid, indent=4)
    return pwd, sidecar_pwd, df_benchmark


def plot_benchmark(df_benchmark, fpath):
    if df_benchmark["ensemble_duration"].values.all():
        df_benchmark["time"] /= df_benchmark["ensemble_duration"]
    gb = df_benchmark.groupby("step")["time"].mean()
    autopct_fun = lambda p: f"{p:.2f}%\n({p / 100 * gb.sum():.2f})"
    ax = gb.plot.pie(y="time", autopct=autopct_fun, figsize=(6, 6))
    fpath.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(
        fpath.with_name(fpath.name + "_timepie").with_suffix(".png"), dpi=300
    )
    plt.close(ax.figure)


def get_sidecar(
    fpath_upd,
    acquisition_sequence_metadata,
    transducer_metadata,
    beamformer,
    timestamps,
):
    params = dict(pair.split("-", 1) for pair in fpath_upd.name.split("_")[:-1])
    task = params["task"]
    task_description = task_descriptions[task]
    sidecar = {
        # Scanner and probe hardware
        "Manufacturer": "Butterfly",
        "ProbeCentralFrequency": acquisition_sequence_metadata.transmit_freq_hz
        * 1e-6,  # in MHz
        "ProbeNumberOfElements": [
            transducer_metadata.lateral.size,
            transducer_metadata.elevation.size,
        ],
        "ProbePitch": 0.208,  # in mm
        "ProbeRadiusOfCurvature": 0,  # not curved
        "ProbeElevationAperture": (
            np.ptp(acquisition_sequence_metadata.receive_element_positions_m[..., 1])
        ),  # in mm
        "ProbeElevationFocus": (
            "n/a"
            if None in acquisition_sequence_metadata.pulse_configs.tx_el_focus
            else np.mean(acquisition_sequence_metadata.pulse_configs.tx_el_focus)
        ),  # in mm
        "Depth": [
            beamformer.depths[0].round(7) * 1000,
            beamformer.depths[-1].round(7) * 1000,
        ],  # in mm
        "Lateral": [
            beamformer.laterals[0].round(7) * 1000,
            beamformer.laterals[-1].round(7) * 1000,
        ],  # in mm
        "Elevation": [
            beamformer.elevations[0].round(7) * 1000,
            beamformer.elevations[-1].round(7) * 1000,
        ],  # in mm
        "UltrasoundPulseRepetitionFrequency": 1
        / acquisition_sequence_metadata.pulse_repetition_interval_s,
        "PlaneWaveElevationAngles": acquisition_sequence_metadata.pulse_configs.tx_el_angle_deg,  # degrees
        "PlaneWaveAzimuthAngles": acquisition_sequence_metadata.pulse_configs.tx_az_angle_deg,  # degrees
        "UltrafastSamplingFrequency": 1
        / acquisition_sequence_metadata.compound_bmode_repetition_interval_s,
        "DataMode": acquisition_sequence_metadata.data_mode,
        "Gain": acquisition_sequence_metadata.extras.gain_mode,
        "TxCycles": acquisition_sequence_metadata.extras.tx_cycles,
        "ChipRepeats": acquisition_sequence_metadata.extras.num_chip_repeats,
        "TGCSlope": acquisition_sequence_metadata.extras.tgc_slope,
        "TGCDuration": acquisition_sequence_metadata.extras.tgc_duration,
        "TGCInitGain": acquisition_sequence_metadata.extras.tgc_initial_gain_db,
        "AFE": acquisition_sequence_metadata.extras.power_mode,
        "TxApertureMask": [
            str(val)
            for val in acquisition_sequence_metadata.transmit_element_mask.tolist()
        ],
        "RxApertureMask": [
            str(val)
            for val in acquisition_sequence_metadata.receive_element_mask.tolist()
        ],
        "ClutterFilterWindowDuration": (
            acquisition_sequence_metadata.compound_bmode_repetition_interval_s
            * acquisition_sequence_metadata.ensemble_length
        )
        * 1000,  # in ms
        # note same as clutter filter duration for now but could be different
        "PowerDopplerIntegrationDuration": (
            acquisition_sequence_metadata.compound_bmode_repetition_interval_s
            * acquisition_sequence_metadata.ensemble_length
        )
        * 1000,  # in ms
        "ClutterFilters": clutter_filters,
        "EnsembleLoops": acquisition_sequence_metadata.extras.num_imaging_loops,
        "VolumeTiming": timestamps,  # in seconds
        # "VolumeTimingStart":
        #    str(raw.scan_timing_metadata.acquire_start_datetime),  # global time
        "SliceEncodingDirection": transducer_metadata.xyz.to_numpy().tolist(),
        # DelayAfterTrigger = 'n/a' # required for acqs containing the pose entity
        "TaskName": task,
        "TaskDescription": task_description,
        "InstitutionName": "Forest Neurotech",
        "InstitutionAddress": "811 Traction Ave Suite 3C, Los Angeles, CA 90013",
        "InstitutionalDepartmentName": "n/a",
    }
    if species == "rat":
        sidecar.update(
            {
                "InstitutionName": "Caltech",
                "InstitutionAddress": "1200 E California Blvd, Pasadena, CA 91125",
                "InstitutionalDepartmentName": "Neuroscience - Biology and Biological Engineering",
            }
        )
    sidecar.update(beamform_params)
    sidecar.update(pwd_params)
    sidecar.update(glm_params)
    for key, val in sidecar.items():
        try:
            json.dumps({key: val})
        except Exception:
            sidecar[key] = str(val)
    return sidecar


def plot_movie(image, sidecar, fpath, name, axis=1, pos=0.5):
    image_data = np.array(image.dataobj).transpose(3, 0, 1, 2)
    fig, (ax_lat, ax_el) = plt.subplots(1, 2, figsize=(8, 4))
    extent = np.array([*sidecar["Lateral"], *sidecar["Depth"][::-1]])
    img_lat = ax_lat.imshow(
        image_data[0, :, image_data.shape[2] // 2].T,
        cmap="hot",
        aspect="auto",
        extent=extent,
    )
    ax_lat.set_title("Lateral")
    ax_lat.set_xlabel("Lateral (mm)")
    ax_lat.set_ylabel("Depth (mm)")

    extent = np.array([*sidecar["Elevation"], *sidecar["Depth"][::-1]])
    img_el = ax_el.imshow(
        image_data[0, image_data.shape[1] // 2].T,
        cmap="hot",
        aspect="auto",
        extent=extent,
    )
    ax_el.set_title("Elevation")
    ax_el.set_xlabel("Elevation (mm)")
    fig.tight_layout()

    def update(frame):
        img_lat.set_array(image_data[frame, :, image_data.shape[2] // 2].T)
        img_el.set_array(image_data[frame, image_data.shape[1] // 2].T)
        return [img_lat, img_el]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=image_data.shape[0],
    )

    fpath.parent.mkdir(parents=True, exist_ok=True)
    ani.save(
        fpath.with_name(fpath.name + f"_{name}").with_suffix(".mp4"),
        writer="ffmpeg",
        fps=fps,
    )
    plt.close(fig)


def find_median_image_idx(images):
    """Find the median image index (unlikely to be artifactual).

    Parameters
    ----------
    images : np.ndarray (n_images, width, height, depth)
        The images to check.

    Returns
    -------
    image_idx : int
        The median image index.
    """
    order = np.argsort(images, axis=0)
    extremeness = np.mean(order, axis=(1, 2, 3))
    image_idx = np.argsort(extremeness)[extremeness.size // 2]
    return image_idx


def register_image(moving, static, affine):
    with redirect_stdout(log_dipy):
        moved, reg_affine = affine_registration(
            moving,
            static,
            moving_affine=affine,
            static_affine=affine,
            pipeline=["rigid"],
        )
    return moved, reg_affine


def register_images(image, fpath, save=True):
    image_data = np.array(image.dataobj).transpose(3, 0, 1, 2)
    static = image_data[find_median_image_idx(image_data)]
    image_reg = np.zeros_like(image_data)
    reg_affines = np.zeros((image_data.shape[0], 4, 4))

    out = Parallel(n_jobs=n_jobs)(
        delayed(register_image)(moving, static, pwd.affine)
        for moving in tqdm(image_data)
    )
    for i, (moved, reg_affine) in enumerate(out):
        image_reg[i] = moved
        reg_affines[i] = reg_affine
    transformations = _affine_to_quat(reg_affines)
    image_reg = nib.Nifti1Image(image_reg.transpose(1, 2, 3, 0), image.affine)
    if save:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        trans_labels = ["rx", "ry", "rz", "tx", "ty", "tz"]
        nib.save(image_reg, fpath)
        np.savetxt(fpath.as_posix().split(".")[0] + ".txt", transformations)
    return image_reg, transformations, trans_labels


def plot_registration_movement(transformations, trans_labels, fpath):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(transformations[:, :3] * 1000, label=trans_labels[:3])
    ax.set_ylim([-2, 2])
    ax2.plot(np.rad2deg(transformations[:, 3:]), label=trans_labels[3:])
    ax2.set_ylim([-10, 10])
    ax.set_xlabel("Power Doppler Frame")
    ax.set_ylabel("Displacement (mm)")
    fig.savefig(
        fpath.with_name(fpath.name + "_registration").with_suffix(".png"), dpi=300
    )


def rat_hrf(t_r, oversampling=50, time_length=32.0, onset=0.0):
    """Rat hemodynamic response function.

    Parameters
    ----------
    t_r : float
        Repitition time, in seconds (sampling period).

    oversampling : int, default=50
        Temporal oversampling factor.

    time_length : float, default=32
        HRF kernel length, in seconds.

    onset : float, default=0
        Onset of the response.

    Returns
    -------
    hrf : array of shape(length / t_r * oversampling, dtype=float)
         HRG sampling on the oversampled time grid.
    """
    return _gamma_difference_hrf(
        t_r,
        oversampling,
        time_length,
        onset,
        delay=3,
        undershoot=10.0,
        dispersion=0.5,
        u_dispersion=1,
        ratio=0,
    )


def fit_glm(events, pwd, timestamps, **kwargs):
    if "smoothing_fwhm" not in kwargs:
        kwargs["smoothing_fwhm"] = None
    glm = FirstLevelModel(
        minimize_memory=False,
        mask_img=False,
        smoothing_fwhm=kwargs.pop("smoothing_fwhm"),
        standardize=True,
    )
    if kwargs["hrf_model"] == "rat":
        kwargs["hrf_model"] = rat_hrf
    events = events[["trial_type", "onset", "duration"]].copy()
    events["trial_type"] = "event"
    design_matrix = make_first_level_design_matrix(
        np.array(timestamps), events, **kwargs
    )

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i] for i, column in enumerate(design_matrix.columns)
    }

    # time has to be first, then xyz
    glm = glm.fit(pwd, design_matrices=design_matrix)

    # if a different hrf is used annoyingly that changes the
    # name of the design matrix column, use indexing but check
    # that it is correct
    event_cols = [col for col in design_matrix.columns if col.startswith("event")]
    assert len(event_cols) == 1  # noqa: S101
    event_col = event_cols[0]

    zmap = glm.compute_contrast(basic_contrasts[event_col], output_type="stat")
    return zmap


def fit_glm_time_shift(events, pwd, timestamps, fpath, save=True, **kwargs):
    event_time_offsets = (
        kwargs.pop("event_time_offsets")
        if "event_time_offsets" in kwargs
        else np.arange(-12, 12.25, 0.25)
    )
    zmaps = []
    for event_time_offset in event_time_offsets:
        events_shifted = events.copy()
        events_shifted["onset"] += event_time_offset
        zmaps.append(fit_glm(events_shifted, pwd, timestamps, **kwargs))
    if save:
        fpath.parent.mkdir(parents=True, exist_ok=True)
        nib.save(zmaps[np.argmin(np.abs(event_time_offsets))], fpath)
    return zmaps


def find_best_zmap(zmaps, n_best_vox, event_time_offsets, fpath, save=True):
    zmap_data = np.array([np.array(zmap.dataobj).ravel() for zmap in zmaps])
    zmap_best_idxs = np.argsort(zmap_data.max(axis=0))[-n_best_vox:]
    zmean = zmap_data[:, zmap_best_idxs].mean(axis=1)
    best_idx = np.argmax(np.abs(zmean))
    best_offset = event_time_offsets[best_idx]
    best_zmap = zmaps[best_idx]
    if save:
        nib.save(best_zmap, fpath)
    return best_zmap, best_offset


def plot_glm_best_voxel_quality_control(zmaps, n_best_vox, event_time_offsets, fpath):
    zmap_data = np.array([np.array(zmap.dataobj).ravel() for zmap in zmaps])
    zmap_best_idxs = np.argsort(zmap_data.max(axis=0))[-n_best_vox:]
    zmean = zmap_data[:, zmap_best_idxs].mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(event_time_offsets, zmap_data[:, zmap_best_idxs])
    ax.plot(event_time_offsets, zmean, color="black", linewidth=3)
    ax.set_xlabel("Time Shift (s)")
    ax.set_ylabel("T-Statistic")
    fig.savefig(fpath.with_name(fpath.name + "_timeshift").with_suffix(".png"), dpi=300)


# %% ###################################
# ######## Start of the script #########
########################################

# First, find subject, session, run and sequence (multi) folders

print("Locating data files and finding experiment start times")
subject_folders = list(sourcedata.glob("sub-*"))
session_folders = [
    session_dir
    for subject_dir in subject_folders
    for session_dir in subject_dir.glob("ses-*")
]
run_folders = [
    run_folder
    for session_folder in session_folders
    for run_folder in session_folder.glob(run_folder_glob)
]
exp_start_times = {
    run_folder: get_exp_start_time(run_folder) for run_folder in run_folders
}
run_folders = [run_folder for run_folder in run_folders if exp_start_times[run_folder]]
sequence_folders = [
    sequence_folder
    for run_folder in run_folders
    for sequence_folder in run_folder.glob(f"acquisitions/{sequence_folder_glob}")
]


# %%
# Write out the event timing (shared across multi sequence)

print("Writing events to BIDS")
for run_folder in tqdm(run_folders):
    base_name = run_folder.parts[-1]
    ids = dict(pair.split("-") for pair in base_name.split("_"))
    data_type = "angio" if ids["task"] == "rest" else "fus"
    out_dir = (
        bids_root / "rawdata" / f"sub-{ids['sub']}" / f"ses-{ids['ses']}" / data_type
    )

    if "fallingblocks" in run_folder.as_posix():
        log_fpath = run_folder / "logs" / "task_falling_blocks.log"
        events = load_log_events(log_fpath, exp_start_times[run_folder])
    else:
        event_fpath = run_folder / "streams" / "task-event_stream.h5"
        if not event_fpath.exists():
            event_fpath = event_fpath.with_name("task_1-event_stream.h5")
        if not event_fpath.exists():  # angio recording
            continue
        events = load_events(event_fpath)
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / f"{base_name}_events.tsv", sep="\t", index=False)


# %%
# Next, read metadata and use it to generate file paths so that
# multi-sequences have unique names

print(
    "Reading metadata and generating acq (sequence) hashes for unique multi-sequence file names"
)
result = Parallel(n_jobs=n_jobs)(
    delayed(get_sequence_folder_info)(sequence_folder)
    for sequence_folder in tqdm(sequence_folders)
)
# info per sequence folder: metadata, fpath_upd, raw_data_files (ensembles)
sequence_folder_info = dict(zip(sequence_folders, result))

with open(fusi_analysis_pipeline_folder / "no_metadata.txt", "w") as fid:
    for sequence_folder in sequence_folders:
        if sequence_folder_info[sequence_folder] is None:
            fid.write(f"{sequence_folder}\n")
            sequence_folder_info.pop(sequence_folder)
sequence_folders = list(sequence_folder_info)

with open(fusi_analysis_pipeline_folder / "no_data.txt", "w") as fid:
    for sequence_folder in sequence_folders:
        if not sequence_folder_info[sequence_folder][1]:
            fid.write(f"{sequence_folder}\n")
            sequence_folder_info.pop(sequence_folder)
sequence_folders = list(sequence_folder_info)

# %%
# Group sequence folders by angio or fus
# Note: sometimes for multi-sequence (angio), there is more than
# one ensemble collected but not more than 3
sequence_folders_fus = [
    sequence_folder
    for sequence_folder in sequence_folders
    if "task-rest" not in sequence_folder.parts[-3]
]
sequence_folders_angio = [
    sequence_folder
    for sequence_folder in sequence_folders
    if "task-rest" in sequence_folder.parts[-3]
]

# %%
# For each functional sequence, decode butterfly format data

if (bids_root / "rawdata" / "scans.tsv").exists():
    scans = pd.read_csv(bids_root / "rawdata" / "scans.tsv", sep="\t")
else:
    scans = pd.DataFrame(columns=["filename", "acq_time", "acq_hash"])

# process power doppler series in series at the sequence folder
# level so they can be in parallel at the ensemble level
print(
    "Converting functional experiment data from Butterfly to sensors x time ultrasound pulse data (upd)"
)
for sequence_folder in tqdm(sequence_folders_fus):
    fpath_upd, raw_data_files, metadata, acq_hash = sequence_folder_info[
        sequence_folder
    ]
    if (
        fpath_upd.relative_to(bids_root).as_posix() in set(scans["filename"])
        and not overwrite
    ):
        continue
    metadata_json = write_raw(
        fpath_upd,
        raw_data_files,
        metadata,
        exp_start_times[sequence_folder.parent.parent],
        n_jobs=n_jobs,
    )
    if metadata_json is None:
        continue
    scans.loc[len(scans)] = {
        "filename": fpath_upd.relative_to(bids_root).as_posix(),
        "acq_time": exp_start_times[sequence_folder.parent.parent],
        "acq_hash": acq_hash,
        **metadata_json,
    }
    scans.to_csv(bids_root / "rawdata" / "scans.tsv", sep="\t", index=False)


# %%
# Beamform and power doppler functional data

scans = pd.read_csv(bids_root / "rawdata" / "scans.tsv", sep="\t")
if (bids_root / "sourcedata" / "scans.tsv").exists():
    scans_proc = pd.read_csv(bids_root / "sourcedata" / "scans.tsv", sep="\t")
else:
    scans_proc = pd.DataFrame(
        columns=[*scans.columns, "pwd_filename", "pwd_reg_filename"]
    )

print("Beamforming, clutter filtering and registering functional data")
for sequence_folder in tqdm(sequence_folders_fus):
    fpath_upd, _, metadata, _ = sequence_folder_info[sequence_folder]

    if fpath_upd.relative_to(bids_root).as_posix() not in set(scans["filename"]):
        print(f"{fpath_upd} not converted from Butterfly, skipping")
        continue

    fpaths = get_fpaths(fpath_upd)

    if (
        fpaths["pwd_reg"].relative_to(bids_root).as_posix()
        in set(scans_proc["pwd_reg_filename"])
        and not overwrite
    ):
        print(f"{fpath_upd} already processed, skipping")
        continue

    pwd, sidecar, df_benchmark = raw_to_power_doppler(
        fpath_upd, metadata, fpaths["bmode"], fpaths["pwd"]
    )
    plot_benchmark(df_benchmark, fpaths["plot"])

    plot_movie(pwd, sidecar, fpaths["plot"], "beforepwdt")

    pwd_reg, transformations, trans_labels = register_images(pwd, fpaths["pwd_reg"])

    plot_registration_movement(transformations, trans_labels, fpaths["plot"])
    plot_movie(pwd_reg, sidecar, fpaths["plot"], "afterpwdt")

    idx_acq = list(scans["filename"]).index(fpath_upd.relative_to(bids_root).as_posix())
    scans_proc.loc[len(scans_proc)] = {
        "pwd_filename": fpaths["pwd"].relative_to(bids_root).as_posix(),
        "pwd_reg_filename": fpaths["pwd_reg"].relative_to(bids_root).as_posix(),
        **scans.loc[idx_acq].to_dict(),
        **sidecar,
    }
    scans_proc.to_csv(bids_root / "sourcedata" / "scans.tsv", sep="\t", index=False)

# %%
# Compute GLMs

scans_proc = pd.read_csv(bids_root / "sourcedata" / "scans.tsv", sep="\t")
if (bids_root / "fusi-analysis-pipeline" / "scans.tsv").exists():
    scans_glm = pd.read_csv(
        bids_root / "fusi-analysis-pipeline" / "scans.tsv", sep="\t"
    )
else:
    scans_glm = pd.DataFrame(
        columns=[*scans_proc.columns, "zmap_filename", "zmap_best_filename"]
    )


def compute_glm(sequence_folder, scans_glm):
    fpath_upd = sequence_folder_info[sequence_folder][0]

    fpaths = get_fpaths(fpath_upd)

    if not fpaths["events"].exists():
        with open(bids_root / "glm_errors.txt", "a") as fid:
            fid.write(f"{fpaths['events']} events not converted\n")
        return

    if not fpaths["pwd_reg"].exists():
        with open(bids_root / "glm_errors.txt", "a") as fid:
            fid.write(f"{fpaths['pwd_reg']} registered power doppler not computed\n")
        return

    if (
        fpaths["zmap"].relative_to(bids_root).as_posix()
        in set(scans_glm["zmap_filename"])
        and not overwrite
    ):
        return

    events = pd.read_csv(fpaths["events"], sep="\t")
    fpaths["events_pwd"].parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(fpaths["events_pwd"], sep="\t", index=False)

    pwd_reg = nib.load(fpaths["pwd_reg"])
    transformations = np.loadtxt(fpaths["pwd_reg"].as_posix().split(".")[0] + ".txt")
    with open(fpaths["pwd"].as_posix().split(".")[0] + ".json") as fid:
        sidecar = json.load(fid)

    kwargs = glm_params.copy()
    kwargs.pop("n_best_vox")  # not for glm
    kwargs["add_regs"] = transformations
    kwargs["add_reg_names"] = ["rx", "ry", "rz", "tx", "ty", "tz"]
    zmaps = fit_glm_time_shift(
        events=events,
        pwd=pwd_reg,
        timestamps=sidecar["VolumeTiming"],
        fpath=fpaths["zmap"],
        **kwargs,
    )

    zmap_best, offset_best = find_best_zmap(
        zmaps,
        glm_params["n_best_vox"],
        glm_params["event_time_offsets"],
        Path(str(fpaths["zmap"]).replace("zmap", "zmapbest")),
    )

    plot_glm_best_voxel_quality_control(
        zmaps,
        glm_params["n_best_vox"],
        glm_params["event_time_offsets"],
        fpaths["plot"],
    )
    return (
        fpaths["zmap"].relative_to(bids_root).as_posix(),
        Path(str(fpaths["zmap"]).replace("zmap", "zmapbest"))
        .relative_to(bids_root)
        .as_posix(),
    )


print("Computing GLMs")
out = Parallel(n_jobs=n_jobs)(
    delayed(compute_glm)(sequence_folder)
    for sequence_folder in tqdm(sequence_folders_fus, scans_glm)
)
for sequence_folder, value in zip(sequence_folders_fus, out):
    if value is None:
        continue
    zmap_fpath, zmap_best_fpath = value
    fpath_upd = sequence_folder_info[sequence_folder][0]
    idx_acq = list(scans_proc["filename"]).index(
        fpath_upd.relative_to(bids_root).as_posix()
    )
    scans_glm.loc[len(scans_glm)] = {
        "zmap_filename": zmap_fpath,
        "zmap_best_filename": zmap_best_fpath,
        **scans_proc.loc[idx_acq].to_dict(),
    }
scans_glm.to_csv(
    bids_root / "fusi-analysis-pipeline" / "scans.tsv", sep="\t", index=False
)

# %%
# For each angio sequence, decode butterfly format data

if (bids_root / "rawdata" / "scans.tsv").exists():
    scans = pd.read_csv(bids_root / "rawdata" / "scans.tsv", sep="\t")
else:
    scans = pd.DataFrame(columns=["filename", "acq_time", "acq_hash"])

# process angiography, in parallel at sequence folder level
print(
    "Converting angio (multi-sequence) scans from Butterfly to sensors x time ultrasound pulse data (upd)"
)
for sequence_folder in tqdm(sequence_folders_angio):
    fpath_upd, raw_data_files, metadata, acq_hash = sequence_folder_info[
        sequence_folder
    ]
    if fpath_upd.relative_to(bids_root).as_posix() in set(scans["filename"]):
        continue
    metadata_json = write_raw(
        fpath_upd,
        raw_data_files[:1],
        metadata,
        exp_start_times[sequence_folder.parent.parent],
        n_jobs=1,
    )
    if metadata_json is None:
        continue
    scans.loc[len(scans.index)] = {
        "filename": fpath_upd.relative_to(bids_root).as_posix(),
        "acq_time": exp_start_times[sequence_folder.parent.parent],
        "acq_hash": acq_hash,
        **metadata_json,
    }
    scans.to_csv(bids_root / "rawdata" / "scans.tsv", sep="\t", index=False)

# %%
# Beamform and power doppler angio data

scans = pd.read_csv(bids_root / "rawdata" / "scans.tsv", sep="\t")
if (bids_root / "sourcedata" / "scans.tsv").exists():
    scans_proc = pd.read_csv(bids_root / "sourcedata" / "scans.tsv", sep="\t")
else:
    scans_proc = pd.DataFrame(
        columns=[*scans.columns, "pwd_filename", "pwd_reg_filename"]
    )

print("Beamforming and clutter filtering angio data")
for sequence_folder in tqdm(sequence_folders_angio):
    fpath_upd, _, metadata, _ = sequence_folder_info[sequence_folder]

    if fpath_upd.relative_to(bids_root).as_posix() not in set(scans["filename"]):
        print(f"{fpath_upd} not converted from Butterfly, skipping")
        continue

    fpaths = get_fpaths(fpath_upd)

    if fpaths["pwd"].relative_to(bids_root).as_posix() in set(
        scans_proc["pwd_filename"]
    ):
        print(f"{fpath_upd} already processed, skipping")
        continue

    pwd, sidecar, df_benchmark = raw_to_power_doppler(
        fpath_upd, metadata, fpaths["bmode"], fpaths["pwd"], save_bmodes=True
    )
    plot_benchmark(df_benchmark, fpaths["plot"])

    idx_acq = list(scans["filename"]).index(fpath_upd.relative_to(bids_root).as_posix())
    scans_proc.loc[len(scans_proc)] = {
        "bmode_filename": fpaths["bmode"].relative_to(bids_root).as_posix(),
        "pwd_filename": fpaths["pwd"].relative_to(bids_root).as_posix(),
        **scans.loc[idx_acq].to_dict(),
        **sidecar,
    }
    scans_proc.to_csv(bids_root / "sourcedata" / "scans.tsv", sep="\t", index=False)
