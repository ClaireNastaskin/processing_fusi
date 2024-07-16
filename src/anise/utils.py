from pathlib import Path
import numpy as np
import json
import pandas as pd
import re
import h5py
import nibabel as nib
from silx.io.dictdump import h5todict

import SimpleITK as sitk


def get_sidecar(bmode_h5, pwd_h5, time_stamps, 
                n_tc=None,
                task_name='',
                task_description='',
                institution_name='',
                institution_address='',
                institutional_department_name=''):
    
    metadata_bmode = h5todict(bmode_h5, path='/metadata')
    metadata_pwd = h5todict(pwd_h5)

    config = dict()
    for config_name in ('angle_sequence', 'poseidon', 'sequence_timing'):
        config_fname = pwd_h5.parent / '..' / 'metadata' / f'{config_name}_config.json'
        if config_fname.exists():
            with open(config_fname, 'r') as fid:
                config[config_name] = json.load(fid)

    # save list of clutter filter
    # filters must appear in the order they are applied.
    n_bntc = get_param('num_blood_and_tissue_components', str(pwd_h5))
    if n_bntc is None:
        n_bntc = metadata_bmode['sequence']['ensemble_length'].item()
    if n_tc is None:
        n_tc = get_param('num_tissue_components', str(pwd_h5))
    clutter_filter_raw = metadata_pwd['metadata']['power_doppler']['clutter_filter']['filters']
    clutter_filter = [
        {
            'FilterType': (clutter_filter_raw['filter_type'].item()
                           if 'filter_type' in clutter_filter_raw else
                           'SVD'),
            'LowThreshold': (clutter_filter_raw['num_tisse_components'][0]
                             if 'num_tissue_components' in clutter_filter_raw else
                             n_tc),
            'HighThreshold': (clutter_filter_raw['num_blood_and_tissue_components'][0]
                             if 'num_blood_and_tissue_components' in clutter_filter_raw else
                             n_bntc),
        }
    ]

    # core parameters, multiple fallbacks
    tr = 1 / metadata_bmode['sequence']['pulse_repetition_interval_s']
    tr_compound = 1 / metadata_bmode['sequence']['compound_bmode_repetition_interval_s']
    angles = [str(a) for a in
              metadata_bmode['sequence']['pulse_configs']['az_angles_deg']]
    data_mode = metadata_bmode['sequence']['data_mode'].item()
    xyz = list(metadata_bmode['transducer']['xyz'])
    if 'ensemble_length' in metadata_bmode['sequence']:
        loops = metadata_bmode['sequence']['ensemble_length'].item()
    else:
        loops = get_param('loops', str(bmode_h5), reverse=True)
    if 'extras' in metadata_bmode['sequence']:
        probe_central_frequency = metadata_bmode['sequence']['extras']['tx_freq_hz'] / 1e6
        gain = metadata_bmode['sequence']['extras']['gain_mode'].item()
        tx_cycles = metadata_bmode['sequence']['extras']['tx_cycles'].item()
        chip_repeats = metadata_bmode['sequence']['extras']['num_chip_repeats'].item()
        tgc_slope = metadata_bmode['sequence']['extras']['tgc_slope'].item()
        tgc_duration = metadata_bmode['sequence']['extras']['tgc_duration'].item()
        tgc_init_gain = metadata_bmode['sequence']['extras']['tgc_initial_gain_db'].item()
        power_mode = metadata_bmode['sequence']['extras']['power_mode'].item()
        tx_az_aperture = metadata_bmode['sequence']['extras']['tx_az_aperture']
        tx_el_aperture = metadata_bmode['sequence']['extras']['tx_el_aperture']
    elif 'poseidon' in config:
        probe_central_frequency = config['poseidon']['tx_freq_hz'] / 1e6
        gain = config['poseidon'].get('gain_mode')
        tx_cycles = config['poseidon'].get('tx_cycles')
        chip_repeats = config['poseidon'].get('num_chip_repeats')
        tgc_slope = config['poseidon'].get('tgc_slope')
        tgc_duration = config['poseidon'].get('tgc_duration')
        tgc_init_gain = config['poseidon'].get('tgc_initial_gain_db')
        power_mode = config['poseidon'].get('power_mode')
        tx_az_aperture = config['poseidon'].get('tx_az_aperture')
        tx_el_aperture = config['poseidon'].get('tx_el_aperture')
    else:
        probe_central_frequency = float(get_param('Hz', str(bmode_h5), reverse=True)) / 1e6
        gain = int(get_param('Gain', str(bmode_h5), reverse=True))
        tx_cycles = int(get_param('c', str(bmode_h5), reverse=True))
        power_mode = int(get_param('AFE', str(bmode_h5), reverse=True))
        chip_repeats = int(get_param('rp', str(bmode_h5), reverse=True))
        tgc_init_gain = int(get_param('TGC', str(bmode_h5)))
        tgc_duration = int(get_param('Tdur', str(bmode_h5), reverse=True))
        tgc_slope = tx_az_aperture = tx_el_aperture = 'n/a'
        
    # get the start time of the acquisition
    if 'scan_timing' in metadata_pwd['metadata']:
        acquire_start_datetime = metadata_pwd['metadata']['scan_timing']['acquire_start_datetime']
        ensemble_start_datetime = metadata_pwd['metadata']['scan_timing']['ensemble_start_datetime']
    else:
        # get the start time of the acquisition from filename
        timestamp  = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', str(pwd_h5))
        global_start_time = pd.to_datetime(timestamp.group(0).replace('-', ':'), 
                                           format='%Y:%m:%dT%H:%M:%S:%f')
        acquire_start_datetime = ensemble_start_datetime = global_start_time

    sidecar = dict(
        # Scanner and probe hardware
        Manufacturer='Butterfly',
        ProbeCentralFrequency = probe_central_frequency, #in MHz
        ProbeNumberOfElements = [len(metadata_bmode['transducer']['lateral_index']), 
                                 len(metadata_bmode['transducer']['elevation_index'])],
        ProbePitch = 0.208, # in mm
        ProbeRadiusOfCurvature = 0, # not curved
        ProbeElevationAperture=(metadata_bmode['transducer']['elevation'][-1] - 
                                metadata_bmode['transducer']['elevation'][0]),  # in mm
        ProbeElevationFocus= 'n/a', # in mm; not available
                                
        # Sequence specifics
        Depth = [metadata_pwd['depth'][0].round(7) * 1000,
                 metadata_pwd['depth'][-1].round(7) * 1000], #in mm
        Lateral = [metadata_pwd['lateral'][0].round(7) * 1000,
                   metadata_pwd['lateral'][-1].round(7) * 1000], #in mm
        Elevation = [metadata_pwd['elevation'][0].round(7) * 1000,
                     metadata_pwd['elevation'][-1].round(7) * 1000], #in mm
        UltrasoundPulseRepetitionFrequency = tr,
        PlaneWaveAngles = angles, # degrees
        UltrafastSamplingFrequency = tr_compound,
        DataMode = data_mode,
        Gain = gain,
        TxCycles = tx_cycles,
        ChipRepeats = chip_repeats,
        TGCSlope = tgc_slope,
        TGCDuration = tgc_duration,
        TGCInitGain = tgc_init_gain,
        AFE = power_mode,
        TxApertureMask = [str(val) for val in tx_az_aperture],
        TxElevationMask = [str(val) for val in tx_el_aperture],

        # Clutter filtering
        ClutterFilterWindowDuration = 'n/a', # in ms; not available yet
        ClutterFilters = clutter_filter, # see above

        # Power Doppler integration
        PowerDopplerIntegrationDuration = 'n/a', # in ms; not available yet
        # Number of (compound) B-Mode images in an ensemble
        EnsembleLoops = loops,

        # Timing parameters
        VolumeTiming = list(time_stamps), # in seconds
        VolumeTimingStart = str(acquire_start_datetime), # global time
        SliceEncodingDirection = xyz, 
        # DelayAfterTrigger = 'n/a' # required for acquisitions containing the pose entity

        # fUS task information
        TaskName = task_name,
        TaskDescription = task_description,

        # Institution information
        InstitutionName = institution_name,
        InstitutionAddress = institution_address,
        InstitutionalDepartmentName = institutional_department_name,
    )
    return sidecar

def get_param(param: str, fname: str, reverse: bool=False) -> str:
    """Get the tissue components from an ensemble file.

    Parameters
    ----------
    param : str
        The name of the parameter to match.
    fname : str
        The filename.
    reverse : bool
        Whether to reverse the search.

    Returns
    -------
    val : str
        The value of the parameter.
    """
    if reverse:
        param = param[::-1]
        fname = fname[::-1]
    str_match = re.search(f'{param}[=_-][-+]?[a-z0-9]+', fname)
    remove_leading = True
    if not str_match:
        str_match = re.search(f'{param}[=_-]?[-+]?[a-z0-9]+', fname)
        remove_leading = False
    if str_match:
        val = str_match.group().replace(param, '')
        if reverse:
            val = val[::-1]
        if remove_leading:
            val = val[1:]  # strip equals indicator
        return val
    return


def match_param(param: str, path_dir: Path) -> str:
    """Find a parameter match.

    Parameters
    ----------
    param : str
        The parameter to match.
    path_dir : pathlib.Path
        The path to check for matches.

    Returns
    -------
    val : str
        The value of the parameter.
    """
    param = param.lower()
    matches = [part[len(param) + 1:] for part in
               path_dir.parts if part.lower().startswith(param)]
    if len(matches) == 1:
        return matches[0]
    return