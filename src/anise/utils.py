from pathlib import Path
import numpy as np
import json
import pandas as pd
import re
import h5py
import nibabel as nib
from silx.io.dictdump import h5todict

import SimpleITK as sitk


def is_valid_hdf5(fname):
    """Check if a file is a readable hdf5.

    Parameters
    ----------
    fname : pathlib.Path
        The file path.

    Notes
    -----
    ``h5py.is_hdf5`` does not properly handle truncated files.
    """
    try:
        with h5py.File(fname, 'r') as _:
            pass
    except:
        return False
    return True


def get_nii_data(img):
    """Get data from nifti image, excluding any nan slices."""
    img_data = np.array(img.dataobj)
    return img_data[..., ~np.isnan(img_data).any(axis=range(0, img.ndim - 1))]


def get_bids_value(bids_path, keyword):
    """Get the keyword value of a BIDS named file.

    Parameters
    ----------
    bids_path : pathlib.Path
        The path to the BIDS file.
    keyword : str
        The keyword to search for.

    Returns
    -------
    value : str
        The value of the BIDS file for the keyword.
    """
    parts = bids_path.stem.split('.')[0].split('_')
    for part in parts:
        if len(part.split('-')) == 2:
            key, value = part.split('-')
            if keyword == key:
                return value
    sidecar_path = bids_path.parent / (bids_path.name.split('.')[0] + '.json')
    if sidecar_path.exists():
        with open(sidecar_path, 'r') as fid:
            sidecar = json.load(fid)
        if keyword in sidecar:
            return sidecar[keyword]
    raise RuntimeError(f'No keyword "{keyword}" found in {bids_path}')


def get_sidecar(bmode_h5, pwd_h5, time_stamps, n_tc,
                task_name='light',
                task_description='Blue LED, flashing at 5Hz',
                institution_name='Caltech',
                institution_address='1200 E California Blvd, Pasadena, CA 91125',
                institutional_department_name='Neuroscience - Biology and Biological Engineering'):
    
    metadata_bmode = h5todict(bmode_h5, path='/metadata')
    metadata_pwd = h5todict(pwd_h5)

    config = dict()
    for config_name in ('angle_sequence', 'poseidon', 'sequence_timing'):
        config_fname = pwd_h5.parent / '..' / 'metadata' / f'{config_name}_config.json'
        if config_fname.exists():
            with open(config_fname, 'r') as fid:
                config[config_name] = json.load(fid)

    n_bntc = get_param('num_blood_and_tissue_components', str(pwd_h5))
    clutter_filter_raw = metadata_pwd['metadata']['power_doppler']['clutter_filter']['filters']
    clutter_filter = [
        {
            'FilterType': (clutter_filter_raw['filter_type'].item()
                           if 'filter_type' in clutter_filter_raw else
                           'Fixed-threshold SVD'),
            'LowThreshold': (clutter_filter_raw['num_tisse_components'][0]
                             if 'num_tissue_components' in clutter_filter_raw else
                             n_tc),
            'HighThreshold': (metadata_bmode['sequence']['ensemble_length'].item()
                              if n_bntc is None else n_bntc)
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
        probe_central_frequency = float(get_param('RF', str(bmode_h5))) / 1e6
        gain = get_param('Gain', str(bmode_h5), reverse=True)
        tx_cycles = get_param('c', str(bmode_h5), reverse=True)
        power_mode = get_param('AFE', str(bmode_h5), reverse=True)
        chip_repeats = tgc_slope = tgc_duration = \
            tx_az_aperture = tx_el_aperture = 'n/a'


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
        Depth = [metadata_pwd['depth'][0].round(7),
                 metadata_pwd['depth'][-1].round(7)],
        Lateral = [metadata_pwd['lateral'][0].round(7),
                   metadata_pwd['lateral'][-1].round(7)],
        Elevation = [str(metadata_pwd['elevation'][0]),
                     str(metadata_pwd['elevation'][-1])],
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
        VolumeTimingStart = 0, # global time
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


def raw_to_power_doppler(raw_frame_path):
    """Raw data to power doppler."""
    pass


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
    matches = [part[len(param) + 1:] for part in
               path_dir.parts if part.lower().startswith(param)]
    if len(matches) == 1:
        return matches[0]
    return


def load_fusi_info(directory, **kwargs):
    """
    Scans the specified directory for h5 files, extracts timestamps
    from the filenames, and organizes this information into a DataFrame. 
    The DataFrame is sorted by timestamps and includes columns for
    relative experiment times and readable timestamps.
    
    Args:
    directory (str): The directory where the h5 files are stored.
    
    Returns:
    DataFrame: A DataFrame containing the filenames, original timestamps,
               experiment relative times, and readable timestamps.
    """
    filenames = [f.name for f in Path(directory).glob('*.h5') if
                 not f.name.startswith('.')]

    for param, value in kwargs.items():
        filenames = [f for f in filenames if get_param(param, f) == value]

    # fix bug where duplicates with and without ensemble prefix
    if any([f.startswith('ensemble') for f in filenames]):
        filenames = [f for f in filenames if f.startswith('ensemble')]

    if not filenames:
        return

    print(filenames)
    # Extract datetime from filenames and store data
    data = []
    for filename in filenames:
        timestamp = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', filename)
        timestamp
        if timestamp:
            data.append({
                'Filename': filename,
                'Timestamp': pd.to_datetime(timestamp.group(0).replace('-', ':'),
                                            format='%Y:%m:%dT%H:%M:%S:%f')
            })

    df = pd.DataFrame(data)
    # Sort the DataFrame by the 'Timestamp' column
    df = df.sort_values(by='Timestamp')

    # Calculate the relative times since the first event
    df['Experiment Time'] = \
        (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

    # Toss first acquisition if well before second
    if len(df) > 1 and (df.iloc[1]['Experiment Time'] -
                        df.iloc[0]['Experiment Time']) > 10:
        df = df.iloc[1:]  # toss first acquisition

    # format excluding the date and keeping the first digit of milliseconds
    df['Readable Timestamp'] = \
        df['Timestamp'].dt.strftime('%H:%M:%S.%f').str[:-5]

    # reset ordering
    df = df.reset_index().drop(columns=['index'])

    return df


''' Fix me: need to figure out shape of raw data to get this to work
def get_raw_nii(raw_path):
    """Fetches the data from raw frames.

    Args:
    raw_path (str): The raw frame directory path.

    Returns:
    nib.Nifti1Image: The data from the bmode images.
    np.ndarray: The time stamps of the images.
    """
    raw_info = load_fusi_info(raw_path)
    data = list()
    tr = None
    for file_path in bmode_info['Filename']:
        with h5py.File(file_path, 'r') as file:
            data.append(file['beamformed'][:])
            assert tr is None or file['metadata']['sequence'][
                'compound_bmode_repetition_interval_s'][()] == tr
            tr = file['metadata']['sequence'][
                'compound_bmode_repetition_interval_s'][()]
    n_loops = data[0].shape[0]
    data = np.flip(np.vstack(data).transpose((1, 2, 3, 0)), axis=2)
    time_stamps = np.array([t + tr * i for t in bmode_info['Experiment Time']
                            for i in range(n_loops)])
    return nib.Nifti1Image(data, np.diag([0.15, 0.15, 0.15, 1])), time_stamps
'''


def get_bmode_nii(bmode_path):
    """Fetches the data from bmode files.

    Args:
    bmode_path (str): The bmode directory path.

    Returns:
    nib.Nifti1Image: The data from the bmode images.
    np.ndarray: The time stamps of the images.
    """
    bmode_info = load_fusi_info(bmode_path)
    data = list()
    tr = None
    for file_name in bmode_info['Filename']:
        with h5py.File(bmode_path / file_name, 'r') as file:
            data.append(file['beamformed'][:])
            assert tr is None or file['metadata']['sequence'][
                'compound_bmode_repetition_interval_s'][()] == tr
            tr = file['metadata']['sequence'][
                'compound_bmode_repetition_interval_s'][()]
    n_loops = data[0].shape[0]
    data = np.flip(np.vstack(data).transpose((1, 2, 3, 0)), axis=2)
    time_stamps = np.array([t + tr * i for t in bmode_info['Experiment Time']
                            for i in range(n_loops)])
    return nib.Nifti1Image(data, np.diag([0.15, 0.15, 0.15, 1])), time_stamps


def get_power_doppler_nii(power_doppler_path, num_tissue_components):
    """Fetches the data from power doppler files.

    Args:
    power_doppler_path (str): The directory where the h5 files are stored.

    Returns:
    nib.Nifti1Image: The data from the power doppler images.
    np.ndarray: The time stamps for the start of the image.
    """
    fusi_info = load_fusi_info(
        power_doppler_path, num_tissue_components=num_tissue_components)
    fusi_data = get_fusi_frames(power_doppler_path, fusi_info)
    return (nib.Nifti1Image(fusi_data, np.diag([0.15, 0.15, 0.15, 1])),
            fusi_info['Experiment Time'])


def get_fusi_frames(power_doppler_path, power_doppler_df, frame_indices=-1):
    """
    Fetches the data from specific frames in the h5 files based on the list of frame indices.
    
    Args:
    power_doppler_path (str): The directory where the h5 files are stored.
    power_doppler_df (DataFrame): DataFrame containing the filenames and timestamps.
    frame_indices (list of int): The indices of the frames to fetch.
    
    Returns:
    np.array: The data from the specified frames in the h5 files,
              stacked along the last dimension.
    """
    # Check if all frame_indices are valid
    if frame_indices == -1:
        frame_indices = list(range(len(power_doppler_df)))
    elif any(frame_index < 0 or frame_index >= len(power_doppler_df)
             for frame_index in frame_indices):
        raise ValueError("One or more invalid frame indices")
    
    # Initialize a list to hold the data arrays
    data_list = []

    for frame_index in frame_indices:
        # Get the filename for the desired frame
        filename = power_doppler_df.iloc[frame_index]['Filename']
        file_path = Path(power_doppler_path) / filename
        # Load the h5 file
        with h5py.File(file_path, 'r') as file:
            data = file['power_doppler'][:]
            data_list.append(data)
    
    # Stack the data arrays along the last dimension,
    # reorient third dimension correctly
    stacked_data = np.flip(np.stack(data_list, axis=-1), axis=2)

    return stacked_data


def register_image_stack(image_stack):
    sitk_images = [sitk.GetImageFromArray(image_stack[..., i])
                   for i in range(image_stack.shape[-1])]
    fixed_image = sitk_images[0]
    registered_images = [fixed_image]
    
    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_stack.shape[-1]))  # Initialize with zeros
    transform_params[0, :] = 0  # If the last row is unused, set it to 1 or some default value

    # Setup registration method to be more constrained
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()  # Using correlation metric for small motions
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1, minStep=1e-4, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    initial_transform = sitk.Euler2DTransform()
    initial_transform.SetIdentity()
    registration_method.SetInitialTransform(initial_transform)

    # Apply registration
    for i, moving_image in enumerate(sitk_images[1:], start=1):
        # Set a tighter initial transform using moments
        initial_transform = sitk.Euler2DTransform(
            sitk.CenteredTransformInitializer(
                fixed_image, moving_image, sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS
            )
        )
        registration_method.SetInitialTransform(initial_transform)

        final_transform = registration_method.Execute(fixed_image, moving_image)
        tx, ty = final_transform.GetTranslation()
        angle = final_transform.GetAngle()
        
        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        transform_params[2, i] = angle

        resampled_image = sitk.Resample(
            moving_image, fixed_image, final_transform,
            sitk.sitkLinear, 0.0, moving_image.GetPixelID()
        )
        registered_images.append(resampled_image)
    
    registered_array = np.stack([sitk.GetArrayFromImage(img)
                                 for img in registered_images], axis=-1)
    
    return registered_array, transform_params
