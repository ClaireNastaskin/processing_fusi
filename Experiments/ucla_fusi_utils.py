from nilearn import plotting



import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import pandas as pd
import re
import h5py
from pathlib import Path
from datetime import datetime, timedelta
import json

import nilearn as nl
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

from anise.io.load_matlab_dataset import load_data, load_selected_data
import SimpleITK as sitk



class DirectoryManager:
    def __init__(self, base_path, sequence):
        self.base_path = base_path
        self.sequence = sequence
        self.log_file_path = self.base_path / 'logs' / 'task_1.log'
        self.task_event_file_path = self.base_path / 'streams' / 'daq_1_event_stream.h5'
        self.probe_event_file_path = self.base_path / 'streams' / 'probe_1_event_stream.h5'
        self.sequence_data_path = self.base_path / 'acquisitions' / self.sequence
        self.raw_data_path = self.sequence_data_path / 'raw_frame_data'
        self.beamformed_path = self.sequence_data_path / 'beamformed'
        self.power_doppler_path = self.sequence_data_path / 'power_doppler'
        self.meta_data_path = self.power_doppler_path  # This seems to be a mistake in the original code, corrected here
        self.fUSI_data_path = self.sequence_data_path / 'fUSI'
        print(self.fUSI_data_path)
        # Ensure fUSI_data_path exists
        if not self.fUSI_data_path.exists():
            self.fUSI_data_path.mkdir(parents=True, exist_ok=True)

    def list_acquisition_directories(self):
        acquisitions_path = self.base_path / 'acquisitions'
        directories = [d.name for d in acquisitions_path.iterdir() if d.is_dir()]
        print("Directories in acquisitions path:")
        for directory in directories:
            print(f'    {directory}')

    def print_paths(self):
        print("List of all directories in acquisitions path:")
        print(f'    log_file_path: {self.log_file_path}')
        print(f'    task_event_file_path: {self.task_event_file_path}')
        print(f'    probe_event_file_path: {self.probe_event_file_path}')
        print(f'    sequence_data_path: {self.sequence_data_path}')
        print(f'    raw_data_path: {self.raw_data_path}')
        print(f'    beamformed_path: {self.beamformed_path}')
        print(f'    power_doppler_path: {self.power_doppler_path}')
        print(f'    meta_data_path: {self.meta_data_path}')
        print(f'    USI_data_path: {self.fUSI_data_path}')


import imageio
import os
from IPython.display import Video

def create_video_from_3d_data(data_3d, output_path='movie_test.mp4', fps=10):
    """
    Create a video from 3D data array where each slice along the last axis is a frame in the video.

    Args:
    data_3d (numpy.ndarray): The 3D array where each slice along the last axis is an image frame.
    output_path (str): The file path where the video will be saved.
    fps (int): Frames per second for the output video.
    """
    writer = imageio.get_writer(output_path, fps=fps)
    for i in range(data_3d.shape[2]):
        writer.append_data(data_3d[:, :, i])
    writer.close()
    return Video(output_path)

def load_fUSi_info(directory,num_tissue_components=50):
    """
    Scans the specified directory for h5 files, extracts timestamps from the filenames, and organizes this information into a DataFrame. 
    The DataFrame is sorted by timestamps and includes columns for relative experiment times and readable timestamps.
    
    Args:
    directory (str): The directory where the h5 files are stored.
    
    Returns:
    DataFrame: A DataFrame containing the filenames, original timestamps, experiment relative times, and readable timestamps.
    """
    filenames = [f for f in os.listdir(directory) if f.endswith('.h5') and f'num_tissue_components={num_tissue_components}' in f]
    print(filenames)
    # Extract datetime from filenames and store data
    data = []
    for filename in filenames:
        timestamp = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', filename)
        timestamp
        if timestamp:
            data.append({
                'Filename': filename,
                'Timestamp': pd.to_datetime(timestamp.group(0).replace('-', ':'), format='%Y:%m:%dT%H:%M:%S:%f')
            })
    

    power_doppler_df=pd.DataFrame(data)
    # Sort the DataFrame by the 'Timestamp' column
    power_doppler_df = power_doppler_df.sort_values(by='Timestamp')

    # Calculate the relative times since the first event
    power_doppler_df['Experiment Time'] = (power_doppler_df['Timestamp'] - power_doppler_df['Timestamp'].iloc[0]).dt.total_seconds()

    #  format excluding the date and keeping the first digit of milliseconds
    power_doppler_df['Readable Timestamp'] = power_doppler_df['Timestamp'].dt.strftime('%H:%M:%S.%f').str[:-5]

    return power_doppler_df


def process_task_log(log_file_path,fusi_info,timeoffset=0):
    """
    Reads a log file, extracts timestamps and event descriptions from each line, and organizes this information into a DataFrame.
    The DataFrame is sorted by timestamps and includes columns for relative experiment times and readable timestamps.
    
    Args:
    log_file_path (str): The path to the log file containing event data.
    
    Returns:
    DataFrame: A DataFrame containing the timestamps, event descriptions, experiment relative times, and readable timestamps.
    """
    # Read the entire log file
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    #  parse lines
    data = []
    for line in lines:
        # Extract timestamp and event description
        timestamp = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2}', line)
        # timestamp = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2}', line)
        event = re.search(r'Stimulus (on|off)', line)
        if timestamp and event:
            data.append({
                'Timestamp': pd.to_datetime(timestamp.group(0)).tz_localize(None)- timedelta(hours=timeoffset),
                # 'Timestamp': pd.to_datetime(timestamp.group(0)),
                'Event': event.group(0)
            })

    behavior_df=pd.DataFrame(data)     
    # Calculate the relative times since the first event
    behavior_df['Experiment Time'] = (behavior_df['Timestamp'] - fusi_info['Timestamp'].iloc[0]).dt.total_seconds()      
    #  format excluding the date and keeping the first digit of milliseconds
    behavior_df['Readable Timestamp'] = behavior_df['Timestamp'].dt.strftime('%H:%M:%S.%f').str[:-5]
    return behavior_df


def calculate_stimulus_events(behavior_df):
    """
    Analyzes a DataFrame containing behavioral event data to compute the durations of stimulus events.
    It extracts the times when the stimulus was turned on and off, calculates the duration for each stimulus event, and returns a DataFrame with the stimulus conditions, onset times, and durations.
    
    Args:
    behavior_df (DataFrame): A DataFrame with columns for timestamps, event descriptions, and relative experiment times.
    
    Returns:
    DataFrame: A DataFrame containing the trial type, onset times, and durations of each stimulus event.
    """
    
    on_times = behavior_df[behavior_df['Event'] == 'Stimulus on']['Experiment Time'].reset_index(drop=True)
    off_times = behavior_df[behavior_df['Event'] == 'Stimulus off']['Experiment Time'].reset_index(drop=True)

    # Calculating the duration for which the stimulus was on
    if len(on_times) == len(off_times):
        stimulus_durations = off_times - on_times
    else:
        print("Mismatch in 'on' and 'off' events count.")
        return None

    # Extract conditions and onset times
    stimulus_df = behavior_df[behavior_df['Event'] == 'Stimulus on']
    conditions = stimulus_df['Event'].tolist()
    onsets = stimulus_df['Experiment Time'].tolist()

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": stimulus_durations}
    )

    return events


def get_fusi_frames(power_doppler_path, power_doppler_df, frame_indices=-1):
    """
    Fetches the data from specific frames in the h5 files based on the list of frame indices.
    
    Args:
    power_doppler_path (str): The directory where the h5 files are stored.
    power_doppler_df (DataFrame): DataFrame containing the filenames and timestamps.
    frame_indices (list of int): The indices of the frames to fetch.
    
    Returns:
    np.array: The data from the specified frames in the h5 files, stacked along the last dimension.
    """
    # Check if all frame_indices are valid
    if frame_indices == -1:
        frame_indices = list(range(len(power_doppler_df)))
    elif any(frame_index < 0 or frame_index >= len(power_doppler_df) for frame_index in frame_indices):
        raise ValueError("One or more invalid frame indices")
    
    # Initialize a list to hold the data arrays
    data_list = []
    
    for frame_index in frame_indices:
        # Get the filename for the desired frame
        filename = power_doppler_df.iloc[frame_index]['Filename']
        file_path = os.path.join(power_doppler_path, filename)
        # print(file_path)
        # Load the h5 file
        with h5py.File(file_path, 'r') as file:
            # Assuming the dataset name in the h5 file is 'beamformed'
            # data = file['beamformed'][:]
            data = file['power_doppler'][:]
            data_list.append(data)
    
    # Stack the data arrays along the last dimension
    stacked_data = np.stack(data_list, axis=-1)
    
    N=2
    frames_data_replicated = np.tile(stacked_data, (1, N, 1, 1))
    frames_data_replicated = np.flip(frames_data_replicated, axis=2) # get in the right coordinate system

    return frames_data_replicated



def register_image_stack(image_stack):
    sitk_images = [sitk.GetImageFromArray(image_stack[:, :, i]) for i in range(image_stack.shape[2])]
    fixed_image = sitk_images[0]
    registered_images = [fixed_image]
    
    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_stack.shape[2]))  # Initialize with zeros
    transform_params[0, :] = 0  # If the last row is unused, set it to 1 or some default value

    # Setup registration method to be more constrained
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()  # Using correlation metric for small motions
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    initial_transform = sitk.Euler2DTransform()
    initial_transform.SetIdentity()
    registration_method.SetInitialTransform(initial_transform)

    # Apply registration
    for i, moving_image in enumerate(sitk_images[1:], start=1):
        # Set a tighter initial transform using moments
        initial_transform = sitk.Euler2DTransform(sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS))
        registration_method.SetInitialTransform(initial_transform)

        final_transform = registration_method.Execute(fixed_image, moving_image)
        tx, ty = final_transform.GetTranslation()
        angle = final_transform.GetAngle()
        
        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        transform_params[2, i] = angle

        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        registered_images.append(resampled_image)
    
    registered_array = np.stack([sitk.GetArrayFromImage(img) for img in registered_images], axis=2)
    
    return registered_array, transform_params


import ants

def register_ants(image_stack):
    fixed_image = ants.from_numpy(image_stack[..., 0])
    transformed_images = [fixed_image.numpy()]
    # Initialize transformations array with zero for the first image (identity transformation)
    transformations = np.zeros((3, image_stack.shape[2]))  # 3 rows for tx, ty, angle

    for i in range(1, image_stack.shape[2]):
        moving_image = ants.from_numpy(image_stack[..., i])
        result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')

        # Extract the transformed image for visualization or further analysis
        transformed_image = result['warpedmovout']
        transformed_images.append(transformed_image.numpy())

        # Extract parameters from the transform
        transform = result['fwdtransforms'][0] if len(result['fwdtransforms']) > 0 else None
        if transform:
            # Using ANTsPy API to extract parameters
            tx, ty, angle = extract_parameters(transform)
            transformations[:, i] = [tx, ty, angle]
        else:
            # Append zeros if no transform is found (though it should find one)
            transformations[:, i] = [0, 0, 0]

    registered_stack = np.stack(transformed_images, axis=-1)
    return registered_stack, transformations

def extract_parameters(transform):
    # This function will adapt based on the available methods in the ANTsTransform object
    transform_object = ants.read_transform(transform)
    tx, ty = transform_object.parameters[4:6]  # Assuming these indices contain translations
    angle = transform_object.parameters[2]  # Assuming this index contains the rotation angle
    return tx, ty, angle




def extract_events_from_h5(h5_file_path):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as file:
        # Extract data from the 'data' dataset
        dataset = file['data'][:]
        
        # Extract fields
        events = dataset['event']
        timestamps = dataset['timestamp']

    # Convert data to DataFrame
    data = []
    for i in range(len(timestamps)):
        data.append({
            'Event': events[i].decode('utf-8'),
            'Timestamp': timestamps[i]
            
        })

    behavior_df = pd.DataFrame(data)

    return behavior_df





def calculate_stimulus_eventsV2(behavior_df):
    """
    Analyzes a DataFrame containing behavioral event data to compute the durations of stimulus events.
    It extracts the times when the stimulus was turned on and off, calculates the duration for each stimulus event, and returns a DataFrame with the stimulus conditions, onset times, and durations.
    
    Args:
    behavior_df (DataFrame): A DataFrame with columns for timestamps, event descriptions, and relative experiment times.
    
    Returns:
    DataFrame: A DataFrame containing the trial type, onset times, and durations of each stimulus event.
    """
    
    on_times = behavior_df[behavior_df['Event'] == 'stimulus_onset']['Timestamp'].reset_index(drop=True)
    off_times = behavior_df[behavior_df['Event'] == 'stimulus_offset']['Timestamp'].reset_index(drop=True)

    # Calculating the duration for which the stimulus was on
    if len(on_times) == len(off_times):
        stimulus_durations = off_times - on_times
    else:
        print("Mismatch in 'on' and 'off' events count.")
        return None

    # Extract conditions and onset times
    stimulus_df = behavior_df[behavior_df['Event'] == 'stimulus_onset']
    conditions = stimulus_df['Event'].tolist()
    onsets = stimulus_df['Timestamp'].tolist()

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": stimulus_durations}
    )

    return events



def extract_probe_events_from_h5(h5_file_path):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as file:
        # Extract data from the 'data' dataset
        dataset = file['data'][:]
        
        # Extract fields
        events = dataset['event']
        timestamps = dataset['timestamp']
        payload = dataset['payload']
    

    # Convert data to DataFrame
    data = []
    for i in range(len(timestamps)):
        

        event=events[i].decode('utf-8')

        if event=='frame_saved':
            payload_data = json.loads(payload[i])


            # Extract the filename from the output_path
            file_name = os.path.basename(payload_data["output_path"])


            data.append({
                'Event': events[i].decode('utf-8'),
                'raw_file_name': file_name,
                'raw_output_path': payload_data["output_path"],
                'acquisition_idx': payload_data["acquisition_idx"],
                'time_stamp': timestamps[i]
                
            })

    behavior_df = pd.DataFrame(data)
    # Filter behavior_df for rows where the 'Event' column is 'frame_saved'

    return behavior_df





def calculate_stimulus_events_caltech_daq(behavior_df):
    """
    Analyzes a DataFrame containing behavioral event data to compute the durations of stimulus events.
    It extracts the times when the stimulus was turned on and off, calculates the duration for each stimulus event, and returns a DataFrame with the stimulus conditions, onset times, and durations.
    
    Args:
    behavior_df (DataFrame): A DataFrame with columns for timestamps, event descriptions, and relative experiment times.
    
    Returns:
    DataFrame: A DataFrame containing the trial type, onset times, and durations of each stimulus event.
    """
    
    on_times = behavior_df[behavior_df['Event'] == 'pwm_enabled']['Timestamp'].reset_index(drop=True)
    off_times = behavior_df[behavior_df['Event'] == 'pwm_disabled']['Timestamp'].reset_index(drop=True)

    # Calculating the duration for which the stimulus was on
    if len(on_times) == len(off_times):
        stimulus_durations = off_times - on_times
    else:
        print("Mismatch in 'on' and 'off' events count.")
        stimulus_durations = off_times - on_times[0:-1]
        # stimulus_durations = stimulus_durations.append(pd.Series([stimulus_durations.mean()]))
        stimulus_durations = pd.concat([stimulus_durations, pd.Series([stimulus_durations.mean()])], ignore_index=True)

        # return None

    # Extract conditions and onset times
    stimulus_df = behavior_df[behavior_df['Event'] == 'pwm_enabled']
    conditions = stimulus_df['Event'].tolist()
    onsets = stimulus_df['Timestamp'].tolist()

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": stimulus_durations}
    )

    return events
