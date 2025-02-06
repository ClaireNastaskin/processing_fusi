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





def motion_correct_with_phase_correlation(imgs, im_ref=None):
    ir_all = None
    imgs_shifted = None
    if im_ref is None:
        im_ref = imgs[0]
    for img in imgs:
        x_shift_est, y_shift_est, ir = phase_corr_translation(im_ref, img)
        img2 = shift(img, (x_shift_est,y_shift_est))  
        img2 = np.expand_dims(img2, axis=0)
        if imgs_shifted is None:
            imgs_shifted = img2
        else:
            imgs_shifted = np.concatenate((imgs_shifted, img2),axis=0)
        ir = np.expand_dims(ir, axis=0)
        if ir_all is None:
            ir_all = ir
        else:
            ir_all = np.concatenate((ir_all, ir),axis=0)
    
    return imgs_shifted, ir_all

def pairwise_cross_correlation_variance(imgs):
    """
    Args:
        imgs: 3D numpy array (time x depth x lateral): stack of images to be aligned
    returns:
        ir_avg: 2D numpy array : average of the phase correlation between all pairs of images

    """
    ir_avg = np.zeros([imgs.shape[0],imgs.shape[0]])
    shape = imgs.shape
    fft_imgs = np.zeros([imgs.shape[0],imgs.shape[1],imgs.shape[2]], dtype=np.complex_)
    for i in range(len(imgs)):
        fft_imgs[i] = fft2(imgs[i])

    for i in range(len(imgs)):
        for j in range(i+1,len(imgs)):
            ir = abs(ifft2((fft_imgs[i] * fft_imgs[j].conjugate()) / (abs(fft_imgs[i]) * abs(fft_imgs[j]))))
            ir = np.expand_dims(ir, axis=0)
            ir_avg[i,j] = ir.var()
    return ir_avg




from numpy.fft import fft2, ifft2, fftshift
def phase_corr_translation(im1, im2):
    
    """ FFT phase correlation
    Kuglin, C. D. and Hines, D. C., 1975. The Phase Correlation Image Alignment Method. 
    Proceeding of IEEE International Conference on Cybernetics and Society, pp. 163-165, New York, NY, USA.

    returns
    -------
        ir normalized cross-correlation

    """
    shape = im2.shape
    f1 = fft2(im1)
    f2 = fft2(im2)
    ir = abs(ifft2((f1 * f2.conjugate()) / (abs(f1) * abs(f2))))
    x, y = np.unravel_index(np.argmax(ir), shape)
    if x > shape[0] // 2:
        x -= shape[0]
    if y > shape[1] // 2:
        y -= shape[1]
    return x, y, ir

def register_phase_corr(image_stack):
    # sitk_images = [sitk.GetImageFromArray(image_stack[:, :, i]) for i in range(image_stack.shape[2])]
    fixed_image = image_stack[..., 0]
    registered_images = [fixed_image]
    
    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_stack.shape[2]))  # Initialize with zeros
    metric_value = np.zeros((1, image_stack.shape[2]))  # Initialize with zeros

    transform_params[0, :] = 0  # If the last row is unused, set it to 1 or some default value
    
    
    
    # Apply registration
    for i in range(1, image_stack.shape[2]):
        moving_image = image_stack[..., i]
    # Ensure moving_image is correctly reshaped to match fixed_image dimensions
        # moving_image = moving_image[:fixed_image.shape[0], :fixed_image.shape[1]]
        
        tx,ty,metric_value_i=phase_corr_translation(fixed_image, moving_image)
    
        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        # transform_params[2, i] = angle

        metric_value[0,i] = metric_value_i

        resampled_image = sitk.Resample(moving_image, fixed_image, transform_params[:,i], sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        registered_images.append(resampled_image)
    
    registered_array = np.stack([sitk.GetArrayFromImage(img) for img in registered_images], axis=2)
    
    return registered_array, transform_params, metric_value





class DirectoryManager:
    def __init__(self, base_path, sequence):
        self.base_path = base_path
        self.sequence = sequence
        self.log_file_path = self.base_path / 'logs' / 'task_1.log'
        self.task_event_file_path = self.base_path / 'streams' / 'task_1-event_stream.h5'
        self.probe_event_file_path = self.base_path / 'streams' / 'probe_1-event_stream.h5'
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





def extract_task_events(task_event_stream):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(task_event_stream, 'r') as file:
        # Extract data from the 'data' dataset
        dataset = file['data'][:]
        
        # Extract fields
        events = dataset['event']
        timestamps = dataset['timestamp']
        payload=dataset["payload"]

    # Convert data to DataFrame
    data = []
    for i in range(len(timestamps)):
        event_decoded = events[i].decode('utf-8')
        if event_decoded == 'start_playing' or event_decoded == 'stop_playing':
            stimulus_payload = json.loads(payload[i].decode('utf-8'))
            data.append({
                'Event': event_decoded,
                'Timestamp': timestamps[i],
                'Stimulus': stimulus_payload["stimulus"][:-4]
            })
        elif event_decoded == 'stimulus_onset' or event_decoded == 'stimulus_offset':
            stimulus_payload = json.loads(payload[i].decode('utf-8'))
            data.append({
                'Event': event_decoded,
                'Timestamp': timestamps[i],
                'Stimulus': stimulus_payload["task"]
            })

    behavior_df = pd.DataFrame(data)


    events=extract_nilearn_compatible_events(behavior_df)

    return events

def extract_nilearn_compatible_events(behavior_df):
    """
    Analyzes a DataFrame containing behavioral event data to compute the durations of stimulus events.
    It extracts the times when the stimulus was turned on and off, calculates the duration for each stimulus event, and returns a DataFrame with the stimulus conditions, onset times, and durations.
    
    Args:
    behavior_df (DataFrame): A DataFrame with columns for timestamps, event descriptions, and relative experiment times.
    
    Returns:
    DataFrame: A DataFrame containing the trial type, onset times, and durations of each stimulus event.
    """
    if 'start_playing' in behavior_df['Event'].values and 'stop_playing' in behavior_df['Event'].values:
        on_times = behavior_df[behavior_df['Event'] == 'start_playing']['Timestamp'].reset_index(drop=True)
        off_times = behavior_df[behavior_df['Event'] == 'stop_playing']['Timestamp'].reset_index(drop=True)
        stimulus_df = behavior_df[behavior_df['Event'] == 'start_playing']
        
    elif 'stimulus_onset' in behavior_df['Event'].values and 'stimulus_offset' in behavior_df['Event'].values:
        on_times = behavior_df[behavior_df['Event'] == 'stimulus_onset']['Timestamp'].reset_index(drop=True)
        off_times = behavior_df[behavior_df['Event'] == 'stimulus_offset']['Timestamp'].reset_index(drop=True)
        stimulus_df = behavior_df[behavior_df['Event'] == 'stimulus_onset']
    # Calculating the duration for which the stimulus was on
    if len(on_times) == len(off_times):
        stimulus_durations = off_times - on_times
    else:
        print("Mismatch in 'on' and 'off' events count.")
        return None

    # return stimulus_df
    conditions = stimulus_df['Stimulus'].tolist()
    onsets = stimulus_df['Timestamp'].tolist()

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": stimulus_durations}
    )

    return events




def load_fusi_frames(power_doppler_path, probe_events, frame_indices=-1):
    """
    Fetches the data from specific frames in the h5 files based on the list of frame indices.
    
    Args:
    power_doppler_path (str): The directory where the h5 files are stored.
    probe_events (DataFrame): DataFrame containing the filenames and timestamps.
    frame_indices (list of int): The indices of the frames to fetch.
    
    Returns:
    np.array: The data from the specified frames in the h5 files, stacked along the last dimension.
    """
    # Check if all frame_indices are valid
    if frame_indices == -1:
        frame_indices = list(range(len(probe_events)))
    elif any(frame_index < 0 or frame_index >= len(probe_events) for frame_index in frame_indices):
        raise ValueError("One or more invalid frame indices")
    
    # Initialize a list to hold the data arrays
    data_list = []
    
    for frame_index in frame_indices:
        # Get the filename for the desired frame
        filename = probe_events.iloc[frame_index]['fusi_file_name']
        file_path = os.path.join(power_doppler_path, filename)

        # Load the h5 file
        with h5py.File(file_path, 'r') as file:
            # Assuming the dataset name in the h5 file is 'beamformed'
            try:
                data = file['beamformed'][:]
            except KeyError:
                data = file['power_doppler'][:]
            data_list.append(data)
    
    # Stack the data arrays along the last dimension
    stacked_data = np.stack(data_list, axis=-1)
    
    N=2
    frames_data_replicated = np.tile(stacked_data, (1, N, 1, 1))
    frames_data_replicated = np.flip(frames_data_replicated, axis=2) # get in the right coordinate system

    return frames_data_replicated

def match_and_add_fusi_filenames(probe_events, filtered_filenames):
        # Create a new column in probe_events to store the matched filenames from filtered_filenames
        probe_events['fusi_file_name'] = ''

        # Iterate through each row in probe_events
        for index, row in probe_events.iterrows():
            # Extract the filename from the current row in probe_events
            probe_filename = row['raw_file_name']
            # print(probe_filename[0:-3])
            # Check if this filename exists in the filtered_filenames list
            matched_filenames = [filename for filename in filtered_filenames if probe_filename[0:-3] in filename]

            # If there is a match, add the filename to the new column
            if matched_filenames:
                probe_events.at[index, 'fusi_file_name'] = matched_filenames[0]
        
        return probe_events

def process_power_doppler_files(power_doppler_path, num_tissue_components, probe_events):
    import os
    import re

    filenames = [f for f in os.listdir(power_doppler_path) if f.endswith('.h5') and f.startswith('ensemble')]
    if not filenames:
        raise ValueError("No power doppler files found in the specified directory.")

    # Extract unique values of num_tissue_components from filenames
    unique_tissue_components = set(re.findall(r'num_tissue_components=(\d+)', ' '.join(filenames)))

    # Check if the desired num_tissue_components is in the unique set of tissue components available in filenames
    if str(num_tissue_components) in unique_tissue_components:
        # Filter filenames to include only those with 'num_tissue_components=N' where N is the number of tissue components
        filtered_filenames = [f for f in filenames if f'num_tissue_components={num_tissue_components}' in f]
        print("Filtered filenames:", filtered_filenames)
    else:
        # Find the closest available num_tissue_components if the desired one is not available
        closest_num_tissue_components = min(unique_tissue_components, key=lambda x: abs(int(x) - num_tissue_components))
        filtered_filenames = [f for f in filenames if f'num_tissue_components={closest_num_tissue_components}' in f]
        print(f"Warning: num_tissue_components={num_tissue_components} is not available. Using closest available value: {closest_num_tissue_components}")
        print("Filtered filenames:", filtered_filenames)


    # Call the function to match and add filenames
    probe_events = match_and_add_fusi_filenames(probe_events, filtered_filenames)

    # Check if every entry in 'fusi_file_name' column is populated
    if probe_events['fusi_file_name'].isnull().any():
        raise ValueError("Some entries in 'fusi_file_name' are not populated.")
    else:
        print("All entries in 'fusi_file_name' are properly populated.")

    # Finally, load in the actual fusi data
    fusi_data = load_fusi_frames(power_doppler_path, probe_events)

    return fusi_data, probe_events



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


from numpy.fft import fft2, ifft2, fftshift
def phase_corr_translation(im1, im2):
    
    """ FFT phase correlation
    Kuglin, C. D. and Hines, D. C., 1975. The Phase Correlation Image Alignment Method. 
    Proceeding of IEEE International Conference on Cybernetics and Society, pp. 163-165, New York, NY, USA.

    returns
    -------
        ir normalized cross-correlation

    """
    shape = im2.shape
    f1 = fft2(im1)
    f2 = fft2(im2)
    ir = abs(ifft2((f1 * f2.conjugate()) / (abs(f1) * abs(f2))))
    x, y = np.unravel_index(np.argmax(ir), shape)
    if x > shape[0] // 2:
        x -= shape[0]
    if y > shape[1] // 2:
        y -= shape[1]
    return x, y, ir

def register_phase_corr(image_stack):
    # sitk_images = [sitk.GetImageFromArray(image_stack[:, :, i]) for i in range(image_stack.shape[2])]
    fixed_image = image_stack[..., 0]
    registered_images = [fixed_image]
    
    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_stack.shape[2]))  # Initialize with zeros
    metric_value = np.zeros((1, image_stack.shape[2]))  # Initialize with zeros

    transform_params[0, :] = 0  # If the last row is unused, set it to 1 or some default value
    
    
    
    # Apply registration
    for i in range(1, image_stack.shape[2]):
        moving_image = image_stack[..., i]
    # Ensure moving_image is correctly reshaped to match fixed_image dimensions
        # moving_image = moving_image[:fixed_image.shape[0], :fixed_image.shape[1]]
        
        tx,ty,metric_value_i=phase_corr_translation(fixed_image, moving_image)
    
        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        # transform_params[2, i] = angle

        metric_value[0,i] = metric_value_i

        resampled_image = sitk.Resample(moving_image, fixed_image, transform_params[:,i], sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        registered_images.append(resampled_image)
    
    registered_array = np.stack([sitk.GetArrayFromImage(img) for img in registered_images], axis=2)
    
    return registered_array, transform_params, metric_value



def register_image_stack(image_stack):
    sitk_images = [sitk.GetImageFromArray(image_stack[:, :, i]) for i in range(image_stack.shape[2])]
    fixed_image = sitk_images[0]
    registered_images = [fixed_image]
    
    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_stack.shape[2]))  # Initialize with zeros
    metric_value = np.zeros((1, image_stack.shape[2]))  # Initialize with zeros

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
        metric_value_i = registration_method.GetMetricValue()
        
        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        transform_params[2, i] = angle

        metric_value[0,i] = metric_value_i

        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        registered_images.append(resampled_image)
    
    registered_array = np.stack([sitk.GetArrayFromImage(img) for img in registered_images], axis=2)
    
    return registered_array, transform_params, metric_value


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




def extract_probe_events(h5_file_path):
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
        if event=='ensemble_saved':
            payload_data = json.loads(payload[i])

            # Extract the filename from the output_path
            file_name = os.path.basename(payload_data["ensemble_path"])


            data.append({
                'Event': events[i].decode('utf-8'),
                'raw_file_name': file_name,
                # 'raw_output_path': payload_data["output_path"],
                'acquisition_idx': payload_data["acquisition_idx"],
                'time_stamp': timestamps[i]
                
            })

    probe_data = pd.DataFrame(data)
    # Filter behavior_df for rows where the 'Event' column is 'frame_saved'

    return probe_data


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



