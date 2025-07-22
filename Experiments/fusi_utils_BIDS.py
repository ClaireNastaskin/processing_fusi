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
        img2 = shift(img, (x_shift_est, y_shift_est))
        img2 = np.expand_dims(img2, axis=0)
        if imgs_shifted is None:
            imgs_shifted = img2
        else:
            imgs_shifted = np.concatenate((imgs_shifted, img2), axis=0)
        ir = np.expand_dims(ir, axis=0)
        if ir_all is None:
            ir_all = ir
        else:
            ir_all = np.concatenate((ir_all, ir), axis=0)

    return imgs_shifted, ir_all


def pairwise_cross_correlation_variance(imgs):
    """
    Args:
        imgs: 3D numpy array (time x depth x lateral): stack of images to be aligned
    returns:
        ir_avg: 2D numpy array : average of the phase correlation between all pairs of images

    """
    ir_avg = np.zeros([imgs.shape[0], imgs.shape[0]])
    shape = imgs.shape
    fft_imgs = np.zeros(
        [imgs.shape[0], imgs.shape[1], imgs.shape[2]], dtype=np.complex_
    )
    for i in range(len(imgs)):
        fft_imgs[i] = fft2(imgs[i])

    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            ir = abs(
                ifft2(
                    (fft_imgs[i] * fft_imgs[j].conjugate())
                    / (abs(fft_imgs[i]) * abs(fft_imgs[j]))
                )
            )
            ir = np.expand_dims(ir, axis=0)
            ir_avg[i, j] = ir.var()
    return ir_avg


from numpy.fft import fft2, ifft2, fftshift


def phase_corr_translation(im1, im2):
    """FFT phase correlation
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


def register_phase_corr(image_seq):
    # sitk_images = [sitk.GetImageFromArray(image_seq[:, :, i]) for i in range(image_seq.shape[2])]
    fixed_image = image_seq[..., 0]
    registered_images = [fixed_image]

    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_seq.shape[2]))  # Initialize with zeros
    metric_value = np.zeros((1, image_seq.shape[2]))  # Initialize with zeros

    transform_params[0, :] = (
        0  # If the last row is unused, set it to 1 or some default value
    )

    # Apply registration
    for i in range(1, image_seq.shape[2]):
        moving_image = image_seq[..., i]
        # Ensure moving_image is correctly reshaped to match fixed_image dimensions
        # moving_image = moving_image[:fixed_image.shape[0], :fixed_image.shape[1]]

        tx, ty, metric_value_i = phase_corr_translation(fixed_image, moving_image)

        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        # transform_params[2, i] = angle

        metric_value[0, i] = metric_value_i

        resampled_image = sitk.Resample(
            moving_image,
            fixed_image,
            transform_params[:, i],
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
        registered_images.append(resampled_image)

    registered_array = np.stack(
        [sitk.GetArrayFromImage(img) for img in registered_images], axis=2
    )

    return registered_array, transform_params, metric_value


class DirectoryManager:
    def __init__(self, base_path, sequence):
        self.base_path = base_path
        self.sequence = sequence
        self.log_file_path = self.base_path / "logs" / "task_1.log"
        self.task_event_file_path = (
            self.base_path / "streams" / "task_1-event_stream.h5"
        )
        self.probe_event_file_path = (
            self.base_path / "streams" / "probe_1-event_stream.h5"
        )
        self.sequence_data_path = self.base_path / "acquisitions" / self.sequence
        self.raw_data_path = self.sequence_data_path / "raw_frame_data"
        self.beamformed_path = self.sequence_data_path / "beamformed"
        self.power_doppler_path = self.sequence_data_path / "power_doppler"
        self.meta_data_path = (
            self.power_doppler_path
        )  # This seems to be a mistake in the original code, corrected here
        self.fUSI_data_path = self.sequence_data_path / "fUSI"
        print(self.fUSI_data_path)
        # Ensure fUSI_data_path exists
        if not self.fUSI_data_path.exists():
            self.fUSI_data_path.mkdir(parents=True, exist_ok=True)

    def list_acquisition_directories(self):
        acquisitions_path = self.base_path / "acquisitions"
        directories = [d.name for d in acquisitions_path.iterdir() if d.is_dir()]
        print("Directories in acquisitions path:")
        for directory in directories:
            print(f"    {directory}")

    def print_paths(self):
        print("List of all directories in acquisitions path:")
        print(f"    log_file_path: {self.log_file_path}")
        print(f"    task_event_file_path: {self.task_event_file_path}")
        print(f"    probe_event_file_path: {self.probe_event_file_path}")
        print(f"    sequence_data_path: {self.sequence_data_path}")
        print(f"    raw_data_path: {self.raw_data_path}")
        print(f"    beamformed_path: {self.beamformed_path}")
        print(f"    power_doppler_path: {self.power_doppler_path}")
        print(f"    meta_data_path: {self.meta_data_path}")
        print(f"    USI_data_path: {self.fUSI_data_path}")


def extract_task_events(task_event_stream,probe_events):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(task_event_stream, "r") as file:
        # Extract data from the 'data' dataset
        dataset = file["data"][:]

        # Extract fields
        events = dataset["event"]
        timestamps = dataset["timestamp"]
        payload = dataset["payload"]

    # Convert data to DataFrame
    data = []
    for i in range(len(timestamps)):
        event_decoded = events[i].decode("utf-8")
        if event_decoded == "start_playing":
            stimulus_payload = json.loads(payload[i].decode("utf-8"))
            data.append(
                {
                    "event": "start",
                    "timestamp": timestamps[i],
                    "payload": stimulus_payload["stimulus"][:-4],
                }
            )

        elif event_decoded == "stop_playing":
            stimulus_payload = json.loads(payload[i].decode("utf-8"))
            data.append(
                {
                    "event": "stop",
                    "timestamp": timestamps[i],
                    "payload": stimulus_payload["stimulus"][:-4],
                }
            )

    behavior_df = pd.DataFrame(data)
    events = extract_nilearn_compatible_events(behavior_df,probe_events)

    return events


def extract_task_events_psychotria(task_event_stream,probe_events):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(task_event_stream, "r") as file:
        # Extract data from the 'data' dataset
        dataset = file["data"][:]

        # Extract fields
        events = dataset["event"]
        timestamps = dataset["timestamp"]
        payload = dataset["payload"]

    # Convert data to DataFrame
    data = []
    for i in range(len(timestamps)):
        event_decoded = events[i].decode("utf-8")
        if (
            event_decoded == "event"
            and json.loads(payload[i].decode("utf-8"))["start"] == True
        ):
            stimulus_payload = json.loads(payload[i].decode("utf-8"))
            data.append(
                {
                    "event": "start",
                    "timestamp": timestamps[i],
                    "payload": stimulus_payload["stimulus"][:-4],
                }
            )

        elif (
            event_decoded == "event"
            and json.loads(payload[i].decode("utf-8"))["stop"] == True
        ):
            stimulus_payload = json.loads(payload[i].decode("utf-8"))
            data.append(
                {
                    "event": "stop",
                    "timestamp": timestamps[i],
                    "payload": stimulus_payload["stimulus"][:-4],
                }
            )

    behavior_df = pd.DataFrame(data)
    events = extract_nilearn_compatible_events(behavior_df,probe_events)

    return events


def extract_nilearn_compatible_events(behavior_df,probe_events):
    """
    Analyzes a DataFrame containing behavioral event data to compute the durations of stimulus events.
    It extracts the times when the stimulus was turned on and off, calculates the duration for each stimulus event, and returns a DataFrame with the stimulus conditions, onset times, and durations.

    Args:
    behavior_df (DataFrame): A DataFrame with columns for timestamps, event descriptions, and relative experiment times.

    Returns:
    DataFrame: A DataFrame containing the trial type, onset times, and durations of each stimulus event.
    """

    if "start" in behavior_df["event"].values:
        on_times = behavior_df[behavior_df["event"] == "start"][
            "timestamp"
        ].reset_index(drop=True)
        off_times = behavior_df[behavior_df["event"] == "stop"][
            "timestamp"
        ].reset_index(drop=True)
        stimulus_df = behavior_df[behavior_df["event"] == "start"]

    # Calculating the duration for which the stimulus was on
    if len(on_times) == len(off_times):
        stimulus_durations = off_times - on_times
        conditions = stimulus_df["payload"].tolist()
        onsets = stimulus_df["timestamp"].tolist()
        events = pd.DataFrame({"trial_type": conditions, "onset": onsets, "duration": stimulus_durations})
    else:
        # print("Mismatch in 'on' and 'off' events count.")
        off_times.loc[len(off_times)] = probe_events['time_stamp'].iloc[-1]
        stimulus_durations = off_times - on_times
        if stimulus_durations.iloc[-1] < 0:
            on_times = on_times[:-1]
            off_times = off_times[:-1]
            stimulus_df = stimulus_df[:-1]
            stimulus_durations = off_times - on_times
        conditions = stimulus_df["payload"].tolist()
        onsets = stimulus_df["timestamp"].tolist()
        events = pd.DataFrame({"trial_type": conditions, "onset": onsets, "duration": stimulus_durations})

    return events


def load_fusi_frames(power_doppler_path, probe_events, frame_indices=-1):
    """
    Fetches the data from specific frames in the nifti file based on the list of frame indices.

    Args:
    power_doppler_path (str): Path to the nifti file (.nii.gz)
    probe_events (DataFrame): DataFrame containing the filenames and timestamps.
    frame_indices (list of int): The indices of the frames to fetch.

    Returns:
    np.array: The data from the specified frames in the nifti file.
    """
    probe_events = utils.extract_probe_events(probe_events)
    power_doppler_path = dm.power_doppler_path
    frame_indices = -1
    # Load the nifti file
    nifti_img = nib.load(power_doppler_path)
    data = nifti_img.get_fdata()

    # Check if all frame_indices are valid
    if frame_indices == -1:
        frame_indices = list(range(len(probe_events)))
    elif any(
        frame_index < 0 or frame_index >= data.shape[0] for frame_index in frame_indices
    ):
        raise ValueError("One or more invalid frame indices")

    # Extract the requested frames
    frames_data = data[frame_indices, :, :, :]
    frames_data = np.flip(frames_data, axis=2)  # get in the right coordinate system

    return frames_data


def match_and_add_fusi_filenames(probe_events, filtered_filenames):
    # Create a new column in probe_events to store the matched filenames from filtered_filenames
    probe_events["fusi_file_name"] = ""

    # Iterate through each row in probe_events
    for index, row in probe_events.iterrows():
        # Extract the filename from the current row in probe_events
        probe_filename = row["raw_file_name"]
        # print(probe_filename[0:-3])
        # Check if this filename exists in the filtered_filenames list
        matched_filenames = [
            filename
            for filename in filtered_filenames
            if probe_filename[0:-3] in filename
        ]

        # If there is a match, add the filename to the new column
        if matched_filenames:
            probe_events.at[index, "fusi_file_name"] = matched_filenames[0]

    return probe_events


def process_power_doppler_files(power_doppler_path, probe_events):
    import os
    import re

    filenames = [
        f
        for f in os.listdir(power_doppler_path)
        if f.endswith(".h5") and f.startswith("ensemble")
    ]
    if not filenames:
        raise ValueError("No power doppler files found in the specified directory.")

    # Extract unique values of num_tissue_components from filenames
    unique_tissue_components = set(
        re.findall(r"num_tissue_components=(\d+)", " ".join(filenames))
    )

    # Check if the desired num_tissue_components is in the unique set of tissue components available in filenames
    if str(num_tissue_components) in unique_tissue_components:
        # Filter filenames to include only those with 'num_tissue_components=N' where N is the number of tissue components
        filtered_filenames = [
            f
            for f in filenames
            if f"num_tissue_components={num_tissue_components}" in f
        ]
        print("Filtered filenames:", filtered_filenames)
    else:
        # Find the closest available num_tissue_components if the desired one is not available
        closest_num_tissue_components = min(
            unique_tissue_components, key=lambda x: abs(int(x) - num_tissue_components)
        )
        filtered_filenames = [
            f
            for f in filenames
            if f"num_tissue_components={closest_num_tissue_components}" in f
        ]
        print(
            f"Warning: num_tissue_components={num_tissue_components} is not available. Using closest available value: {closest_num_tissue_components}"
        )
        print("Filtered filenames:", filtered_filenames)

    # Call the function to match and add filenames
    probe_events = match_and_add_fusi_filenames(probe_events, filtered_filenames)

    # Check if every entry in 'fusi_file_name' column is populated
    if probe_events["fusi_file_name"].isnull().any():
        raise ValueError("Some entries in 'fusi_file_name' are not populated.")
    else:
        print("All entries in 'fusi_file_name' are properly populated.")

    # Finally, load in the actual fusi data
    fusi_data = load_fusi_frames(power_doppler_path, probe_events)

    return fusi_data, probe_events


import imageio
import os
from IPython.display import Video


def create_video_from_3d_data(data_3d, output_path="movie_test.mp4", fps=10):
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
    """FFT phase correlation
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


def register_phase_corr(image_seq):
    # sitk_images = [sitk.GetImageFromArray(image_seq[:, :, i]) for i in range(image_seq.shape[2])]
    fixed_image = image_seq[..., 0]
    registered_images = [fixed_image]

    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_seq.shape[2]))  # Initialize with zeros
    metric_value = np.zeros((1, image_seq.shape[2]))  # Initialize with zeros

    transform_params[0, :] = (
        0  # If the last row is unused, set it to 1 or some default value
    )

    # Apply registration
    for i in range(1, image_seq.shape[2]):
        moving_image = image_seq[..., i]
        # Ensure moving_image is correctly reshaped to match fixed_image dimensions
        # moving_image = moving_image[:fixed_image.shape[0], :fixed_image.shape[1]]

        tx, ty, metric_value_i = phase_corr_translation(fixed_image, moving_image)

        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        # transform_params[2, i] = angle

        metric_value[0, i] = metric_value_i

        resampled_image = sitk.Resample(
            moving_image,
            fixed_image,
            transform_params[:, i],
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
        registered_images.append(resampled_image)

    registered_array = np.stack(
        [sitk.GetArrayFromImage(img) for img in registered_images], axis=2
    )

    return registered_array, transform_params, metric_value


def register_3d_image_stack(image_stack):
    # Assuming `image_stack` has dimensions (x, y, z, time)
    num_frames = image_stack.shape[3]
    global_signal = np.mean(image_stack, axis=(0, 1, 2))
    # Find the frame corresponding to the median global signal
    median_frame_idx = np.argmin(np.abs(global_signal - np.median(global_signal)))
    print(f"Median frame index: {median_frame_idx}")
    # Convert each 3D volume to a SimpleITK image
    sitk_images = [
        sitk.GetImageFromArray(image_stack[:, :, :, t]) for t in range(num_frames)
    ]
    fixed_image = sitk_images[median_frame_idx]
    registered_images = [fixed_image]

    # Array to store transformation parameters (tx, ty, tz, and angles if applicable)
    transform_params = np.zeros((6, num_frames))  # 3 for translations in x, y, z

    # Setup registration method with appropriate metric and optimizer
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()  # Metric suitable for small motions
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1, minStep=1e-4, numberOfIterations=100
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Use a 3D Euler transform for 3D volumes
    initial_transform = sitk.Euler3DTransform()
    initial_transform.SetIdentity()
    registration_method.SetInitialTransform(initial_transform)

    # Apply registration for each 3D volume in the time sequence
    for i, moving_image in enumerate(sitk_images[1:], start=1):
        # Initialize a 3D transform with moments-based alignment
        initial_transform = sitk.Euler3DTransform(
            sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        )
        registration_method.SetInitialTransform(initial_transform)

        # Execute registration
        final_transform = registration_method.Execute(fixed_image, moving_image)
        tx, ty, tz = final_transform.GetTranslation()
        RotX = final_transform.GetAngleX()
        RotY = final_transform.GetAngleY()
        RotZ = final_transform.GetAngleZ()

        # Store transformation parameters for analysis
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        transform_params[2, i] = tz
        transform_params[3, i] = RotX
        transform_params[4, i] = RotY
        transform_params[5, i] = RotZ

        # Resample the moving image to align it with the fixed image
        resampled_image = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
        registered_images.append(resampled_image)

    # Convert the list of SimpleITK images back to a NumPy array
    registered_array = np.stack(
        [sitk.GetArrayFromImage(img) for img in registered_images], axis=-1
    )

    return registered_array, transform_params


def simpleITK_bline_registration(fusi2register):
    """
    Perform non-rigid registration of all volumes in a 4D fUSI dataset to a reference volume using SimpleITK.

    Parameters:
    fusi_data (numpy.ndarray): 4D array with shape (x, y, z, t).
    frame2register2 (int): Index of the reference volume along the time axis.

    Returns:
    numpy.ndarray: Registered 4D array with the same shape as `fusi_data`.
    list: List of transformation parameters for each timepoint.
    """

    x, y, z, t = fusi2register.shape
    reference_volume = fusi2register[..., 1]  # Reference volume

    # Convert reference volume to a SimpleITK image
    reference_image = sitk.GetImageFromArray(reference_volume.astype(np.float32))

    # Prepare for output
    registered_data = np.zeros_like(fusi2register)
    transformation_parameters = []

    # B-spline registration parameters
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsLBFGSB()
    registration_method.SetInterpolator(sitk.sitkLinear)

    # B-spline grid setup
    grid_physical_spacing = [
        50.0,
        50.0,
        50.0,
    ]  # Physical spacing (can adjust based on data)
    mesh_size = [
        int(sz / spc)
        for sz, spc in zip(reference_image.GetSize(), grid_physical_spacing)
    ]
    initial_transform = sitk.BSplineTransformInitializer(
        reference_image, mesh_size, order=3
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Set multi-resolution strategy
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])

    # Register each volume
    for i in range(t):
        moving_volume = fusi2register[..., i]
        moving_image = sitk.GetImageFromArray(moving_volume.astype(np.float32))

        # Perform registration
        final_transform = registration_method.Execute(reference_image, moving_image)

        # Resample the moving image to the reference frame
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetTransform(final_transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        registered_image = resampler.Execute(moving_image)

        # Store the registered volume
        registered_data[..., i] = sitk.GetArrayFromImage(registered_image)

        # Save transformation parameters
        transform_parameters = final_transform.GetParameters()
        transformation_parameters.append(transform_parameters)

        print(f"Volume {i + 1}/{t} registered.")

    return registered_data, transformation_parameters


def register_image_seq(image_seq):
    sitk_images = [
        sitk.GetImageFromArray(image_seq[:, :, i]) for i in range(image_seq.shape[2])
    ]
    fixed_image = sitk_images[0]
    registered_images = [fixed_image]

    # Array to store transformation parameters: [tx, ty, angle, 1]
    transform_params = np.zeros((3, image_seq.shape[2]))  # Initialize with zeros
    metric_value = np.zeros((1, image_seq.shape[2]))  # Initialize with zeros

    transform_params[0, :] = (
        0  # If the last row is unused, set it to 1 or some default value
    )

    # Setup registration method to be more constrained
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()  # Using correlation metric for small motions
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1, minStep=1e-4, numberOfIterations=100
    )
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
                fixed_image,
                moving_image,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        )
        registration_method.SetInitialTransform(initial_transform)

        final_transform = registration_method.Execute(fixed_image, moving_image)
        tx, ty = final_transform.GetTranslation()
        angle = final_transform.GetAngle()
        metric_value_i = registration_method.GetMetricValue()

        # Store the transformation parameters
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        transform_params[2, i] = angle

        metric_value[0, i] = metric_value_i

        resampled_image = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
        registered_images.append(resampled_image)

    registered_array = np.stack(
        [sitk.GetArrayFromImage(img) for img in registered_images], axis=2
    )

    return registered_array, transform_params, metric_value


def register_3d_image_seq(image_seq, reference_image):
    # Assuming `image_seq` has dimensions (time, x, y, z)
    # Assuming reference_image has dimensions (x, y, z)

    if image_seq[0].shape != reference_image.shape:
        raise ValueError(
            "The reference image and the image stack do not have the same dimensions."
        )

    num_frames = len(image_seq)
    # Convert each 3D volume to a SimpleITK image
    sitk_images = [
        # sitk.GetImageFromArray(image_seq[t]) for t in range(num_frames)
        sitk.GetImageFromArray(image)
        for image in image_seq
    ]
    fixed_image = reference_image
    registered_images = [fixed_image]

    # Array to store transformation parameters (tx, ty, tz, and angles if applicable)
    transform_params = np.zeros((6, num_frames))  # 3 for translations in x, y, z

    # Setup registration method with appropriate metric and optimizer
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()  # Metric suitable for small motions
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1, minStep=1e-4, numberOfIterations=100
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Use a 3D Euler transform for 3D volumes
    initial_transform = sitk.Euler3DTransform()
    initial_transform.SetIdentity()
    registration_method.SetInitialTransform(initial_transform)

    # Apply registration for each 3D volume in the time sequence
    for i, moving_image in enumerate(sitk_images[1:], start=1):
        # Initialize a 3D transform with moments-based alignment
        initial_transform = sitk.Euler3DTransform(
            sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        )
        registration_method.SetInitialTransform(initial_transform)

        # Execute registration
        final_transform = registration_method.Execute(fixed_image, moving_image)
        tx, ty, tz = final_transform.GetTranslation()
        RotX = final_transform.GetAngleX()
        RotY = final_transform.GetAngleY()
        RotZ = final_transform.GetAngleZ()

        # Store transformation parameters for analysis
        transform_params[0, i] = tx
        transform_params[1, i] = ty
        transform_params[2, i] = tz
        transform_params[3, i] = RotX
        transform_params[4, i] = RotY
        transform_params[5, i] = RotZ

        # Resample the moving image to align it with the fixed image
        resampled_image = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
        registered_images.append(resampled_image)

    # Convert the list of SimpleITK images back to a NumPy array
    registered_array = np.stack(
        [sitk.GetArrayFromImage(img) for img in registered_images], axis=-1
    )

    return registered_array, transform_params


import ants


import ants
import numpy as np
from scipy.spatial.transform import Rotation


def extract_transform_parameters(transform_file):
    """
    Extract translation and rotation parameters from an ANTs transformation matrix.

    Parameters:
    -----------
    transform_file : str
        Path to the ANTs transformation matrix file (.mat file)

    Returns:
    --------
    dict containing:
        - translation: [x, y, z] translation in mm
        - rotation: [x, y, z] rotation in degrees
        - matrix: full 4x4 transformation matrix
    """
    # Read the transformation matrix using ANTs
    transform = ants.read_transform(transform_file)

    # For rigid transforms, parameters are stored as:
    # [tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33]
    params = transform.parameters

    # Extract translation
    translation = params[:3]

    # Extract rotation matrix (3x3)
    rotation_matrix = params[3:].reshape(3, 3)

    # Convert rotation matrix to Euler angles (in degrees)
    r = Rotation.from_matrix(rotation_matrix)
    rotation = r.as_euler("xyz", degrees=True)

    # Construct full 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation

    return {"translation": translation, "rotation": rotation, "matrix": matrix}


"""
# Assuming you have a list of 3D images and a fixed reference image
fixed = ants.image_read('reference.nii.gz')
time_series = [ants.image_read(f'timepoint_{i}.nii.gz') for i in range(num_timepoints)]

# Perform registration with parameter extraction
result = register_time_series_with_params(
    fixed_image=fixed,
    time_series_images=time_series,
    type_of_transform="Rigid",  # Use Rigid for simpler parameter extraction
    outprefix="registration",
    verbose=True
)

# Access results
registered_images = result['registered_images']
transforms = result['transforms']
parameters = result['parameters']

# Print parameters for each time point
for i, params in enumerate(parameters):
    if params:
        print(f"\nTime point {i}:")
        print(f"Translation (mm): {params['translation']}")
        print(f"Rotation (degrees): {params['rotation']}")
"""


def register_ants(
    fixed_image, time_series_images, type_of_transform="Rigid", outprefix="", **kwargs
):
    """
    Register a series of 3D images and extract transformation parameters.

    Parameters:
    -----------
    fixed_image : ANTsImage
        The reference image to register all time points to
    time_series_images : list of ANTsImage
        List of 3D images representing different time points
    type_of_transform : str
        Type of registration transform to use (default: "Rigid")
    outprefix : str
        Prefix for output files
    **kwargs : dict
        Additional arguments to pass to ants.registration()

    Returns:
    --------
    dict containing:
        - registered_images: list of registered images
        - transforms: list of transform files
        - parameters: list of dictionaries containing translation and rotation for each time point
    """
    registered_images = []
    transforms = []
    parameters = []

    for i, moving_image in enumerate(time_series_images):
        # Create unique output prefix for this time point
        current_prefix = f"{outprefix}_timepoint_{i:03d}"

        # Perform registration
        result = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=type_of_transform,
            outprefix=current_prefix,
            **kwargs,
        )

        # Store the registered image and transform files
        registered_images.append(result["warpedmovout"])
        transforms.append(result["fwdtransforms"])

        # Extract transformation parameters
        # Note: for rigid/affine transforms, the first transform file will be the .mat file
        if isinstance(result["fwdtransforms"], list):
            mat_file = result["fwdtransforms"][0]  # Get the .mat file
            if mat_file.endswith(".mat"):
                params = extract_transform_parameters(mat_file)
                parameters.append(params)
            else:
                parameters.append(None)
        else:
            parameters.append(None)

    return {
        "registered_images": registered_images,
        "transforms": transforms,
        "parameters": parameters,
    }


def extract_parameters(transform):
    # This function will adapt based on the available methods in the ANTsTransform object
    transform_object = ants.read_transform(transform)
    tx, ty = transform_object.parameters[
        4:6
    ]  # Assuming these indices contain translations
    angle = transform_object.parameters[
        2
    ]  # Assuming this index contains the rotation angle
    return tx, ty, angle


def extract_probe_events(h5_file_path, seq_type):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file
    seq_type (str): type of sequence to filter for

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(h5_file_path, "r") as file:
        # Extract data from the 'data' dataset
        dataset = file["data"][:]

        # Extract fields
        events = dataset["event"]
        timestamps = dataset["timestamp"]
        payload = dataset["payload"]

    # Convert data to DataFrame
    data = []
    for i in range(len(timestamps)):
        event = events[i].decode("utf-8")
        payload_str = payload[i].decode("utf-8")
        payload_data = json.loads(payload_str)

        if event == "ensemble_saved" and seq_type in payload_data["sequence_id"]:
            # Extract the filename from the output_path
            file_name = os.path.basename(payload_data["ensemble_path"])

            data.append(
                {
                    "Event": event,
                    "raw_file_name": file_name,
                    "acquisition_idx": payload_data["acquisition_idx"],
                    "time_stamp": timestamps[i],
                }
            )

    probe_data = pd.DataFrame(data)
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

    on_times = behavior_df[behavior_df["Event"] == "pwm_enabled"][
        "Timestamp"
    ].reset_index(drop=True)
    off_times = behavior_df[behavior_df["Event"] == "pwm_disabled"][
        "Timestamp"
    ].reset_index(drop=True)

    # Calculating the duration for which the stimulus was on
    if len(on_times) == len(off_times):
        stimulus_durations = off_times - on_times
    else:
        print("Mismatch in 'on' and 'off' events count.")
        stimulus_durations = off_times - on_times[0:-1]
        # stimulus_durations = stimulus_durations.append(pd.Series([stimulus_durations.mean()]))
        stimulus_durations = pd.concat(
            [stimulus_durations, pd.Series([stimulus_durations.mean()])],
            ignore_index=True,
        )

        # return None

    # Extract conditions and onset times
    stimulus_df = behavior_df[behavior_df["Event"] == "pwm_enabled"]
    conditions = stimulus_df["Event"].tolist()
    onsets = stimulus_df["Timestamp"].tolist()

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": stimulus_durations}
    )

    return events

def boxcar_smooth(data, window_size):
    if window_size > 0:
        smoothed_data = np.copy(data)
        for t in range(data.shape[3]):
            start_index = max(0, t - window_size // 2)
            end_index = min(data.shape[3], t + window_size // 2 + 1)
            smoothed_data[:, :, :, t] = np.mean(
                data[:, :, :, start_index:end_index], axis=3
            )
    else:
        smoothed_data = data
    return smoothed_data


from nilearn.maskers import NiftiSpheresMasker
from nilearn.reporting import get_clusters_table
import numpy as np
from nilearn.glm.first_level import FirstLevelModel


def extract_time_course_from_main_clusters(contrast, threshold,sample_masks,fusi_img, fusi_glm, probe_events, t_r, radius):

    z_map = fusi_glm.compute_contrast(contrast)
    table = get_clusters_table(z_map, stat_threshold=threshold, cluster_threshold=10)
    table.set_index("Cluster ID", drop=True)
    print(table.head(2))

    # Select rows 0 and 1 (the first two rows) of the tableget the 2 largest clusters' max x, y, and z coordinates
    coords = table.loc[0:1, ["X", "Y", "Z"]].to_numpy()

    # extract time series from each coordinate
    masker = NiftiSpheresMasker(
        coords,
        radius=radius,
        detrend=True,
        standardize="zscore",
        t_r=t_r,
        allow_overlap=True,
    )

    real_timeseries = masker.fit_transform(fusi_img,sample_mask=sample_masks)
    predicted_timeseries = masker.fit_transform(fusi_glm.predicted[0])

    ###############################################################
    # Create pandas DataFrames for real and predicted timecourses #
    ###############################################################

    # Initialize empty arrays for timestamps
    real_timestamps = probe_events["time_stamp"].to_numpy()[sample_masks]
    pred_timestamps = probe_events["time_stamp"].to_numpy()[sample_masks]

    # Create DataFrames
    real_timecourse = pd.DataFrame({
        'signal_roi1': real_timeseries[:, 0],
        'signal_roi2': real_timeseries[:, 1],
        'time_stamps': real_timestamps
    })

    predicted_timecourse = pd.DataFrame({
        'signal_roi1': predicted_timeseries[:, 0], 
        'signal_roi2': predicted_timeseries[:, 1],
        'time_stamps': pred_timestamps
    })


    return [z_map, coords, real_timecourse, predicted_timecourse]





import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, show
import pandas as pd

def display_real_and_predicted_timecourses(contrast, real_timecourse, predicted_timecourse,task_events, coords, z_map, mean_image):
    colors = ["blue", "navy", "purple", "magenta", "olive", "teal"]
    task_colors = ['blue', 'purple', 'gray', 'lightgreen', 'red', 'orange', 'yellow', 'pink', 'cyan']

    # plot the time series and corresponding locations
    fig1, axs1 = plt.subplots(2, 2, figsize=(24, 14))
    
    # Add master title
    fig1.suptitle(contrast, fontsize=16)

    for i in range(2):
        # plotting time series
        axs1[0, i].set_title(f"Cluster peak {coords[i]}\n")
        axs1[0, i].plot(
            real_timecourse["time_stamps"],
            real_timecourse[f"signal_roi{i+1}"],
            c="r",
            ls="-",
            lw=2,
            label='Real timecourse in ROI'
        )
        axs1[0, i].plot(
            predicted_timecourse["time_stamps"],
            predicted_timecourse[f"signal_roi{i+1}"],
            c="black",
            ls="--",
            lw=1,
            label=f'Predicted {contrast} contribution to total signal'
        )
        axs1[0, i].set_xlabel("Time")
        axs1[0, i].set_ylabel("Signal intensity", labelpad=0)
        
        # Add colored rectangles for stimulus periods
        task_list_type = task_events.trial_type.unique()
        for task_index in range(len(task_list_type)):
            task = task_list_type[task_index]
            task_events_timing = pd.DataFrame()
            task_events_timing["Event"] = ["start", "stop"] * len(task_events[task_events["trial_type"] == task])
            task_events_timing["Timestamp"] = [
                onset
                for onset, duration in zip(
                    task_events[task_events["trial_type"] == task]["onset"],
                    task_events[task_events["trial_type"] == task]["duration"],
                )
                for _ in range(2)
            ]
            task_events_timing.loc[task_events_timing["Event"] == "stop", "Timestamp"] += (
                task_events[task_events["trial_type"] == task]["duration"].values.repeat(1)
            )
            task_events_timing = task_events_timing.sort_values("Timestamp").reset_index(drop=True)
            start_times = task_events_timing[task_events_timing["Event"] == "start"]["Timestamp"]
            stop_times = task_events_timing[task_events_timing["Event"] == "stop"]["Timestamp"]
            for start, stop in zip(start_times, stop_times):
                # Ensure task_index is within bounds of task_colors list
                color = task_colors[task_index % len(task_colors)]
                axs1[0, i].axvspan(start, stop, color=color, alpha=0.3, label=task if start == start_times.iloc[0] else "")
        
        # Add legend
        axs1[0, i].legend()
        
        # plotting image below the time series
        roi_img = plot_stat_map(
            z_map,
            cut_coords=[coords[i][2]],
            threshold=3.1,
            figure=fig1,
            axes=axs1[1, i],
            display_mode="y",
            colorbar=False,
            bg_img=mean_image,
            cmap='inferno'
        )
        roi_img.add_markers([coords[i]], colors[i], 300)
    fig1.set_size_inches(24, 14)

    show()

def display_predicted_timecourses(contrast, predicted_timecourse,task_events, coords, z_map, mean_image):
    colors = ["blue", "navy", "purple", "magenta", "olive", "teal"]
    task_colors = ['blue', 'purple', 'gray', 'lightgreen', 'red', 'orange', 'yellow', 'pink', 'cyan']

    # plot the time series and corresponding locations
    fig1, axs1 = plt.subplots(2, 2, figsize=(24, 14))
    
    # Add master title
    fig1.suptitle(contrast, fontsize=16)

    for i in range(2):
        # plotting time series
        axs1[0, i].set_title(f"Cluster peak {coords[i]}\n")
        axs1[0, i].plot(
            predicted_timecourse["time_stamps"],
            predicted_timecourse[f"signal_roi{i+1}"],
            c="black",
            ls="--",
            lw=1,
            label=f'Predicted {contrast} contribution to total signal'
        )
        axs1[0, i].set_xlabel("Time")
        axs1[0, i].set_ylabel("Signal intensity", labelpad=0)
        
        # Add colored rectangles for stimulus periods
        task_list_type = task_events.trial_type.unique()
        for task_index in range(len(task_list_type)):
            task = task_list_type[task_index]
            task_events_timing = pd.DataFrame()
            task_events_timing["Event"] = ["start", "stop"] * len(task_events[task_events["trial_type"] == task])
            task_events_timing["Timestamp"] = [
                onset
                for onset, duration in zip(
                    task_events[task_events["trial_type"] == task]["onset"],
                    task_events[task_events["trial_type"] == task]["duration"],
                )
                for _ in range(2)
            ]
            task_events_timing.loc[task_events_timing["Event"] == "stop", "Timestamp"] += (
                task_events[task_events["trial_type"] == task]["duration"].values.repeat(1)
            )
            task_events_timing = task_events_timing.sort_values("Timestamp").reset_index(drop=True)
            start_times = task_events_timing[task_events_timing["Event"] == "start"]["Timestamp"]
            stop_times = task_events_timing[task_events_timing["Event"] == "stop"]["Timestamp"]
            for start, stop in zip(start_times, stop_times):
                # Ensure task_index is within bounds of task_colors list
                color = task_colors[task_index % len(task_colors)]
                axs1[0, i].axvspan(start, stop, color=color, alpha=0.3, label=task if start == start_times.iloc[0] else "")
        
        # Add legend
        axs1[0, i].legend()
        
        # plotting image below the time series
        roi_img = plot_stat_map(
            z_map,
            cut_coords=[coords[i][2]],
            threshold=3.1,
            figure=fig1,
            axes=axs1[1, i],
            display_mode="y",
            colorbar=False,
            bg_img=mean_image,
            cmap='inferno'
        )
        roi_img.add_markers([coords[i]], colors[i], 300)
    fig1.set_size_inches(24, 14)

    show()

def display_regressor_temporal_component(
    contrast,
    fusi_glm,
    probe_events,
    t_r,
    sample_masks,
    task_events,
):
    z_map = fusi_glm.compute_contrast(contrast)
    table = get_clusters_table(z_map, stat_threshold=2, cluster_threshold=5)
    table.set_index("Cluster ID", drop=True)

    # Select the first row of the tableget the largest clusters' max x, y, and z coordinates
    coords = table.loc[[0], ["X","Y","Z"]].to_numpy()

    # extract time series from each coordinate
    masker = NiftiSpheresMasker(
        coords,
        radius=2,
        detrend=True,
        standardize="zscore",
        t_r=t_r,
        allow_overlap=True,
    )

    predicted_timeseries = masker.fit_transform(fusi_glm.predicted[0])
    pred_timestamps = probe_events["time_stamp"].to_numpy()[sample_masks]
    predicted_timecourse = pd.DataFrame({
        'signal': predicted_timeseries[:, 0], 
        'time_stamps': pred_timestamps
    })

    task_colors = ['blue', 'purple', 'gray', 'lightgreen', 'red', 'orange', 'yellow', 'pink', 'cyan']  

    # plot the time series and corresponding locations
    fig1, axs1 = plt.subplots(1, 1, figsize=(12, 5))

    # Add master title
    fig1.suptitle(contrast, fontsize=16)
    # plotting regressor time series
    axs1.plot(
        predicted_timecourse["time_stamps"],
        predicted_timecourse[f"signal"],
        c="black",
        ls="--",
        lw=1,
        label=f'Predicted {contrast} contribution to total signal'
    )
    axs1.set_xlabel("Time")
    axs1.set_ylabel("Signal intensity", labelpad=0)

    # Add colored rectangles for stimulus periods
    task_list_type = task_events.trial_type.unique()
    for task_index in range(len(task_list_type)):
        task = task_list_type[task_index]
        task_events_timing = pd.DataFrame()
        task_events_timing["Event"] = ["start", "stop"] * len(task_events[task_events["trial_type"] == task])
        task_events_timing["Timestamp"] = [
            onset
            for onset, duration in zip(
                task_events[task_events["trial_type"] == task]["onset"],
                task_events[task_events["trial_type"] == task]["duration"],
            )
            for _ in range(2)
        ]
        task_events_timing.loc[task_events_timing["Event"] == "stop", "Timestamp"] += (
            task_events[task_events["trial_type"] == task]["duration"].values.repeat(1)
        )
        task_events_timing = task_events_timing.sort_values("Timestamp").reset_index(drop=True)
        start_times = task_events_timing[task_events_timing["Event"] == "start"]["Timestamp"]
        stop_times = task_events_timing[task_events_timing["Event"] == "stop"]["Timestamp"]
        for start, stop in zip(start_times, stop_times):
            color = task_colors[task_index % len(task_colors)]
            axs1.axvspan(start, stop, color=color, alpha=0.3, label=task if start == start_times.iloc[0] else "")

    # Add legend
    axs1.legend()