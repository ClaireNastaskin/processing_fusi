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

import SimpleITK as sitk


def register_3d_image_stack(image_stack):
    # Assuming `image_stack` has dimensions (x, y, z, time)
    num_frames = image_stack.shape[3]

    # Convert each 3D volume to a SimpleITK image
    sitk_images = [
        sitk.GetImageFromArray(image_stack[:, :, :, t]) for t in range(num_frames)
    ]
    fixed_image = sitk_images[0]
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