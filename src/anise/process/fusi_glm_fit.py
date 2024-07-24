import argparse
from pathlib import Path
import sys
import re
# Add the Anise directory to the Python module search path
anise_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(anise_dir))
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import mean_img
from Experiments import ucla_fusi_utils as utils

from nilearn.plotting import plot_design_matrix
from nilearn.image import concat_imgs, mean_img, resample_img
from nilearn.maskers import NiftiSpheresMasker
from nilearn.reporting import get_clusters_table
from sklearn.metrics import r2_score

# def unpack_parameters(sequence: str):
#     parts = sequence.split('_')
#     # Initialize a dictionary to store the parameter values
#     parameters = {}
#     # Iterate through the parts and extract the parameter values
#     for part in parts:
#         # Use regular expressions to match the parameter name and value
#         match = re.match(r'^(\w+)(\d+(?:p\d+)?)([a-zA-Z]*)$', part)
#         if match:
#             name, value, unit = match.groups()
#             # Convert the value to the appropriate data type
#             if 'p' in value:
#                 value = value.replace('p', '.')
#             if name in ['tx_freq_hz', 'num_angles', 'tx_cycles', 'num_chip_repeats', 'num_imaging_loops']:
#                 value = int(value)
#             elif name in ['tgc_initial_gain_db', 'tgc_final_gain_db', 'tgc_duration', 'Dmin', 'Dmax', 'el']:
#                 value = float(value)
#             # Store the parameter name and value in the dictionary
#             parameters[name] = value
#     return parameters


# def load_and_preprocess_data(dm: utils.DirectoryManager, event_time_offset: float, num_tissue_components: int, apply_image_registration: bool) -> tuple:
#     """
#     Load and preprocess fUSI data.

#     Args:
#         dm (DirectoryManager): Directory manager for the fUSI data.
#         event_time_offset (float): Offset in seconds to apply to stimulus
#         num_tissue_components (int): Number of tissue components.
#         apply_image_registration (bool): Whether to apply image registration.

#     Returns:
#         tuple: A tuple containing the preprocessed fUSI data, events, and transformations (if applicable).
#     """
#     fusi_info = utils.load_fUSi_info(dm.power_doppler_path, num_tissue_components=num_tissue_components)
#     task_info = utils.extract_events_from_h5(dm.task_event_file_path)
#     events = utils.calculate_stimulus_events_caltech_daq(task_info)
#     events['onset'] += event_time_offset

#     frame_indices = -1
#     fusi_data = utils.get_fusi_frames(dm.power_doppler_path, fusi_info, frame_indices)

#     transformations = None
#     if apply_image_registration:
#         fusi_data_3d = fusi_data[:, 0, :, :]
#         fusi_data_3d_reg, transformations = utils.register_image_stack(fusi_data_3d)
#         fusi_data_3d_reg_expanded = np.expand_dims(fusi_data_3d_reg, axis=1)
#         fusi_data = np.tile(fusi_data_3d_reg_expanded, (1, 2, 1, 1))

#     return fusi_data, fusi_info, events, transformations


def create_design_matrix(fusi_info: dict, events: dict, transformations: np.ndarray, 
apply_image_registration: bool) -> np.ndarray:
    """
    Create the design matrix for GLM analysis.

    Args:
        fusi_info (dict): Dictionary containing fUSI information.
        events (dict): Dictionary containing event information.
        transformations (np.ndarray): Array containing image transformations.
        apply_image_registration (bool): Whether image registration was applied.

    Returns:
        np.ndarray: The design matrix.
    """
    hrf_model = "glover"
    if apply_image_registration:
        design_matrix = make_first_level_design_matrix(
            fusi_info['Experiment Time'],
            events,
            drift_model="polynomial",
            drift_order=1,
            add_regs=transformations.transpose(),
            add_reg_names=["tx", "ty", "rot"],
            hrf_model=hrf_model,
        )
    else:
        design_matrix = make_first_level_design_matrix(
            fusi_info['Experiment Time'],
            events,
            drift_model="polynomial",
            drift_order=1,
            hrf_model=hrf_model,
        )
    return design_matrix


def perform_glm_analysis(nifti_data, design_matrix: np.ndarray, output_path: Path, smoothing_fwhm: float) -> tuple:
    """
    Perform GLM analysis on fUSI data.

    Args:
        nifti_data (nifti): Processed nifti data.
        design_matrix (np.ndarray): Design matrix for GLM analysis.
        output_path(Oath): Directory manager for the fUSI data.
        smoothing_fwhm (float): Spatial smoothing FWHM in mm.

    Returns:
        tuple: A tuple containing the GLM results and the maximum t-stat value.
    """
    

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }

    fmri_glm = FirstLevelModel(minimize_memory=False, 
                                mask_img=False, 
                                smoothing_fwhm=smoothing_fwhm, 
                                standardize=True,
                                )
    fmri_glm = fmri_glm.fit(nifti_data, design_matrices=design_matrix)

    z_map = fmri_glm.compute_contrast(basic_contrasts['pwm_enabled'], output_type="stat")
    max_tstat = np.max(z_map.get_fdata())

    # Save the z_map as a NIfTI file
    z_map_filename = output_path / "z_map.nii.gz"
    z_map.to_filename(z_map_filename)
    return fmri_glm, z_map, max_tstat


def plot_results(fmri_glm, z_map: np.ndarray, dm: utils.DirectoryManager, sequence: str):
    """
    Plot the GLM analysis results.

    Args:
        fmri_glm: The fitted GLM model.
        z_map (np.ndarray): The computed z-map.
        dm (DirectoryManager): Directory manager for the fUSI data.
        sequence (str): The sequence name.
    """
    mean_image = mean_img(fmri_glm.masker_.mask_img_)
    plotting.plot_stat_map(
        z_map,
        bg_img=mean_image,
        threshold=3,
        display_mode="y",
        black_bg=True,
        title="pwm_enabled contrast",
    )
    plt.savefig(dm.fUSI_data_path / f"tstat_map_{sequence}.png")
    plt.close()


def get_ROI_activation(nifti_data, fmri_glm, design_matrix, basic_contrasts, time_stamp, events, param_glm, output_file=None):
    # Find ROIs by identifying the locations that show a significant response to the stimulus
    mean_image = mean_img(nifti_data)

    # Loop over all conditions in basic_contrasts
    for contrast_id, contrast_val in basic_contrasts.items():
        if contrast_id not in ['drift_1', 'constant','tx','ty','rot']:
            print(f"Processing contrast: {contrast_id}")
            z_map = fmri_glm.compute_contrast(contrast_val, output_type="stat")
            table = get_clusters_table(z_map, 
                                       stat_threshold=param_glm["stat_threshold"], 
                                       cluster_threshold=20)
            table.set_index("Cluster ID", drop=True)

            if not table.empty and len(table) >= param_glm["num_locations"]:
                # get the num_locations largest clusters' max x, y, and z coordinates
                coords = table.loc[range(0, param_glm["num_locations"]), ["X", "Y", "Z"]].values
                # coords = np.vstack([coords, [10.45000061, 0., 18.40000045]])

                # Now use the ROIs to extract the time series 
                masker = NiftiSpheresMasker(
                    coords,
                    radius=0.1,  # specifies the size of the spherical region around each given coordinate from which the signal will be extracted
                    detrend=True,
                    standardize=param_glm["standardize"],  # zscore psc
                    t_r= param_glm["t_r"],
                    memory_level=1,
                    verbose=0,
                )
                real_timeseries = masker.fit_transform(nifti_data)
                predicted_timeseries = np.tile(design_matrix[contrast_id].values, (real_timeseries.shape[1], 1)).T

                # Calculate the R-squared value for each ROI
                r_squared_values = []
                for i in range(real_timeseries.shape[1]):
                    r_squared = r2_score(real_timeseries[:, i], predicted_timeseries[:, i])
                    r_squared_values.append(r_squared)
                
                # Plotting the time series for each ROI and the corresponding location of each ROI
                colors = ["blue", "purple", "magenta", "olive", "teal"]
                N = real_timeseries.shape[-1]
                fig, axs = plt.subplots(2, N)
                # Set the title of the figure with a bigger font size
                fig.suptitle(f"Stimulus: {contrast_id}", fontsize=20)
                # get plot limits, common across all axis
                mx = real_timeseries.max() + 0.05 * real_timeseries.max()
                mn = real_timeseries.min() + 0.05 * real_timeseries.min()

                for i in range(N):
                    # plotting time series
                    axs[0, i].set_title(f"Cluster peak {coords[i].round(1)}, R2: {r_squared_values[i]:.2f}\n")
                    if masker.standardize != "psc":
                        axs[0, i].plot(time_stamp, predicted_timeseries[:, i], c="k", ls="--", lw=1)

                    axs[0, i].plot(time_stamp, real_timeseries[:, i], c=colors[i], lw=2)

                    axs[0, i].set_xlabel("Time (S)")

                    if masker.standardize == "psc":
                        axs[0, i].set_ylabel('Percent Signal Change', labelpad=0)
                    else:
                        axs[0, i].set_ylabel("Signal intensity", labelpad=0)

                    # plotting image below the time series
                    for index, row in events.iterrows():
                        if row['trial_type'] == contrast_id and row['onset'] > time_stamp[0]:
                            axs[0, i].axvspan(row['onset'] + param_glm["behavior_offset"],
                            row['onset'] + row['duration'] + param_glm["behavior_offset"],
                            color='gray', alpha=0.5)

                    roi_img = plotting.plot_stat_map(
                        z_map,
                        cut_coords=coords[i],
                        threshold=4,
                        figure=fig,
                        axes=axs[1, i],
                        display_mode="y",
                        colorbar=False,
                        bg_img=mean_image,
                    )

                    axs[0, i].set_ylim(mn, mx)
                    roi_img.add_markers([coords[i]], colors[i], 200)
                fig.set_size_inches(24, 14)
                
                if output_file:
                    fig.savefig(output_file,dpi=400, bbox_inches='tight')
            else:
                print(f"No valid clusters found for {contrast_id} that meet the criteria.")

def main(base_path: Path, sequence: str, event_time_offset: float, apply_image_registration: bool, smoothing_fwhm: float, num_tissue_components: int, plot_figures: bool = False):
    """
    Main function to perform GLM analysis on fUSI data.

    Args:
        base_path (Path): Path to the base directory.
        sequence (str): Sequence name.
        event_time_offset (float): Offset in seconds to apply to stimulus onsets.
        apply_image_registration (bool): Whether to apply image registration.
        smoothing_fwhm (float): Spatial smoothing FWHM in mm.
        num_tissue_components (int): Number of tissue components.
        plot_figures (bool, optional): Whether to plot the analysis results. Defaults to False.
    """
    dm = utils.DirectoryManager(base_path, sequence)
    parameters = unpack_parameters(sequence)
    dm.dvclive_dir = dm.sequence_data_path / "metrics"
    dm.dvclive_dir.mkdir(parents=True, exist_ok=True)
    with Live(dm.dvclive_dir) as live:
        # Log the unpacked parameter values to DVC Live using the dictionary
        for name, value in parameters.items():
            live.log_param(name, value)
    fusi_data, fusi_info, events, transformations = load_and_preprocess_data(dm, event_time_offset, num_tissue_components, apply_image_registration)
    design_matrix = create_design_matrix(fusi_info, events, transformations, apply_image_registration)
    fmri_glm, z_map, max_tstat = perform_glm_analysis(fusi_data, design_matrix, dm, smoothing_fwhm)
    print(f"Maximum t-stat value: {max_tstat}")
    if plot_figures:
        plot_results(fmri_glm, z_map, dm, sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform GLM analysis on fUSI data.")
    parser.add_argument("base_path", type=Path, help="Path to the base directory.")
    parser.add_argument("sequence", type=str, help="Sequence name.")
    parser.add_argument("--event-time-offset", type=float, default=-3, help="Offset in seconds to apply to stimulus onsets.")
    parser.add_argument("--no-image-registration", action="store_false", dest="apply_image_registration", help="Disable image registration.")
    parser.add_argument("--smoothing-fwhm", type=float, default=0.5, help="Spatial smoothing FWHM in mm.")
    parser.add_argument("--num-tissue-components", type=int, default=40, help="Number of tissue components.")
    parser.add_argument("--plot", action="store_true", help="Enable plotting figures.")
    args = parser.parse_args()

    main(args.base_path, args.sequence, args.event_time_offset, args.apply_image_registration, args.smoothing_fwhm, args.num_tissue_components, plot_figures=args.plot)