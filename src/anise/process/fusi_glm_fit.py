import argparse
from pathlib import Path
import sys
import re
from dvclive import Live
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


def unpack_parameters(sequence: str):
    parts = sequence.split('_')
    # Initialize a dictionary to store the parameter values
    parameters = {}
    # Iterate through the parts and extract the parameter values
    for part in parts:
        # Use regular expressions to match the parameter name and value
        match = re.match(r'^(\w+)(\d+(?:p\d+)?)([a-zA-Z]*)$', part)
        if match:
            name, value, unit = match.groups()
            # Convert the value to the appropriate data type
            if 'p' in value:
                value = value.replace('p', '.')
            if name in ['tx_freq_hz', 'num_angles', 'tx_cycles', 'num_chip_repeats', 'num_imaging_loops']:
                value = int(value)
            elif name in ['tgc_initial_gain_db', 'tgc_final_gain_db', 'tgc_duration', 'Dmin', 'Dmax', 'el']:
                value = float(value)
            # Store the parameter name and value in the dictionary
            parameters[name] = value
    return parameters


def load_and_preprocess_data(dm: utils.DirectoryManager, event_time_offset: float, num_tissue_components: int, apply_image_registration: bool) -> tuple:
    """
    Load and preprocess fUSI data.

    Args:
        dm (DirectoryManager): Directory manager for the fUSI data.
        event_time_offset (float): Offset in seconds to apply to stimulus
        num_tissue_components (int): Number of tissue components.
        apply_image_registration (bool): Whether to apply image registration.

    Returns:
        tuple: A tuple containing the preprocessed fUSI data, events, and transformations (if applicable).
    """
    fusi_info = utils.load_fUSi_info(dm.power_doppler_path, num_tissue_components=num_tissue_components)
    task_info = utils.extract_events_from_h5(dm.task_event_file_path)
    events = utils.calculate_stimulus_events_caltech_daq(task_info)
    events['onset'] += event_time_offset

    frame_indices = -1
    fusi_data = utils.get_fusi_frames(dm.power_doppler_path, fusi_info, frame_indices)

    transformations = None
    if apply_image_registration:
        fusi_data_3d = fusi_data[:, 0, :, :]
        fusi_data_3d_reg, transformations = utils.register_image_stack(fusi_data_3d)
        fusi_data_3d_reg_expanded = np.expand_dims(fusi_data_3d_reg, axis=1)
        fusi_data = np.tile(fusi_data_3d_reg_expanded, (1, 2, 1, 1))

    return fusi_data, fusi_info, events, transformations


def create_design_matrix(fusi_info: dict, events: dict, transformations: np.ndarray, apply_image_registration: bool) -> np.ndarray:
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


def perform_glm_analysis(fusi_data: np.ndarray, design_matrix: np.ndarray, dm: utils.DirectoryManager, smoothing_fwhm: float) -> tuple:
    """
    Perform GLM analysis on fUSI data.

    Args:
        fusi_data (np.ndarray): Preprocessed fUSI data.
        design_matrix (np.ndarray): Design matrix for GLM analysis.
        dm (DirectoryManager): Directory manager for the fUSI data.
        smoothing_fwhm (float): Spatial smoothing FWHM in mm.

    Returns:
        tuple: A tuple containing the GLM results and the maximum t-stat value.
    """
    affine = np.diag([0.15, 0.15, 0.15, 0])
    nifti_img = nib.Nifti1Image(fusi_data, affine)
    nifti_img.to_filename(dm.fUSI_data_path / "fUSi_data.nii.gz")
    nifti_data = image.load_img(dm.fUSI_data_path / "fUSi_data.nii.gz")

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }

    fmri_glm = FirstLevelModel(minimize_memory=False, mask_img=False, smoothing_fwhm=smoothing_fwhm, standardize=True)
    fmri_glm = fmri_glm.fit(nifti_data, design_matrices=design_matrix)

    z_map = fmri_glm.compute_contrast(basic_contrasts['pwm_enabled'], output_type="stat")
    max_tstat = np.max(z_map.get_fdata())

    print(f"DVCLive directory: {dm.dvclive_dir}")
    # Save the z_map as a NIfTI file
    z_map_filename = dm.fUSI_data_path / "z_map.nii.gz"
    z_map.to_filename(z_map_filename)
    # Log the maximum t-stat value using DVCLive
    with Live(dm.dvclive_dir) as live:
        live.log_metric("max_tstat", max_tstat)
        live.log_artifact(str(z_map_filename))  # Track the z_map file with DVC
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