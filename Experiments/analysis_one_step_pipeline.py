import os
import re
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from IPython.display import HTML, Video
from anise.gui import *
import anise.motion_utils as motion_utils
import anise.utils as utils

import nibabel as nib

import neptune
from neptune.types import File
from neptune.utils import stringify_unsupported
from dotenv import load_dotenv

load_dotenv()
NEPTUNE_API_TOKEN = os.environ["NEPTUNE_API_TOKEN"]

def set_params(n_images):
    # clustering parameters
    param_c = dict(
        outlier_threshold = 500, # pwd intensity
        dim_red_method = 'VGG', # 'PCA' or 'UMAP' or 'tSNE' or 'VGG'
        pca_n_components = min(100, n_images - 1), # not applied for VGG
        perplexity = min(200, n_images - 1), # specific for tSNE
        DBSCAN_eps = 1, # smaller eps, more clusters
        )

    # registration parameters
    param_reg = dict(
        auto_optimize_ref_frame = True,
        metric_name = 'mutual_information',        
        apply_image_registration = True,
        apply_temporal_smoothing = False,
        temporal_smoothing_window_size = 4, # Number of time points for smoothing
        crop_border = [0, 0, 0, 0], # crop the image to remove border
        )

    # GLM parameters  
    param_glm =  dict(
        apply_transform_in_design_mtx = True and param_reg["apply_image_registration"],
        smoothing_fwhm = .5, #0 spatial smoothing in mm to apply to the data before computing GLM
        behavior_offset = 0, # in seconds
        stat_threshold = 1,
        num_ROI_locations = 2, # How many ROIs to visualize. An additional random ROI will be added
        hrf_model = "glover",
        standardize = "psc",
        t_r = 3.5,
        )
    return param_c, param_reg, param_glm

def run_in_plane_clustering(fusi_data, param_c, reg_plot_dir, log_neptune):

    # find and confirm if there's any outlier frames that should be removed e.g. high brightness or noise
    print('\n[In-plane] Perfrom in plane images detection...')

    #############################################################################
    #                   Display Doppler intensity time course                   #
    #############################################################################

    doppler_intensity_per_frame, outlier_intensity_frames, fig_pwdi, ax = \
        motion_utils.plot_pd_intensity_over_time(fusi_data,
                                                outlier_threshold=param_c["outlier_threshold"])
    plt.show(block=False)
    fig_pwdi.savefig(reg_plot_dir / 'doppler_intensity_over_time.png', dpi=300, bbox_inches='tight')
    plt.close(fig_pwdi)

    #############################################################################
    #             If needed: identify any out-of-plane images                   #
    #############################################################################

    # dimensionality reduction and clustering
    embedding = motion_utils.dim_red(fusi_data, param_c)
    labels, n_clusters, n_noise, eps = motion_utils.dbscan_clustering(embedding, 
                                                                    starting_eps=param_c["DBSCAN_eps"])
    param_c["DBSCAN_eps"] = eps

    # ## Relabel cluster number to match with our label convention
    pred_labels = motion_utils.remap_cluster_to_labels(labels)
    pred_labels_oop_ind = np.where(pred_labels < 0)[0]
    outliers_intensity_ind = np.where(outlier_intensity_frames)[0]
    # combine with outlier_intensity_frames
    pred_labels[outlier_intensity_frames] = -1
    in_plane_indices = np.where(np.array(pred_labels) == 0)[0]

    # logging the results
    print(f'[In-plane] Summary of out-of-plane detection:')
    print(f'\tOutlier indices from feature extraction + clustering:   {pred_labels_oop_ind}')
    print(f'\tOutlier indices from measuring power doppler intensity: {outliers_intensity_ind}')
    # calculate overlap between outlier_intensity_frames and pred_labels
    if not any(pred_labels_oop_ind) and not any(outliers_intensity_ind):
        overlap = len(set(pred_labels_oop_ind).intersection(outlier_intensity_frames))
        print('\tPercentage of overlap between the two types of outlier:', \
            overlap/len(outlier_intensity_frames)*100 if len(outlier_intensity_frames) > 0 else \
            overlap/len(pred_labels_oop_ind)*100)
    # print predicted labels 20 in a row
    print('\n[In-plane] Remapped predicted labels:')
    for row in range(len(pred_labels) // 20 + 1):
        print('\t', row*20, '\t', pred_labels[row * 20:(row + 1) * 20])
    prct_in_plane = len(in_plane_indices)/len(pred_labels)*100
    print(f'\n[In-plane] Percentage of in-plane frames: {prct_in_plane:.2f}%\n')

    # show the outlier images, if any
    if np.any(pred_labels != 0):
        clim_max = np.percentile(fusi_data, 99)
        fig_oop, axs = motion_utils.show_imgs(fusi_data, np.where(pred_labels != 0)[0], clim=[0, clim_max])
        fig_oop.savefig(reg_plot_dir / 'out_of_plane_images.png', dpi=300, bbox_inches='tight')
        plt.close(fig_oop)
    
    # Find centroid the clustering results
    centroids, closest_points_in_plane = motion_utils.sort_in_plane_points_closest_to_centroid(
                                            embedding, pred_labels, n_clusters)
    
    if log_neptune is not None:
        log_neptune["parameters/clustering"] = stringify_unsupported(param_c)
        log_neptune["outputs/percent_in_plane"].append(prct_in_plane)
        log_neptune["outputs/n_clusters"].append(n_clusters)

    return embedding, pred_labels, centroids, closest_points_in_plane

def run_registration(fusi_data, metadata, pred_labels, closest_points_in_plane,
                    param_reg, reg_plot_dir, filename, log_neptune):
    
    #############################################################################
    #                   If needed: perform image registration                   #
    #############################################################################
    register_dir = reg_plot_dir.parent.parent
    movie_dir = register_dir / 'movies'
    fus_dir = register_dir / 'fus'
    movie_dir.mkdir(parents=True, exist_ok=True)
    fus_dir.mkdir(parents=True, exist_ok=True)

    in_plane_indices = np.where(np.array(pred_labels) == 0)[0]

    # Perform image registration on in-plane frame indices
    prct_improve = 0.0
    registered_images = None
    final_ref_frame = None
    if param_reg["apply_image_registration"]:
        print('[Registration] Applying image registration...\n')
        registered_images, extra, final_ref_frame = motion_utils.auto_ants_registration_on_in_plane(
            fusi_data, in_plane_indices, closest_points_in_plane, 
            optimize_ref=param_reg["auto_optimize_ref_frame"])
        # plot registration transformation outputs
        fig_p, axs, prct_improve = motion_utils.plot_transform_params(extra, oop_labels=pred_labels, 
                                        metric_name=param_reg["metric_name"], 
                                        annotate_ref_frame=final_ref_frame)
        fig_p.savefig(reg_plot_dir / 'transform_params.png', dpi=400, bbox_inches='tight')
        plt.close(fig_p)
    else:
        print('[Registration] Skipping image registration...\n')
        extra = pd.DataFrame({'in_plane_indices': in_plane_indices})

    #############################################################################
    #                                Saving outputs                             #
    #############################################################################

    # Save outputs as csv
    tsv_filename = register_dir / "fus" / (filename[:-4]+'outputs.tsv')
    extra.to_csv(tsv_filename, index=False, sep='\t')

    # save the registered images to a gif or a mp4
    ani = None
    if param_reg["apply_image_registration"]:
        ani = MakeAnimation(registered_images, 
                            output_file=str(register_dir / 'movies' / f'{filename}_after.gif'), 
                            fps=10, 
                            depth=metadata['Depth'], 
                            lateral=metadata['Lateral'], 
                            time=metadata['VolumeTiming'],
                            )
        ani = MakeAnimation(registered_images, 
                            output_file=str(register_dir / 'movies' / f'{filename}_after.mp4'), 
                            fps=10, 
                            depth=metadata['Depth'], 
                            lateral=metadata['Lateral'], 
                            time=metadata['VolumeTiming'],
                            )
    if log_neptune is not None and ani is not None:
        log_neptune["outputs/movies/before"].upload(str(register_dir / 'movies' / f'{filename}_before.mp4'))
        log_neptune["outputs/movies/after"].upload(str(register_dir / 'movies' / f'{filename}_after.mp4'))

    # crop the border of the registered images
    if any(param_reg["crop_border"]) and registered_images is not None:
        registered_images = motion_utils.crop_images(registered_images, crop_border=param_reg["crop_border"])
        print(f'Cropped the border of the registered images by {param_reg["crop_border"]}')

    N=2 # duplicate the data along the elevation dimension
    # save the registered images to nifti
    if param_reg["apply_image_registration"]:
        reg_pwd = np.tile(np.expand_dims(registered_images, 1), (1, N, 1, 1))
        if param_reg["apply_temporal_smoothing"]:
            pwd_smooth = utils.boxcar_smooth(reg_pwd, param_reg["temporal_smoothing_window_size"])
            filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", pwd_smooth, 
                                                filename=filename, 
                                                filename_tag='register_smooth')
        else:
            filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", reg_pwd, 
                                                filename=filename, 
                                                filename_tag='register')
    else:
        data_selected = np.tile(np.expand_dims(fusi_data[:, :, in_plane_indices], 1), (1, N, 1, 1))
        if param_reg["apply_temporal_smoothing"]:
            pwd_smooth = utils.boxcar_smooth(data_selected, param_reg["temporal_smoothing_window_size"])
            filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", pwd_smooth,
                                                filename=filename, 
                                                filename_tag='in_plane_no_reg_smooth')
        else:
            filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", data_selected,
                                                filename=filename, 
                                                filename_tag='in_plane_no_reg')
    nifti_full_path = register_dir / "fus" / f'{filename_new}.nii.gz'

    # load in nifti format
    nifti_data = nib.load(nifti_full_path)
    print(f'Loaded NIFTI data from: {nifti_full_path}')
    print(f'New NIFTI data shape:   {nifti_data.shape}\n')

    # log the results to neptune
    if log_neptune is not None:
        try:
            log_neptune["parameters/registration"] = stringify_unsupported(param_reg)
            log_neptune["outputs/percent_improvement"].append(prct_improve)
            log_neptune["outputs/transforms"].upload(str(tsv_filename))
            for plot in reg_plot_dir.iterdir():
                fname = plot.name
                log_neptune["outputs/plots/registration/" + fname].upload(str(plot))
        except Exception as e:
            print(f'Error in logging registration outputs: {e}')

    return nifti_data, extra, final_ref_frame

def plot_embedding_clusters(embedding, pred_labels, centroids, final_ref_frame, reg_plot_dir, log_neptune):
    # label the nearest point to the in-plane centroid in the embedding space
    if final_ref_frame is not None:
        title_str = f'Cluster centroid and chosen reference image (#{final_ref_frame})'
        in_plane_indices = np.where(np.array(pred_labels) == 0)[0]
        loc = [in_plane_indices[final_ref_frame]]
    else:
        title_str = 'Clusters and centroids'
        loc = []
    output_filename = reg_plot_dir / 'embedding_clusters.png'
    fig, ax = motion_utils.plot_embedding(embedding, pred_labels, 
                                          centroids=centroids, 
                                          highlight_points=loc,
                                          title=title_str, 
                                          output_file=output_filename)
    plt.close(fig)
    if log_neptune is not None:
        log_neptune["outputs/plots/registration/embedding_clusters.png"].upload(str(output_filename))

def run_glm(nifti_data, task_events, extra, metadata, param_glm, glm_zmap_dir, glm_plot_dir, log_neptune):
    #############################################################################
    #                    Perform first level GLM analsysis                      #
    #############################################################################

    from nilearn import image, plotting
    from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
    from nilearn.plotting import plot_design_matrix
    from nilearn.image import concat_imgs, mean_img, resample_img
    from nilearn.maskers import NiftiSpheresMasker
    from nilearn.reporting import get_clusters_table
    from sklearn.metrics import r2_score
    from anise.process.fusi_glm_fit import get_ROI_activation, create_design_matrix

    #######################################
    #     Create design matrix for GLM    #
    #######################################
    print('[GLM] Perform first level GLM analsysis...')

    # Sub-period of time points that are in-plane
    in_plane_indices = extra['in_plane_indices'].values
    time_stamp = np.array(metadata['VolumeTiming'])[in_plane_indices]
    if "time_stamp" in task_events.columns:
        task_events = task_events.drop("time_stamp", axis=1)

    # create design matrix
    transformations = extra[["translation_x", "translation_y", "rotation"]].to_numpy()
    design_matrix, design_matrix_name = create_design_matrix(time_stamp, task_events, \
        transformations, param_glm["apply_transform_in_design_mtx"], hrf_model=param_glm["hrf_model"])
    plot_design_matrix(design_matrix, output_file=glm_plot_dir / f'{design_matrix_name}.png')
    
    # This cell creates an identity matrix of size equal to the number of columns in the design_matrix.
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }
    print('[GLM] Basic contrasts:')
    for key in basic_contrasts.keys():
        print(f'\t{key:15}: {basic_contrasts[key]}')

    # Initialize and fit the GLM model with specified parameters
    print("[GLM] Fitting a GLM")
    fmri_glm = FirstLevelModel(minimize_memory=False,
                                mask_img=False,
                                smoothing_fwhm=param_glm["smoothing_fwhm"],
                                standardize=True)
    fmri_glm = fmri_glm.fit(nifti_data, design_matrices=design_matrix)

    print("[GLM] Computing contrasts")
    mean_image = mean_img(nifti_data)

    # Iterate on contrasts
    for contrast_id, contrast_val in basic_contrasts.items():
        print(f"\tcontrast id: {contrast_id}")
        # compute the contrasts
        z_map = fmri_glm.compute_contrast(contrast_val, output_type="stat")
        # Save z_map in .mat format in the base_path folder
        z_map.to_filename(glm_zmap_dir /f'{contrast_id}.nii.gz')
        # plot the contrasts as soon as they're generated
        # the display is overlaid on the mean fMRI image
        # a threshold of 3.5 is used, more sophisticated choices are possible
        plotting.plot_stat_map(
                z_map,
                bg_img=mean_image,
                threshold=2.58,        
                cut_coords=[0],
                display_mode="y",        
                black_bg=True,        
                title=contrast_id,
                output_file=glm_zmap_dir / f'{contrast_id}.png',
                )

    # # Plot time series for ROI response and stimulus regressor
    # Identify ROIs using an automated clustering approach. Extract the time series for each ROI. Plot the time series and corresponding locations.
    param_glm["standardize"] = "psc"

    output_file = glm_plot_dir / f"{design_matrix_name}_time_series"
    r2_val, figs = get_ROI_activation(nifti_data, fmri_glm, design_matrix, basic_contrasts, time_stamp, task_events, param_glm, output_file)

    # zscore standardization
    param_glm["standardize"] = "zscore"

    output_file = glm_plot_dir / f"{design_matrix_name}_time_series_and_contrast_predicted"
    get_ROI_activation(nifti_data, fmri_glm, design_matrix, basic_contrasts, time_stamp, task_events, param_glm, output_file)

    # log the labels to neptune
    if log_neptune is not None:
        try:
            log_neptune["parameters/GLM"] = stringify_unsupported(param_glm)
            log_neptune["outputs/r2"] = stringify_unsupported(r2_val)
            for plot in glm_plot_dir.iterdir():
                fname = plot.name
                log_neptune["outputs/plots/GLM/" + fname].upload(str(plot))
        except Exception as e:
            print(f'Error in logging GLM outputs: {e}')
    return

def main(BIDS_dir, filename, log_neptune_flag):
    #################################################################
    #           Define output path locations and initialize run     #
    #################################################################

    # Get the registration and GLM output directories
    register_dir, glm_dir = utils.get_BIDS_derivative_dir(BIDS_dir)

    # Load the data
    nifti_data, metadata, task_events, exp_id = utils.load_data_from_BIDS(BIDS_dir, filename)
    fusi_data = nifti_data.get_fdata()
    filename = filename.replace('.nii.gz', '')
    folder_tag = filename[re.search(exp_id['run'], filename).start():-5]
    n_images = fusi_data.shape[-1]
    
    # initialize and make directory
    reg_plot_dir = register_dir / 'plots' / folder_tag
    reg_plot_dir.mkdir(parents=True, exist_ok=True)
    glm_zmap_dir = glm_dir / 'zmaps' / folder_tag
    glm_zmap_dir.mkdir(parents=True, exist_ok=True)
    glm_plot_dir = glm_dir / 'plots' / folder_tag
    glm_plot_dir.mkdir(parents=True, exist_ok=True)

    # initialize neptune run
    if log_neptune_flag:
        log_neptune = neptune.init_run(project="forest-neurotech/auto-registration-glm",
                                name=filename,
                                )
        log_neptune["sys/tags"].add([exp_id['run'], exp_id['acq']])
        log_neptune["sys/group_tags"].add([exp_id['sub'], exp_id['sub_type'], exp_id['ses']])
        log_neptune["outputs/n_images"] = n_images
        log_neptune["metadata"] = stringify_unsupported(metadata)
        for key, value in exp_id.items():
            if key in ['run', 'acq']:
                log_neptune["metadata/" + key] = int(value[4:]) # only the number
            else:
                log_neptune["metadata/" + key] = value
    else:
        log_neptune = None

    # Define the list of params to be used in this analysis
    param_c, param_reg, param_glm = set_params(n_images)

    #################################################################
    #                        Main pipeline                          #
    #################################################################

    # Run the analysis pipeline
    el = 0 # elevation
    fusi_data = fusi_data[:,el]

    ## 1. In-plane clustering
    embedding, pred_labels, centroids, closest_points_in_plane = \
        run_in_plane_clustering(fusi_data, param_c, reg_plot_dir, log_neptune)

    ## 2. Image registration
    nifti_data, extra, final_ref_frame = \
    run_registration(fusi_data, metadata, pred_labels, closest_points_in_plane, param_reg, 
                                                    reg_plot_dir, filename, log_neptune)
    
    ### 2.5 need input from both clustering and registration
    plot_embedding_clusters(embedding, pred_labels, centroids, final_ref_frame, reg_plot_dir, 
                            log_neptune)

    ## 3. GLM analysis
    run_glm(nifti_data, task_events, extra, metadata, param_glm, glm_zmap_dir, glm_plot_dir, 
            log_neptune)

    ## 3.5 GLM analysis (repeat without transform values in design matrix)
    param_glm["apply_transform_in_design_mtx"] = False
    run_glm(nifti_data, task_events, extra, metadata, param_glm, glm_zmap_dir, glm_plot_dir, 
            log_neptune)
    
    # End of analysis
    print(f'Analysis for {filename} is completed.')
    
    # End neptune run
    if log_neptune_flag:
        log_neptune.stop()

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    # Required positional arguments 
    parser.add_argument("BIDS_dir", type=Path,
                        help = "Parsed out session directory path in BIDS folder structure",
                        )
    # Optional arguments
    parser.add_argument("--run", type=int, 
                        help="run number to analyze, if not provided, will analyze all runs",
                        default=[],
                        )
    parser.add_argument("--neptune",
                        help="If set True, outputs and params will be logged on neptune",
                        action='store_true',
                        )
    
    args = parser.parse_args()
    BIDS_dir = args.BIDS_dir
    run_id = args.run
    log_neptune_flag = args.neptune

    # If NIFTI has not been parsed in to fUSI_BIDS format, use the following line on terminal to parse data:
    # ```python fUSI_to_BIDS_session.py /2024-06-13/UCLA_006/ --root /cassini/UCLA_collaboration```
    # e.g. if run all runs and log to neptune:
    #   python analysis_one_step_pipeline.py /Users/ewina/Downloads/UCLA_fUSI_BIDS/sourcedata/sub-UCLA_006/ses-2024-06-13 --neptune
    # e.g. if run a single run and log to neptune:
    #   python analysis_one_step_pipeline.py /Users/ewina/Downloads/UCLA_fUSI_BIDS/sourcedata/sub-UCLA_006/ses-2024-06-13 --run 6 --neptune

    if run_id: # if provided as int
        run_folder = f'run-{run_id:02d}'
        filenames = utils.get_run_files_from_BIDS(BIDS_dir, run_folder)
    else: # = default, analyze all runs
        filenames = utils.get_run_files_from_BIDS(BIDS_dir, 'run-')
    
    for filename in filenames:
        try:
            print('\n'+'='*100+'\n')
            main(BIDS_dir, filename, log_neptune_flag)
        except Exception as e:
            print(f'Error in processing {filename}: {e}, skipping to next file...')
            continue