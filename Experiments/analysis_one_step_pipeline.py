# %% [markdown]
# # Basic demonstration of fUSi analysis
# 
# This notebook shows the basic demonstration of analyzing audio data collected at UCLA

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

# %%
#################################################################
#      Define data paths and choose output path location        #
#################################################################

# If NIFTI has not been parsed in to fUSI_BIDS format, use the following line on terminal to parse data:
# ```python fUSI_to_BIDS_session.py /2024-06-13/UCLA_006/ --root /cassini/UCLA_collaboration```

base_path = Path.home() / 'Downloads/UCLA_fUSI_BIDS/sourcedata/sub-UCLA_006/ses-2024-06-13'

run = 5

log_neptune = True

run_folder = f'run-{run:02d}'

# Get the registration and GLM output directories
register_dir, glm_dir = utils.get_BIDS_derivative_dir(base_path)

# find filenames that match with the run
filenames = utils.get_run_files_from_BIDS(base_path, run_folder)
# for filename in filenames:
filename = filenames[0]
nifti_data, metadata, task_events = utils.load_data_from_BIDS(base_path, filename)

fusi_data = nifti_data.get_fdata()
filename = filename.replace('.nii.gz', '')
folder_tag = filename[re.search(run_folder, filename).start():-5]

reg_plot_dir = register_dir / 'plots' / folder_tag
reg_plot_dir.mkdir(parents=True, exist_ok=True)

# %%
# Define the list of params to be used in this analysis

run = {}
if log_neptune:
    run = neptune.init_run(
    project="forest-neurotech/auto-registration-glm",
    name=filename,
    )

# clustering parameters
param_c = dict(
    outlier_threshold = 500, # pwd intensity
    dim_red_method = 'PCA', # 'PCA' or 'UMAP' or 'tSNE' or 'VGG'
    pca_n_components = min(100, fusi_data.shape[-1] - 1), # not applied for VGG
    perplexity = min(200, fusi_data.shape[-1] - 1), # specific for tSNE
    DBSCAN_eps = 1, # smaller eps, more clusters
    )

# registration parameters
param_reg = dict(
    use_thresholded_mask = False,
    auto_optimize_ref_frame = True,
    metric_name = 'mutual_information',        
    apply_image_registration = True,
    apply_temporal_smoothing = False,
    temporal_smoothing_window_size = 4, # Number of time points for smoothing
    crop_border = [0, 0, 0, 0], # crop the image to remove border
    )

# GLM parameters  
param_glm =  dict(
    apply_transform_in_design_mtx = True,
    smoothing_fwhm = .5, #0 spatial smoothing in mm to apply to the data before computing GLM
    behavior_offset = 0, # in seconds
    stat_threshold = 1,
    num_locations = 2, # How many ROIs to visualize. An additional random ROI will be added
    )

# %%
################################################
#     Display Doppler intensity time course    #
################################################

# find and confirm if there's any outlier frames that should be removed e.g. high brightness or noise
doppler_intensity_per_frame, outlier_intensity_frames, fig, ax = \
    motion_utils.plot_pd_intensity_over_time(fusi_data,
                                            outlier_threshold=param_c["outlier_threshold"])
plt.show(block=False)
fig.savefig(reg_plot_dir / 'doppler_intensity_over_time.png', dpi=300, bbox_inches='tight')
plt.close(fig)
# %%
#############################################################################
#             If needed: identify any out-of-plan images                    #
#############################################################################

# elevation
el = 0
# dimensionality reduction and clustering
embedding = motion_utils.dim_red(fusi_data[:,el], param_c)
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
print(f'outlier frames from feature extraction + clustering: {pred_labels_oop_ind}')
print(f'outlier frames from power doppler intensity: {outliers_intensity_ind}')
# calculate overlap between outlier_intensity_frames and pred_labels
if not any(pred_labels_oop_ind) and not any(outliers_intensity_ind):
    overlap = len(set(pred_labels_oop_ind).intersection(outlier_intensity_frames))
    print('Percentage of overlap between the two types of outlier', \
        overlap/len(outlier_intensity_frames)*100 if len(outlier_intensity_frames) > 0 else \
        overlap/len(pred_labels_oop_ind)*100)
# print predicted labels 20 in a row
print('\nRemapped predicted labels:')
for row in range(len(pred_labels) // 20 + 1):
    print(row*20, '\t', pred_labels[row * 20:(row + 1) * 20])
print(f'% of in-plane frames: {len(in_plane_indices)/len(pred_labels)*100:.2f}%')

# %%
# show the outlier images, if any
if np.any(labels != 0):
    fig, axs = motion_utils.show_imgs(fusi_data[:,el], np.where(labels != 0)[0])
fig.savefig(reg_plot_dir / 'out_of_plane_images.png', dpi=300, bbox_inches='tight')
plt.close(fig)

centroids, closest_points_in_plane = motion_utils.sort_in_plane_points_closest_to_centroid(embedding, pred_labels, n_clusters)

#############################################################################
# If needed: perform image registration and display transformation dynamic  #
#############################################################################

# Perform image registration on in-plane frame indices
if param_reg["apply_image_registration"]:
    registered_images, extra, final_ref_frame = motion_utils.auto_ants_registration_on_in_plane(
        fusi_data[:, el], in_plane_indices, closest_points_in_plane, 
        optimize_ref=param_reg["auto_optimize_ref_frame"])

    # plot registration transformation outputs
    fig, axs, prct_improve = motion_utils.plot_transform_params(extra, oop_labels=pred_labels, 
                                    metric_name=param_reg["metric_name"], 
                                    annotate_ref_frame=final_ref_frame)
    fig.savefig(reg_plot_dir / 'transform_params.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    # label the nearest point to the in-plane centroid in the embedding space
    fig, ax = motion_utils.plot_embedding(embedding, pred_labels, centroids=centroids, 
                           title=f'Cluster centroid and chosen reference image (#{final_ref_frame})')
    loc = in_plane_indices[final_ref_frame]
    ax.scatter(embedding[loc, 0], embedding[loc, 1], s=10, c='m')

else:
    registered_images = None
    final_ref_frame = None
    extra = pd.DataFrame({'in_plane_indices': in_plane_indices})
    fig, ax = motion_utils.plot_embedding(embedding, pred_labels, centroids=centroids, 
                           title=f'Clusters and centroids')

fig.savefig(reg_plot_dir / 'embedding_clusters.png', dpi=300, bbox_inches='tight')       
plt.close(fig)

# Save outputs as csv
tsv_filename = register_dir / "fus" / (filename[:-4]+'outputs.tsv')
extra.to_csv(tsv_filename, index=False, sep='\t')

# save the registered images to a gif or a mp4
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

# crop the border of the registered images
if any(param_reg["crop_border"]) and registered_images is not None:
    registered_images = motion_utils.crop_images(registered_images, crop_border=param_reg["crop_border"])

N=2 # duplicate the data along the elevation dimension
# save the registered images to nifti
if param_reg["apply_image_registration"]:
    reg_pwd = np.tile(np.expand_dims(registered_images, 1), (1, N, 1, 1))
    if param_reg["apply_temporal_smoothing"]:
        pd = utils.boxcar_smooth(reg_pwd, param_reg["temporal_smoothing_window_size"])
        filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", pd, 
                                             filename=filename, 
                                             filename_tag='register_smooth')
    else:
        filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", reg_pwd, 
                                             filename=filename, 
                                             filename_tag='register')
else:
    data_selected = fusi_data[:, :, :, in_plane_indices]
    if param_reg["apply_temporal_smoothing"]:
        pd = utils.boxcar_smooth(data_selected, param_reg["temporal_smoothing_window_size"])
        filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", pd,
                                             filename=filename, 
                                             filename_tag='in_plane_no_reg_smooth')
    else:
        filename_new = utils.save_nifti_to_BIDS(register_dir / "fus", data_selected,
                                             filename=filename, 
                                             filename_tag='in_plane_no_reg')

nifti_full_path = register_dir / "fus" / f'{filename_new}.nii.gz'

# load in nifti format
nifti_data = nib.load(nifti_full_path)
print(f'Loaded nifti data from {nifti_full_path}')
print(f'Nifti data shape: {nifti_data.shape}')

if log_neptune:
    run["parameters/clustering"] = param_c
    run["parameters/registration"] = param_reg
    run["outputs/transforms"].upload(str(tsv_filename))
    run["plots/motion_correct/"].upload_files(str(reg_plot_dir))
# %%
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
from anise.process.fusi_glm_fit import get_ROI_activation

# initialize the GLM directory
zmap_dir = glm_dir / 'zmaps' / folder_tag
zmap_dir.mkdir(parents=True, exist_ok=True)
glm_plot_dir = glm_dir / 'plots' / folder_tag
glm_plot_dir.mkdir(parents=True, exist_ok=True)

# %%
#######################################
#     Create design matrix for GLM    #
#######################################

hrf_model = "glover"

# Sub-period of time points that are in-plane
time_stamp = np.array(metadata['VolumeTiming'])[in_plane_indices]

if param_glm["apply_transform_in_design_mtx"]:
    design_matrix_name = 'yes_transform_design_mtx'
    design_matrix = make_first_level_design_matrix(
        time_stamp,
        task_events,
        drift_model="polynomial",
        drift_order=1,
        add_regs=extra[["translation_x", "translation_y", "rotation"]].to_numpy(),
        add_reg_names=["tx","ty","rot"],
        hrf_model=hrf_model,
    )
else:
    design_matrix_name = 'no_transform_design_mtx'
    design_matrix = make_first_level_design_matrix(
        time_stamp,
        task_events,
        drift_model="polynomial",
        drift_order=1,
        hrf_model=hrf_model,
    )

ax = plot_design_matrix(design_matrix)
fig = ax.get_figure()
fig.savefig(glm_plot_dir / f'{design_matrix_name}.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# %%
# This cell creates an identity matrix of size equal to the number of columns in the design_matrix.
# It then constructs a dictionary named 'basic_contrasts' where each key is a column name from the design_matrix
# and the corresponding value is a row from the identity matrix. This effectively sets up basic contrasts
# for each experimental condition represented in the design_matrix.

contrast_matrix = np.eye(design_matrix.shape[1])
basic_contrasts = {
    column: contrast_matrix[i]
    for i, column in enumerate(design_matrix.columns)
}
print('Basic contrasts:')
for key in basic_contrasts.keys():
    print(f'{key:15}: {basic_contrasts[key]}')

# %%
# Initialize and fit the GLM model with specified parameters
print("Fitting a GLM")
fmri_glm = FirstLevelModel(minimize_memory=False,
                            mask_img=False,
                            smoothing_fwhm=param_glm["smoothing_fwhm"],
                            standardize=True)
fmri_glm = fmri_glm.fit(nifti_data, design_matrices=design_matrix)

# %%
# Perform first level GLM analsysis

print("Computing contrasts")
mean_image = mean_img(nifti_data)

# Iterate on contrasts
for contrast_id, contrast_val in basic_contrasts.items():
    print(f"\tcontrast id: {contrast_id}")
    # compute the contrasts
    z_map = fmri_glm.compute_contrast(contrast_val, output_type="stat")
    # Save z_map in .mat format in the base_path folder
    z_map.to_filename(zmap_dir /f'{contrast_id}.nii.gz')
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
            output_file=zmap_dir / f'{contrast_id}.png',
            )

# %%
# # Plot time series for ROI response and stimulus regressor
# 
# Identify ROIs using an automated clustering approach. Extract the time series for each ROI. Plot the time series and corresponding locations.

param_glm["standardize"] = "psc"
param_glm["t_r"] = 2.58

output_file = glm_plot_dir / f"{design_matrix_name}_{contrast_id}_time_series.png"

get_ROI_activation(nifti_data, fmri_glm, design_matrix, basic_contrasts, time_stamp, task_events, param_glm, output_file)
plt.close()

#
# zscore standardization
param_glm["standardize"] = "zscore"
param_glm["t_r"] = 3.5

output_file = glm_plot_dir / f"{design_matrix_name}_{contrast_id}_time_series_and_contrast_predicted.png"
get_ROI_activation(nifti_data, fmri_glm, design_matrix, basic_contrasts, time_stamp, task_events, param_glm, output_file)
plt.close()

# %%
# log the labels to neptune
if log_neptune:
    run["parameters/GLM"] = param_glm
    run["plots/GLM/"].upload_files(str(glm_plot_dir))
    run.stop()
