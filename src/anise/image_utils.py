import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.fft import fft2, ifft2, fftshift
from scipy.ndimage import shift
from scipy import ndimage

import nibabel as nib
import nilearn as nl
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn import plotting

# from anise.io.load_matlab_dataset import load_data, load_selected_data
import SimpleITK as sitk
import ants

def show_imgs(imgs, frame_indices=np.arange(10), timestamps=None, labels=None, fig_height=2, title=[], clim=None):
    # plot subplot frames
    r = len(frame_indices) // 10
    r = r + 1 if len(frame_indices) % 10 != 0 else r
    c = len(frame_indices) if r == 0 else 10
    fig, axs = plt.subplots(r, c, figsize=(12, fig_height*r))
    for i in range(r*c):
        ax = axs[i // 10, i % 10] if r > 1 else axs[i]
        if i < len(frame_indices):
            frame_num = frame_indices[i]
            if clim is not None and len(clim) == 2:
                ax.imshow(imgs[..., frame_num].T, cmap='hot', vmin=clim[0], vmax=clim[1])
            else:
                ax.imshow(imgs[..., frame_num].T, cmap='hot')
            ax.axis('off')
            if title != 'off':
                if timestamps is not None:
                    ax.set_title(f'{timestamps[frame_num]} s')
                elif labels is not None:
                    ax.set_title(f'frame {frame_num},\nlabels={labels[frame_num]}')
                else:
                    ax.set_title(f'frame {frame_num}')
        else:
            ax.remove()
    plt.tight_layout(pad=0.3, w_pad=0.1)
    plt.show()
    return fig, axs

def plot_pd_intensity_over_time(imgs, y_lim=[], outlier_threshold=0):
    fig, ax = plt.subplots()
    # get the average intensity over time by the axis
    doppler_intensity = np.mean(imgs, axis=tuple(range(0, imgs.ndim - 1)))
    
    plt.plot(doppler_intensity)
    plt.title(f'Average Doppler Intensity Over Time = {doppler_intensity.mean().round()}')
    plt.xlabel('Frame #')
    plt.ylabel('Average Intensity')
    if len(y_lim) == 2:
        plt.ylim(y_lim[0], y_lim[1])
    else:
        plt.ylim(0, max(500, doppler_intensity.max()))
    
    # identify the frames with intensity above a threshold (user defined)
    if outlier_threshold > 0:
        median_intensity = np.median(doppler_intensity)
        outlier_intensity_frames = np.abs(doppler_intensity - median_intensity) > outlier_threshold
        outlier_frames = np.where(outlier_intensity_frames)[0]
        print(f'Outlier frames more than {outlier_threshold:.1f} from median: ', outlier_frames)
        
        # plot the median intensity and the outliers
        plt.hlines(median_intensity, 0, len(doppler_intensity),color='k', label='Median Intensity')
        plt.hlines(median_intensity - outlier_threshold, 0, len(doppler_intensity), 
        color='k', linestyle='--')
        plt.hlines(median_intensity + outlier_threshold, 0, len(doppler_intensity), 
        color='k', linestyle='--', label='threshold')
        plt.scatter(outlier_frames, 
                    doppler_intensity[outlier_intensity_frames], 
                    color='r', label='Outlier indices')
        plt.legend()
    else:
        outlier_intensity_frames = []
    plt.show()
    return doppler_intensity, outlier_intensity_frames, fig, ax

def phase_corr_translation(im1, im2):
    
    """ performs FFT phase correlation to find optimal translation between two images

    Kuglin, C. D. and Hines, D. C., 1975. The Phase Correlation Image Alignment Method. 
    Proceeding of IEEE International Conference on Cybernetics and Society, pp. 163-165, New York, 
    NY, USA.

    Args:
        im1, im2 : 2D numpy arrays : image 2 to be aligned to image 1
    returns:
        x, y : int : translation in x and y directions
        ir: cross-correlation

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

def motion_correct_with_phase_correlation(imgs, im_ref=None):
    imgs = imgs.transpose(2,0,1)
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
    imgs = imgs.transpose(2,0,1)

    ir_avg = np.zeros([imgs.shape[0],imgs.shape[0]])
    fft_imgs = np.zeros([imgs.shape[0],imgs.shape[1],imgs.shape[2]], dtype=np.complex_)
    for i in range(len(imgs)):
        fft_imgs[i] = fft2(imgs[i])

    for i in range(len(imgs)):
        for j in range(i+1,len(imgs)):
            ir = abs(ifft2((fft_imgs[i] * fft_imgs[j].conjugate()) / (abs(fft_imgs[i]) * abs(fft_imgs[j]))))
            ir = np.expand_dims(ir, axis=0)
            ir_avg[i,j] = ir.var()
    return ir_avg

# registration related
def get_threshold_image(imgs, threshold=.5):

    # Apply thresholding to isolate vessels
    threshold_imgs = np.zeros_like(imgs)
    density = np.zeros(imgs.shape[-1])
    for i in range(imgs.shape[-1]):
        image = imgs[...,i]
        # Normalize the image zscore
        normalized_image = (image - np.mean(image)) / np.std(image)
        # normalized_image = image / np.max(image)
        image_2 = normalized_image

        # Apply Gaussian blur to reduce noise
        image_blurred = ndimage.gaussian_filter(normalized_image, 1, mode='nearest')
        image_2 = image_blurred

        # Set pixel above threshold is set to 1, otherwise 0
        image_thresh = np.zeros_like(image_2)
        image_thresh[image_2 > threshold] = 1
        
        # Count the number of vessel pixels
        vessel_pixels = np.sum(image_thresh > 0)
        
        # Calculate vessel density (percentage of vessel pixels)
        total_pixels = image.shape[0] * image.shape[1]
        vessel_density = (vessel_pixels / total_pixels) * 100

        threshold_imgs[...,i] = image_thresh
        density[i] = vessel_density
    return threshold_imgs, density

def register_sitk(images, metric='correlation', ref_index=0):
    sitk_images = [sitk.GetImageFromArray(images[..., i]) for i in range(images.shape[-1])]
    # Set the fixed image
    fixed_image = sitk_images[ref_index]

    # Init arrays with first fixed image
    transformed_images = [fixed_image]
    extra = []
    # Setup registration method to be more constrained
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric
    if metric == 'correlation':
        registration_method.SetMetricAsCorrelation()
    elif metric == 'mean_squares':
        registration_method.SetMetricAsMeanSquares()
    elif metric == 'mutual_information':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == 'normalized_correlation':
        registration_method.SetMetricAsANTSNeighborhoodCorrelation()

    # optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Setup the transform
    initial_transform = sitk.Euler2DTransform()
    initial_transform.SetIdentity()
    registration_method.SetInitialTransform(initial_transform)

    # Apply registration
    for i, moving_image in enumerate(sitk_images):
        # Set a tighter initial transform using moments
        initial_transform = sitk.Euler2DTransform(sitk.CenteredTransformInitializer(fixed_image, 
                                                                                    moving_image, sitk.Euler2DTransform(), 
                                                                                    sitk.CenteredTransformInitializerFilter.MOMENTS))
        registration_method.SetInitialTransform(initial_transform)

        # get the similarity metric value before registration
        similarity_metric_before = registration_method.MetricEvaluate(fixed_image, moving_image)

        # Execute the registration method
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        # get transformation value by registration
        tx, ty = final_transform.GetTranslation()
        angle = final_transform.GetAngle()
        # get the similarity metric value after registration
        similarity_metric_after = registration_method.GetMetricValue()

        # append the transformation parameters and similarity metric value to extra
        extra.append({'translation_x': tx, 
                      'translation_y': ty, 
                      'rotation': angle, 
                      'initial_similarity': similarity_metric_before, 
                      'after_similarity': similarity_metric_after,
                      })

        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 
                                        0.0, moving_image.GetPixelID())
        resampled_image = sitk.GetArrayFromImage(resampled_image)
        transformed_images.append(resampled_image)

    registered_array = np.stack(transformed_images, axis=-1)
    extra_df = pd.DataFrame(extra)

    return registered_array, extra_df

def register_ants(images, type_of_transform='Rigid', ref_index=0, crop_border=0):

    # use image nearest to centroid as the reference frame
    print(f'Reference frame for registration is: {ref_index}')
    fixed_image = ants.from_numpy(images[..., ref_index])
    transformed_images = []

    # Initialize transformations array with zero for the first image (identity transformation)
    extra = []
    for i in range(images.shape[-1]):
        moving_image = ants.from_numpy(images[..., i])
        result = ants.registration(fixed=fixed_image, 
                                   moving=moving_image, 
                                   type_of_transform='Rigid',
                                   random_seed=1001,
                                   )

        # Extract the transformed image for visualization or further analysis
        transformed_image = result['warpedmovout']

        # get the similarity metric value before and after registration
        # negative because the similarity metric is minimized for optimization
        similarity_metric_before = - ants.image_mutual_information( fixed_image, moving_image )
        similarity_metric_after = - ants.image_mutual_information( fixed_image, transformed_image )
        
        # Crop the transformed image to remove borders if crop_border is set to a value greater than 0
        lateral, depth = transformed_image.shape[0], transformed_image.shape[1]
        transformed_image = ants.crop_indices(transformed_image, 
                                              lowerind=(crop_border,crop_border), 
                                              upperind=(lateral - crop_border, depth - crop_border) )
        
        # Append the transformed image to the list
        transformed_images.append(transformed_image.numpy())

        # Extract parameters from the transform
        transform = result['fwdtransforms'][0] if len(result['fwdtransforms']) > 0 else None
        if transform:
            # Using ANTsPy API to extract parameters
            transform_object = ants.read_transform(transform)
            tx, ty = transform_object.parameters[4:6]  # Assuming these indices contain translations
            angle = transform_object.parameters[2]  # Assuming this index contains the rotation angle
        else:
            tx, ty, angle = 0, 0, 0
        # append the transformation parameters and similarity metric value to extra
        extra.append({'translation_x': tx, 
                      'translation_y': ty, 
                      'rotation': angle, 
                      'initial_similarity': similarity_metric_before, 
                      'after_similarity': similarity_metric_after,
                      })

    registered_images = np.stack(transformed_images, axis=-1)
    print('Registered images shape:', registered_images.shape)
    extra_df = pd.DataFrame(extra)

    return registered_images, extra_df

def crop_images(images, crop_border):
    if isinstance(crop_border, int):
        x_l = x_r = y_l = y_r = crop_border
    elif isinstance(crop_border, list):
        x_l, x_r, y_l, y_r = crop_border
    transformed_images = []
    for i in range(images.shape[-1]):
        image = images[..., i]
        lateral, depth = image.shape[0], image.shape[1]
        cropped_image = image[x_l:lateral - x_r, y_l:depth - y_r]
        transformed_images.append(cropped_image)
    return np.stack(transformed_images, axis=-1)

def plot_transform_params(extra, oop_labels=[], metric_name='', in_plane_indices=[], 
                            annotate_ref_frame: int = None):
    # Create a new figure with three subplots
    # similarity metric value
    # translation after registration
    # rotation after registration

    if not any(in_plane_indices):
        in_plane_indices = range(len(extra))
    fig, axs = plt.subplots(3, 1,figsize=(8,8))

    # get the range of oop start stops for plotting
    oop_labels_pad = np.pad(oop_labels, (1, 1), 'constant', constant_values=(0, 0))
    range_oop = [i for i in range(len(oop_labels_pad)-1) if oop_labels_pad[i+1] - oop_labels_pad[i] != 0]
    if len(range_oop) == 1:
        range_oop.append(len(oop_labels)) # append last index at the end

    def add_shaded_oop_indices(ax, oop_labels):
    # add shaded region for out-of-plane frames
        if len(range_oop) > 0:
            ax.axvspan(range_oop[0], range_oop[1], color='k', alpha=0.2, label=f'Discarded frames')
            for i in range(2, len(range_oop)-1, 2):
                ax.axvspan(range_oop[i], range_oop[i+1], color='k', alpha=0.2)

    def add_annotation_ref_frame(ax, annotate_ref_frame, y):
        if annotate_ref_frame is not None:
            ax.scatter(annotate_ref_frame, y[annotate_ref_frame], color='magenta', 
                        label=f'reference frames: {annotate_ref_frame}')

    ax = axs[0]
    ax.plot(in_plane_indices, extra['initial_similarity'], label='initial_similarity')
    ax.plot(in_plane_indices, extra['after_similarity'], label='after_similarity')
    before = extra['initial_similarity'].mean()
    after = extra['after_similarity'].mean()
    improvement = (after - before) / before * 100
    add_shaded_oop_indices(ax, range_oop)
    add_annotation_ref_frame(ax, annotate_ref_frame, extra['after_similarity'])
    ax.set_title(f'Similarity Metric: {metric_name}, {improvement:.2f}% improvement')
    ax.set_ylabel('Similarity Metric')
    ax.legend()
    ax.grid(True)

    ax = axs[1]
    ax.plot(in_plane_indices, extra['translation_x'], label='x')
    ax.plot(in_plane_indices, extra['translation_y'], label='y')
    add_shaded_oop_indices(ax, range_oop)
    add_annotation_ref_frame(ax, annotate_ref_frame, extra['translation_x'])
    ax.set_title('Translation')
    ax.set_ylabel('pixel')
    lim = max(np.abs(extra['translation_x']).max(), np.abs(extra['translation_y']).max(), 5)
    ax.set_ylim(-lim, lim)
    ax.legend()
    ax.grid(True)

    ax = axs[2]
    ax.plot(in_plane_indices, extra['rotation'], label='rotation')
    add_shaded_oop_indices(ax, range_oop)
    add_annotation_ref_frame(ax, annotate_ref_frame, extra['rotation'])
    ax.set_title('Rotation')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Angle')
    lim = max(np.abs(extra['rotation'].max()), 0.1)
    ax.set_ylim(-lim, lim)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, axs, improvement

def boxcar_smooth(data, window_size):
    # smooth across frames with a boxcar moving average
    assert window_size>0, "Window size must be greater than 0"
    smoothed_data = np.copy(data)
    for t in range(data.shape[-1]):
        start_index = max(0, t - window_size // 2)
        end_index = min(data.shape[-1], t + window_size // 2 + 1)
        smoothed_data[..., t] = np.mean(data[..., start_index:end_index], axis=-1)
    return smoothed_data

def check_weird_transform(extra, max_translation_limit=10, angle_limit=0.1):
    # check if the transformation is not smooth based on the difference between consecutive frames
    outlier_x = extra['translation_x'].diff().abs().max()
    outlier_y = extra['translation_y'].diff().abs().max()
    outlier_a = extra['rotation'].diff().abs().max() 
    if outlier_x > max_translation_limit: # pixel
        print(f'Weird transformation detected with {outlier_x:.1f} pixel difference along the x axis.')
        return True
    if outlier_y > max_translation_limit: # pixel
        print(f'Weird transformation detected with {outlier_y:.1f} pixel difference along the y axis.')
        return True
    if outlier_a > angle_limit: # angle (radian)
        print(f'Weird transformation detected with {outlier_a:.1f} radian in rotation.')
        return True
    return False

# cluster related
def get_consecutive_labels_counts(labels):
    labels_tmp = np.append(labels, 0.01)
    summary_count = {}
    count_consecutive = 1
    prev_label = labels_tmp[0]
    for i in range(1, len(labels_tmp)):
        curr_label = labels_tmp[i]
        if curr_label == prev_label:
            count_consecutive += 1
        else:
            if prev_label not in summary_count:
                summary_count[prev_label] = [count_consecutive]
            else:
                summary_count[prev_label].extend([count_consecutive])
            count_consecutive = 1
        prev_label = curr_label
    return summary_count

def remap_cluster_to_labels(original_labels):
    if np.any(original_labels==-1):
        return original_labels
    
    # Re-order the labels and assign plane classes
    unique_labels, label_counts = np.unique(original_labels, return_counts=True)
    first_label_appear = np.zeros(len(unique_labels), )
    for i in range(len(unique_labels)):
        first_label_appear[i] = np.where(original_labels == unique_labels[i])[0][0]
    labels = pd.DataFrame({'orginal': original_labels})

    # mapping kmean labels to string labels 
    mapping_dict = {}
    # discard the plane with the least number of points
    discard_index = np.argmin(label_counts)
    mapping_dict[discard_index] = 'discard'
    # in-plane is the plane with the most number of points
    in_plane_index = np.argmax(label_counts)
    mapping_dict[in_plane_index] = 'in-plane'
    unique_labels = np.delete(unique_labels,[in_plane_index, discard_index])
    # order the remaining clusters of plane based on their first appearance
    order_ind = np.argsort(first_label_appear[unique_labels])
    unique_labels = unique_labels[order_ind]
    for i in range(len(unique_labels)):
        mapping_dict[unique_labels[i]] = 'new oop ' + str(i+1)
    # add string labels col to dataframe
    labels['class'] = labels['orginal'].map(mapping_dict)

    # mapping string labels to new labels
    mapping_dict = {}
    mapping_dict = {'discard':-1, 'in-plane': 0}
    if len(label_counts) > 2:
        for i in range(1, len(label_counts)-1):
            mapping_dict['new oop ' + str(i)] = i
    # add relabelled labels col to dataframe
    labels['remap_kmean'] = labels['class'].map(mapping_dict)
    pred_labels = labels['remap_kmean'].values

    return pred_labels

def plot_embedding(embedding, labels=None, title='', legend='on', centroids=[]):
    if labels is None:
        # labels not given, assign all to in-plane
        labels = np.zeros((embedding.shape[0], ), dtype=np.int32)
        legend = 'off'
    else:
        order = np.argsort(labels)
        labels = labels[order]
        embedding = embedding[order]

    colors = ['g', 'c', 'm', 'y', 'k', 'b']
    label_classes = {}
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        colors = ['g'] # all in-plane
    elif -1 in unique_labels:
        colors = ['r'] + colors # discard plane label with red

    for l in unique_labels:
        if l == 0:
            label_classes[l] = 'in-plane'
        elif l < 0:
            label_classes[l] = 'discard out-of-plane ' + str(l)
        elif l > 0:
            label_classes[l] = 'new out-of-plane ' + str(l)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_prop_cycle('color', colors)
    for l in unique_labels:
        ax.scatter(embedding[labels == l, 0],embedding[labels == l, 1],s=5,label=label_classes[l])
    plt.gca().set_aspect('equal', 'datalim')
    if len(centroids) > 0:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='k', s=100, marker='x')
    if legend == 'on':
        fig.legend(loc='lower left', bbox_to_anchor=(1.02,0), ncol=1)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return fig, ax
