import numpy as np

from skimage import restoration
from scipy.spatial.distance import cdist


def find_median_image_idx(images):
    """Find the median image index (unlikely to be artifactual).

    Parameters
    ----------
    images : np.ndarray (n_images, width, height, depth)
        The images to check.

    Returns
    -------
    image_idx : int
        The median image index.
    """
    order = np.argsort(images, axis=0)
    extremeness = np.mean(order, axis=(1, 2, 3))
    image_idx = np.argsort(extremeness)[extremeness.size // 2]
    return image_idx


# image metrics
# 1. contrast-to-noise ratio from max region of interest
# 2. resolution/rank of image
# 3. coherence
# 4. speckle size

def cnr(img: np.ndarray, roi: np.ndarray, thresh_roi: int,
        thresh_noise: int) -> float:
    """Computes the contrast-to-noise ratio.

    Parameters
    ----------
    img : np.ndarray
        The input image.
    roi : np.ndarray | None
        Region of interest in the form x1, y1, x2, y2.
        Noise will be computed outside this region at
        the same depth.
    thresh_roi : int
        The number of highest-intensity voxels to include
        in the signal calculations.
    thresh_noise : int
        The number of lowest-intensity voxels to include
        in the noise calculations.
    
    Returns
    -------
    cnr : float
        The contrast-to-noise ratio.
    """
    if roi is None:
        roi = np.indices(img.shape[:-1]).reshape(3, -1).T
    dmin, dmax = roi[:, -1].min(), roi[:, -1].max()
    roi_idxs = [tuple(idx) for idx in roi]
    roi_pixels = np.array([img[idx] for idx in roi_idxs])
    if thresh_roi is not None and thresh_roi < roi_pixels.shape[0]:
        keep = np.argsort(roi_pixels.mean(axis=1))[-thresh_roi:]
        roi_idxs = [roi_idxs[i] for i in keep]
        roi_pixels = roi_pixels[keep]
    roi_idxs = set(roi_idxs)
    noise_idxs = np.indices(img.shape[:-1]).reshape(3, -1).T
    noise_idxs = noise_idxs[(noise_idxs[:, -1] >= dmin) &
                            (noise_idxs[:, -1] <= dmax)]
    noise_idxs = set([tuple(idx) for idx in noise_idxs
                      if tuple(idx) not in roi_idxs])
    noise_pixels = np.array([img[tuple(idx)] for idx in noise_idxs])
    if thresh_noise is not None and thresh_noise < noise_pixels.shape[0]:
        noise_pixels = noise_pixels[
            np.argsort(noise_pixels.mean(axis=1))[:thresh_noise]]
    num = np.mean(roi_pixels) - np.mean(noise_pixels)
    den = np.sqrt(np.var(roi_pixels) + np.var(noise_pixels))
    return num / den


def cnr_rolling_ball(img: np.ndarray):
    """Computes the contrast-to-noise ratio using the rolling ball
       method to define the background.

    Parameters
    ----------
    img : np.ndarray
        The input image.
    
    Returns
    -------
    cnr : float
        The contrast-to-noise ratio.
    """
    foreground = img.copy()
    background = np.zeros_like(img)
    for i in range(img.shape[-1]):
        background[..., i] = restoration.rolling_ball(img[..., ])
        foreground[..., i] -= background[..., i]
        foreground[..., i] = np.where(foreground[..., i] > background[i].max(),
                                      foreground[..., i], np.nan)
    return (np.nanmean(foreground) - np.mean(background)) / \
        np.sqrt(np.nanvar(foreground) - np.var(background))



def neighbor_coherence(img: np.ndarray) -> float:
    """Computes the coherence across distances.

    Parameters
    ----------
    img : np.ndarray
        The input image.

    Returns
    -------
    coh : float
        Image coherence.
    """
    neighbors = list()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i - 1 >= 0:
                neighbors.append(((i, j), (i - 1, j)))
            if i + 1 < img.shape[0]:
                neighbors.append(((i, j), (i + 1, j)))
            if j - 1 >= 0:
                neighbors.append(((i, j), (i, j - 1)))
            if j + 1 < img.shape[1]:
                neighbors.append(((i, j), (i, j + 1)))
    neighbor_correlations = np.zeros((len(neighbors)))
    for i, (from_idx, to_idx) in enumerate(neighbors):
        neighbor_correlations[i] = \
            np.corrcoef(img[from_idx], img[to_idx])[0, 1]
    return np.mean(neighbor_correlations)


def speckle_size(img: np.ndarray, tx_freq: float,
                 speed_of_sound: float=1540) -> float:
    """Computes the speckle size of the image.

    Parameters
    ----------
    img : np.ndarray
        The input image.
    tx_freq : float
        The transmission frequency.

    Returns
    -------
    speckle_size : float
        The speckle size
    """

    '''
    horizontalFWHM = np.zeros((10, 10))  # First dimension is number of temporal repetition of each Bmode
    verticalFWHM = np.zeros((10, 10))    # Second dimension is different depths at which we look at the size of the speckle

    # Convert pixels in m
    size_pixel_wavelength = row['Pixel size wavelength']
    wavelength_m = speed_of_sound / tx_freq
    size_pixel_m = size_pixel_wavelength * wavelength_m
    ten_mm_in_pixel = 0.010 / size_pixel_m
    five_mm_in_pixel = 0.005 / size_pixel_m
    temporal_acq = Bmode_temp.shape[0]

    for depth in range(1, 10):
        for time in range(temporal_acq):  
            # iteration over temporal_acq and at different depths per slice of 10mm
            depth2study = Bmode_temp[time, :, round(depth*ten_mm_in_pixel-five_mm_in_pixel):round(depth*ten_mm_in_pixel+five_mm_in_pixel)] 
            size1, size2 = depth2study.shape[0], depth2study.shape[1] 

            # Create a subregion of the overall image to be used for autocorrelation
            startIdx1 = round(size1 / 4) + 10
            endIdx1 = size1 - round(size1 / 4) - 10
            startIdz1 = round(size2 / 4) + 10
            endIdz1 = size2 - round(size2 / 4) - 10  
            image2study1 = depth2study[startIdx1:endIdx1, startIdz1:endIdz1] 
            
            # initialize autocorrelation matrix
            auto_corr = np.zeros((40, 40))
            half_auto = auto_corr.shape[0] // 2 

            # Create sliding window for autocorrelation
            for x in range(-half_auto, half_auto):  
                for z in range(-half_auto, half_auto): 

                    startIdx2 = startIdx1 + x
                    endIdx2 = endIdx1 + x
                    startIdz2 = startIdz1 + z
                    endIdz2 = endIdz1 + z

                    image2study2 = depth2study[startIdx2:endIdx2, startIdz2:endIdz2]  

                    cross_corr = np.multiply(image2study1, image2study2) 
                    auto_corr[x + 20, z + 20] = np.sum(cross_corr) 

            auto_corr = auto_corr - np.min(auto_corr)
            auto_corr = auto_corr / np.max(auto_corr)  # Normalize
            # plt.imshow(auto_corr, cmap='hot', interpolation='nearest')
            # plt.colorbar()
            # plt.show()

            # Horizontal FWHM
            horizontalLine = auto_corr[:, half_auto]
            indices = np.where(horizontalLine >= 0.5)[0]  # Find indices where condition is True
            if indices.size > 0:
                index1 = indices[0]  # First index
                index2 = indices[-1]  # Last index
                horizontalFWHM[time, depth] = index2 - index1 + 1
            else:
                horizontalFWHM[time, depth] = 0 

            # Vertical FWHM
            verticalLine = auto_corr[half_auto, :]
            indices = np.where(verticalLine >= 0.5)[0]  # Find indices where condition is True
            if indices.size > 0:
                index1 = indices[0]  # First index
                index2 = indices[-1]  # Last index
                verticalFWHM[time, depth] = index2 - index1 + 1
            else:
                verticalFWHM[time, depth] = 0  

    averagehorizontalSpeckle = np.mean(horizontalFWHM, axis=0)*size_pixel_m
    averageverticalSpeckle = np.mean(verticalFWHM,axis=0)*size_pixel_m

    horizontalSpeckle.append(averagehorizontalSpeckle)  
    verticalSpeckle.append(averageverticalSpeckle)
    '''


def lag_one_coherence(rxdata):
    """
    Lag-one coherence of the receive aperture (DOI: 10.1109/TUFFC.2018.2855653).
    The LOC measures the quality of a signal relative to its noise, and can be
    used to select acoustic output.
    """
    # Compute the correlation coefficient
    xy = np.real(np.nansum(rxdata[:-1] * np.conj(rxdata[1:]), axis=0))
    xx = np.nansum(np.abs(rxdata[:-1]) ** 2, axis=0)
    yy = np.nansum(np.abs(rxdata[1:]) ** 2, axis=0)
    ncc = xy / np.sqrt(xx * yy)
    return ncc
