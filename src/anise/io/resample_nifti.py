"""Resample NIFTI images to a standardized affine."""

from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import tyro
from beartype import beartype as typechecker
from jaxtyping import Real, jaxtyped
from loguru import logger
from nilearn.image import resample_to_img
from nilearn.image.resampling import get_bounds
from tqdm import tqdm


@jaxtyped(typechecker=typechecker)
def standard_affine(
    pixdim: Union[int, float], min_coords: Optional[Real[np.ndarray, "xyz=3"]] = None
) -> np.ndarray:
    """Create an affine matrix with zero rotation, shear, and translation.

    Args:
        pixdim: Pixel dimensions (voxel sizes) in mm
        shape: Image shape (number of voxels in each dimension)

    Returns:
        4x4 affine matrix with diagonal elements set to pixdim values
    """
    affine = np.eye(4)

    # Set diagonal elements to pixdim values
    for i in range(3):
        affine[i, i] = pixdim

    # Adjust the affine to position the image at the correct world coordinates
    affine[:3, 3] = min_coords

    return affine


def estimate_best_pixdim(niimgs: list[nib.Nifti1Image]) -> float:
    """Estimate the best pixel dimensions from a list of NIFTI images.

    Uses the median of the pixel dimensions over all images.

    Args:
        niimgs: List of NIFTI images

    Returns:
        List of pixel dimensions (voxel sizes) in mm
    """
    # Extract pixdim values from each image
    all_pixdims = []
    for img in niimgs:
        # Get pixdim from header (skipping the first value which is qfac)
        pixdim = img.header.get_zooms()
        all_pixdims.append(pixdim)

    # Convert to numpy array for easier manipulation
    all_pixdims = np.array(all_pixdims)

    # Take median over all dimensions
    median_pixdim = np.median(all_pixdims).item()

    return median_pixdim


def create_background_image(
    niimgs: list[nib.Nifti1Image], target_pixdim_mm: float
) -> nib.Nifti1Image:
    """Create a zero-filled background image that encompasses all input images.

    Args:
        niimgs: List of NIFTI images
        target_pixdim_mm: Target pixel dimensions (voxel sizes) in mm

    Returns:
        Zero-filled NIFTI image with standardized affine
    """
    if len(niimgs) == 0:
        raise ValueError("No images provided")

    # Get bounds for all images
    all_bounds = [get_bounds(img.shape[:3], img.affine) for img in niimgs]

    # Extract min and max values for each dimension
    # Convert list of bounds to array of shape (n_images, 3, 2) where the last dimension
    # contains (min, max) for each coordinate
    bounds_array = np.array(all_bounds)
    assert isinstance(bounds_array, Real[np.ndarray, "n_images xyz=3 min_max=2"])

    # Get global min and max across all images
    min_coords = np.min(bounds_array[:, :, 0], axis=0)
    max_coords = np.max(bounds_array[:, :, 1], axis=0)

    # Create standardized affine
    target_affine = standard_affine(pixdim=target_pixdim_mm, min_coords=min_coords)

    # Calculate required shape based on world coordinate extent and target_pixdim_mm
    world_size = max_coords - min_coords
    target_shape = np.ceil(world_size / target_pixdim_mm).astype(int)

    # Create zero-filled image
    data = np.zeros(target_shape, dtype=np.float32)

    # Create NIFTI image
    background_img = nib.Nifti1Image(data, target_affine)

    # Double-check that xyzt_units are set to mm
    if not np.allclose(background_img.header.get_zooms()[:-1], target_pixdim_mm):
        msg = f"Target pixdim {target_pixdim_mm} does not match header zooms {background_img.header.get_zooms()}"
        raise ValueError(msg)

    logger.debug(
        f"Created background image with shape {target_shape} and pixdim {target_pixdim_mm}"
    )
    logger.debug(f"World bounds: {min_coords} to {max_coords}")

    return background_img


def resample_to_standard_affine(
    niimgs: Union[nib.Nifti1Image, list[nib.Nifti1Image]],
    target_pixdim_mm: Optional[list[float]] = None,
    fill_value: Union[float, np.floating] = -np.inf,
) -> list[nib.Nifti1Image]:
    """Resample NIFTI images to a standardized affine.

    Args:
        niimgs: NIFTI image(s) to resample (file paths or Nifti1Image objects)
        target_pixdim_mm: Target pixel dimensions (voxel sizes) in mm.
            If None, will be estimated from input images.

    Returns:
        List of resampled NIFTI images
    """
    # Convert input to list of Nifti1Image objects
    if isinstance(niimgs, (str, nib.Nifti1Image)):
        niimgs = [niimgs]

    # Check for single-voxel dimensions which can cause resampling issues
    for idx, img in enumerate(niimgs):
        if any(dim == 1 for dim in img.shape[:3]):
            msg = (
                f"Image {idx} has a dimension of size 1 (shape: {img.shape[:3]}). "
                "nilearn's resampling functions don't work properly with single-voxel "
                "dimensions. Please ensure all images have dimensions > 1 in x, y, and "
                "z."
            )
            raise ValueError(msg)

    # Estimate target_pixdim_mm if not provided
    if target_pixdim_mm is None:
        target_pixdim_mm = estimate_best_pixdim(niimgs)
        logger.debug(f"Estimated target pixel dimensions: {target_pixdim_mm}")

    # Create background image that encompasses all input images
    background_img = create_background_image(niimgs, target_pixdim_mm)
    # Check if any dimension in the target shape is 1
    if any(dim == 1 for dim in background_img.shape[:3]):
        msg = (
            f"Calculated target shape {background_img.shape[:3]} has a dimension of"
            " size 1. nilearn's resampling functions don't work properly with"
            " single-voxel dimensions. Consider using a different target_pixdim_mm"
            " value or padding your input images."
        )
        raise ValueError(msg)

    # Resample each image to the background image
    resampled_imgs = []
    for img in tqdm(niimgs, desc="Resampling", unit="NIFTI"):
        resampled = resample_to_img(
            source_img=img,
            target_img=background_img,
            copy_header=True,
            fill_value=fill_value,
            force_resample=True,
        )
        # Restore previous display range
        resampled.header["cal_min"] = img.header["cal_min"]
        resampled.header["cal_max"] = img.header["cal_max"]
        resampled_imgs.append(resampled)

    return resampled_imgs


def aggregate_nifti_images(
    niimgs: list[nib.Nifti1Image],
    fill_value: Union[float, np.floating] = -np.inf,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Aggregate multiple NIFTI images with the same affine.

    For each voxel, sums the values across all images and divides by the square root
    of the number of images that have valid (non-fill) values at that voxel.

    Args:
        niimgs: List of NIFTI images with identical affines
        fill_value: Value used to indicate background/missing data

    Returns:
        tuple containing:
            - Aggregated NIFTI image
            - Overlay count NIFTI image (with intent code 1002 for Label)

    Raises:
        ValueError: If input images have different affines
    """
    if not niimgs:
        raise ValueError("No images provided")

    # Check that all affines match
    reference_niimg = niimgs[0]
    reference_affine = reference_niimg.affine
    for idx, img in enumerate(niimgs[1:], start=1):
        if not np.allclose(img.affine, reference_affine):
            msg = f"Affine of image {idx} does not match reference affine"
            raise ValueError(msg)

    # Initialize accumulator arrays
    dtype = (
        np.complex64
        if np.issubdtype(reference_niimg.dataobj.dtype, np.complexfloating)
        else np.float32
    )
    result_array = np.zeros(shape=reference_niimg.shape, dtype=dtype)
    overlay_count = np.zeros(shape=reference_niimg.shape, dtype=np.uint32)

    for img in tqdm(niimgs, desc="Aggregating", unit="NIFTI"):
        data = img.get_fdata(dtype=dtype)
        valid_mask = np.logical_and(np.isfinite(data), data != fill_value)
        result_array[valid_mask] += data[valid_mask]
        overlay_count[valid_mask] += 1
        img.uncache()

    # Normalize by sqrt(overlay_count)
    mask = overlay_count > 0
    result_array[mask] = result_array[mask] / np.sqrt(overlay_count[mask])
    # Set background voxels to fill_value
    result_array[~mask] = fill_value

    # Create output images
    aggregated_img = nib.Nifti1Image(
        result_array,
        reference_affine,
        header=niimgs[0].header.copy(),
    )

    overlay_count_img = nib.Nifti1Image(
        overlay_count,
        reference_affine,
        header=niimgs[0].header.copy(),
    )
    overlay_count_img.header.set_intent("label")

    return aggregated_img, overlay_count_img


def main(
    input_files: list[Path],
    output_dir: Path,
    target_pixdim_mm: Optional[float] = None,
    resampled_suffix: Optional[str] = "_resampled",
    aggregate_filename: Optional[Path] = Path("aggregate.nii.gz"),
    fill_value: Union[float, np.floating] = -np.inf,
):
    """Resample NIFTI images to a standardized affine.

    Args:
        input_files: List of input NIFTI file paths
        output_dir: Directory to save resampled images
        target_pixdim_mm: Target pixel dimensions (voxel sizes) in mm
        resampled_suffix: Suffix to add to output filenames
            if None, do not save resampled data
        aggregate_filename: Filename to save aggregate resampled data
            if None, do not save aggregate data
        fill_value: Value to fill background voxels with
    """
    # Function is only meaningful if we save out a file
    if resampled_suffix is None and aggregate_filename is None:
        raise ValueError("Use --resampled-suffix or --aggregate-filename")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resample images
    resampled_imgs = resample_to_standard_affine(
        niimgs=[nib.load(f) for f in input_files],
        target_pixdim_mm=target_pixdim_mm,
        fill_value=fill_value,
    )

    # Save resampled images
    for input_path, resampled_img in zip(input_files, resampled_imgs):
        # Handle .nii and .nii.gz files
        stem = input_path.stem.removesuffix(".nii")
        suffix = ".nii.gz" if input_path.suffix == ".gz" else ".nii"
        output_filename = f"{stem}{resampled_suffix}{suffix}"
        output_path = output_dir / output_filename

        # Save resampled image
        nib.save(resampled_img, output_path)
        logger.info(f"Saved resampled image to {output_path}")

    if aggregate_filename is not None:
        # can use absolute path or relative to output_dir
        aggregate_path = (
            aggregate_filename
            if aggregate_filename.is_absolute()
            else output_dir / aggregate_filename
        )
        aggregated_img, overlay_count_img = aggregate_nifti_images(
            niimgs=resampled_imgs,
            fill_value=fill_value,
        )
        nib.save(aggregated_img, aggregate_path)
        logger.info(f"Saved aggregate image to {aggregate_path}")

        # Save overlay count image
        overlay_count_path = aggregate_path.parent / (
            aggregate_path.stem.removesuffix(".nii") + "_overlay_count.nii.gz"
        )
        nib.save(overlay_count_img, overlay_count_path)
        logger.info(f"Saved overlay count image to {overlay_count_path}")


if __name__ == "__main__":
    tyro.cli(main)
