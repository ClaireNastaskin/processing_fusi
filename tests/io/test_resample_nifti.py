"""Integration tests for the resample_nifti module."""

import nibabel as nib
import numpy as np
import pytest

from anise.io import resample_nifti
from anise.io.download import cached_download

# Define constants for test data
TEST_DATA_URLS = {
    "t1": "https://github.com/neurolabusc/niivue-images/raw/refs/heads/main/chris_t1.nii.gz",
    "t2": "https://github.com/neurolabusc/niivue-images/raw/refs/heads/main/chris_t2.nii.gz",
}


@pytest.fixture(scope="session")
def downloaded_nifti_files():
    """Download and cache test NIfTI files.

    Returns:
        dict: Dictionary mapping file names to file paths
    """
    cached_files = {}

    for name, url in TEST_DATA_URLS.items():
        # Download and cache the file
        filepath = cached_download(
            url=url,
            timeout=60,  # Longer timeout for larger files
            project_name="anise-tests",
        )

        cached_files[name] = filepath

    return cached_files


@pytest.fixture(scope="session")
def nifti_images(downloaded_nifti_files) -> list[nib.Nifti1Image]:
    """Load test NIfTI files as nibabel images.

    Returns:
        dict: Dictionary mapping file names to nibabel images
    """
    return [nib.load(path) for path in downloaded_nifti_files.values()]


@pytest.mark.network
def test_resample_to_standard_affine_with_real_data(
    nifti_images: list[nib.Nifti1Image],
):
    """Test resampling with real neuroimaging data from the web."""
    # Resample images
    target_pixdim = 2.0  # 2mm isotropic
    resampled_imgs = resample_nifti.resample_to_standard_affine(
        niimgs=nifti_images,
        target_pixdim_mm=target_pixdim,
        fill_value=-100,
    )

    # Verify we got the expected number of images back
    assert len(resampled_imgs) == 2

    # Check that all resampled images have the same shape and affine
    first_shape = resampled_imgs[0].shape[:3]
    first_affine = resampled_imgs[0].affine

    for img in resampled_imgs:
        assert img.shape[:3] == first_shape
        assert np.allclose(img.affine, first_affine)

    # Check that the pixel dimensions are as expected
    for img in resampled_imgs:
        pixdims = img.header.get_zooms()[:3]
        assert np.allclose(pixdims, target_pixdim)

    # Verify that the data is not all zeros or fill values
    for img in resampled_imgs:
        data = img.get_fdata()
        assert np.any(data != -100), "Data should not be all fill values"
        assert np.any(data != 0), "Data should not be all zeros"


@pytest.mark.network
def test_main_function_integration(downloaded_nifti_files, tmp_path):
    """Test the main function with real data."""
    # Setup output directory
    output_dir = tmp_path / "resampled"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the main function
    input_nifti_files = list(downloaded_nifti_files.values())
    resample_nifti.main(
        input_files=input_nifti_files,
        output_dir=output_dir,
        target_pixdim_mm=1.5,
        fill_value=-100,
        output_suffix="_std",
    )

    # Check that output files were created
    output_files = list(output_dir.glob("*.nii.*"))
    assert len(output_files) == len(input_nifti_files)
