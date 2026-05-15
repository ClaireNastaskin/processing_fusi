import zarr
import nibabel as nib
import numpy as np
from pathlib import Path


def convert_zarr_to_niigz(zarr_path, output_path=None):
    """Convert a 4D zarr array of power doppler volumes to a NIfTI .nii.gz file."""
    zarr_path = Path(zarr_path)
    z = zarr.open(str(zarr_path), mode="r")
    print(f"Zarr shape: {z.shape}, dtype: {z.dtype}")

    # Load all volumes and move the volumes axis (first) to last for NIfTI convention
    data = np.array(z)
    data = np.moveaxis(data, 0, -1)
    print(f"Output array shape: {data.shape}")

    # Compute global signal (mean across spatial dims) per frame
    global_signal = np.nanmean(data, axis=(0, 1, 2))
    nonzero_mask = global_signal != 0
    n_removed = np.sum(~nonzero_mask)
    print(f"Removing {n_removed} frames with global signal == 0 (keeping {np.sum(nonzero_mask)} of {len(global_signal)})")
    data = data[..., nonzero_mask]

    # Create a NIfTI image with an identity affine
    img = nib.Nifti1Image(data, affine=np.eye(4))

    if output_path is None:
        output_path = zarr_path.parent / (zarr_path.stem + ".nii.gz")
    output_path = Path(output_path)

    nib.save(img, str(output_path))
    print(f"Saved NIfTI to {output_path}")


if __name__ == "__main__":
    # --- Set your path here ---
    zarr_path = "/Users/clairenastaskin/data/sub-Forest001/2026-02-26/EV-069a1020-d08e-7e5c-8000-c56462d87147/sliding_window_0.8s/power_doppler.zarr"

    convert_zarr_to_niigz(zarr_path)
