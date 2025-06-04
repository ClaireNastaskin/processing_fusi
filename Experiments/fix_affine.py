import nibabel as nib
import numpy as np
from pathlib import Path

base_path = Path('/Users/clairenastaskin/data/2025-05-07/sub-Forest001_ses-20250507_task-movements_run-001')  
pwd_file_name = 'sub-Forest001_ses-20250507_task-movements_run-001_pwd.nii.gz'
lmbda = 0.513 # in mm
pwd_path = base_path / pwd_file_name
# Load fUSI nifti file and get data into a numpy array
nifti_img = nib.load(pwd_path)
fusi_data = nifti_img.get_fdata()
fusi_data = np.nan_to_num(fusi_data, nan=0) # no nan
affine = np.array([
    [lmbda, 0, 0, -fusi_data.shape[0]/2*lmbda], # in pixels
    [0, -lmbda, 0, fusi_data.shape[1]/2*lmbda],
    [0, 0, -lmbda, -25],
    [0, 0, 0, 1]
])
nifti_img_fixed = nib.Nifti1Image(fusi_data, affine)
nifti_img_fixed.to_filename(base_path / 'sub-Forest001_ses-20250507_task-movements_run-001_pwd_fixed.nii.gz')