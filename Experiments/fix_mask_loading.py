from pathlib import Path
import nibabel as nib

# Your run01 path
run01 = Path("/Users/clairenastaskin/data/2025-07-08/sub-Forest001_ses-20250708_task-movements_run-004")

# Find the mask file that ends with "mask.nii.gz"
mask_files = list(run01.glob("*mask.nii.gz"))

if not mask_files:
    print(f"No mask files found in {run01}")
    print("Available files in directory:")
    for file in run01.iterdir():
        print(f"  {file.name}")
    raise FileNotFoundError(f"No mask files found in {run01}")

if len(mask_files) > 1:
    print(f"Multiple mask files found: {mask_files}")
    print("Using the first one. You can modify this to choose a specific file.")
    mask_file = mask_files[0]
else:
    mask_file = mask_files[0]

print(f"Loading mask file: {mask_file}")
nifti_mask = nib.load(mask_file)
mask = nifti_mask.get_fdata()
print('mask shape:', mask.shape) 