import os
import nibabel as nib
import numpy as np

# Path to your folder
folder = "/Users/clairenastaskin/data/2025-07-08/sub-Forest001_ses-20250708_task-sensory_run-004/realigned_volumes/"

# Loop through files in the folder
for fname in os.listdir(folder):
    if fname.endswith("z_map_realigned.nii.gz"):
        filepath = os.path.join(folder, fname)

        # Load the NIfTI file
        img = nib.load(filepath)
        data = img.get_fdata()
        affine = img.affine

        # Multiply data by 10
        new_data = data * 10

        # Create new NIfTI image
        new_img = nib.Nifti1Image(new_data, affine)

        # Create new filename
        new_fname = fname.replace("z_map_realigned.nii.gz", "z_map_realigned10.nii.gz")
        new_path = os.path.join(folder, new_fname)

        # Save the modified image
        nib.save(new_img, new_path)
        print(f"Saved: {new_path}")
