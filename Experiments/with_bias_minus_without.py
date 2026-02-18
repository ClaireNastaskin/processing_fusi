######################
# Experiment: With bias minus without bias
######################

from pathlib import Path
import nibabel as nib

svd_value = 50
stim = "Right foot"
base_path = Path("/Users/clairenastaskin/data/Post_acquisition_parameter_sweep/sub-Forest001/ses-20250708/task-movements_run-004/")
Activation_map_with_bias_path = base_path / f"power_doppler_volumes_200bmodes_1.6s/svd_{svd_value}/{stim}_boxcar3_z_map.nii.gz"
Activation_map_without_bias_path = base_path / f"power_doppler_volumes_without_bias_200bmodes_1.6s/svd_{svd_value}/{stim}_boxcar3_z_map.nii.gz"

with_bias_img = nib.load(Activation_map_with_bias_path)
without_bias_img = nib.load(Activation_map_without_bias_path)

Difference_map = with_bias_img.get_fdata() - without_bias_img.get_fdata()

# save Difference_map
Difference_map_path = base_path / f"{stim}_fingers_svd_{svd_value}_difference_map_with_without_bias.nii.gz"
Difference_map_img = nib.Nifti1Image(Difference_map, with_bias_img.affine, with_bias_img.header)
nib.save(Difference_map_img, Difference_map_path)
