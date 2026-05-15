import zarr, nibabel as nib, numpy as np
z = zarr.open('/Users/clairenastaskin/data/sub-Forest001/2026-02-26/EV-069a1020-d08e-7e5c-8000-c56462d87147/sliding_window_0.8s/power_doppler.zarr', mode='r')
ref = nib.load('/Users/clairenastaskin/data/sub-Forest001/2026-02-26/EV-069a1020-d08e-7e5c-8000-c56462d87147/sliding_window_0.8s/pwd.nii')
vol = np.array(z[0])
nib.save(nib.Nifti1Image(vol, ref.affine), 'Users/clairenastaskin/data/sub-Forest001/2026-02-26/EV-069a1020-d08e-7e5c-8000-c56462d87147/sliding_window_0.8s')

