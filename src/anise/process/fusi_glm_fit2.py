from pathlib import Path
from tqdm import tqdm
import numpy as np
from shutil import copyfile

import anise.utils
import anise.io.behavior_loader

import nibabel as nib
from nilearn import plotting, image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import _gamma_difference_hrf
from nilearn.image import mean_img

import matplotlib.pyplot as plt


def rat_hrf(t_r, oversampling=50, time_length=32.0, onset=0.0):
    """Rat hemodynamic response function.

    Parameters
    ----------
    t_r : float
        Repitition time, in seconds (sampling period).

    oversampling : int, default=50
        Temporal oversampling factor.

    time_length : float, default=32
        HRF kernel length, in seconds.

    onset : float, default=0
        Onset of the response.

    Returns
    -------
    hrf : array of shape(length / t_r * oversampling, dtype=float)
         HRG sampling on the oversampled time grid.
    """
    return _gamma_difference_hrf(
        t_r,
        oversampling,
        time_length,
        onset,
        delay=3,
        undershoot=10.0,
        dispersion=0.5,
        u_dispersion=1,
        ratio=0,
    )


"""
t = np.linspace(0, 32, 3200)
fig, ax = plt.subplots()
ax.plot(t, glover_hrf(0.5), label='glover')
ax.plot(t, rat_hrf(0.5), label='rat')
ax.set_xlabel('Time (s)')
ax.set_ylabel('HRF')
ax.legend()
fig.show()
"""


def fit_glm(power_doppler_img, time_stamps, transformations,
            event, events, hrf='rat'):
    """Compute a GLM on power doppler data.

    Parameters
    ----------
    power_doppler_img : nib.Nifti1Image
        Power Doppler data.
    time_stamps : np.ndarray
        The time stamps for the power doppler images.
    transformations : np.ndarray
        The transformations from the registration
    event : str
        The name of the event
    events : pd.DataFrame
        The event data.

    Returns
    -------
    zmap : nib.Nifti1Image
        The T-statistic map from the GLM.
    """
    fmri_glm = FirstLevelModel(minimize_memory=False, mask_img=False,
                               smoothing_fwhm=None, standardize=True)
    design_matrix = make_first_level_design_matrix(
        time_stamps,
        events,
        drift_model="polynomial",
        drift_order=1,
        add_regs=transformations.T,
        add_reg_names=["tx", "ty", "rot"],
        hrf_model=(rat_hrf if hrf == 'rat' else hrf),
    )

    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }

    fmri_glm = fmri_glm.fit(power_doppler_img, design_matrices=design_matrix)
    
    # if a different hrf is used annoyingly that changes the
    # name of the design matrix column, use indexing but check
    # that it is correct
    event_cols = [col for col in design_matrix.columns if col.startswith(event)]
    assert len(event_cols) == 1
    event_col = event_cols[0]

    zmap = fmri_glm.compute_contrast(
        basic_contrasts[event_col],
        output_type="stat"
    )
    return zmap


def fit_glm_time_shift(power_doppler_img, time_stamps, transformations,
                       event, events, out_dir, hrf='rat',
                       shift=12, shift_res=0.25, n_best_vox=20):
    """Compute a GLM on power doppler data checking for the best time shift.

    Parameters
    ----------
    power_doppler_img : nib.Nifti1Image
        Power Doppler data.
    time_stamps : np.ndarray
        The time stamps for the power doppler images.
    transformations : np.ndarray
        The transformations from the registration
    event : str
        The name of the event
    events : pd.DataFrame
        The event data.
    out_dir : pathlib.Path
        The output directory.
    hrf : str
        The hemodynamic response function to use.
    shift : float
        How far to check the events being shifted.
    shift_res : float
        The resolution of shifts to check.
    n_best_vox : int
        The number of best voxels to use for considering
        in order to determine the best time shift.

    Returns
    -------
    event_time_offsets : np.ndarray
        All the time offsets tried.
    max_zmaps: np.ndarray
        The maximum z-score per offset.
    best_offset : float
        The best offset as found by averaging intensity time courses.
    """
    (out_dir / 'time_shift').mkdir(parents=True, exist_ok=True)

    max_zmaps = list()
    zmaps = list()
    event_time_offsets = np.arange(-shift, shift + shift_res, shift_res).round(2)
    for event_time_offset in tqdm(event_time_offsets):
        events_shifted = events.copy()
        events_shifted['onset'] += event_time_offset
        zmap = fit_glm(power_doppler_img, time_stamps, transformations,
                       event, events_shifted, hrf=hrf)
        nib.save(zmap, (out_dir / 'time_shift' /
                        f"eto-{event_time_offset}_zmap.nii.gz"))
        zmaps.append(zmap)
        max_zmap = np.max(zmap.get_fdata())
        max_zmaps.append(max_zmap)

        mean_image = mean_img(power_doppler_img)
        display = plotting.plot_stat_map(
            zmap,
            bg_img=mean_image,
            cut_coords=[0],
            threshold=3,
            display_mode="y",
            black_bg=True,
            title=f"pwm_enabled contrast\n(shifted {event_time_offset.round(2)})",
        )
        display.savefig(out_dir / 'time_shift' /
                        f"eto-{event_time_offset}_zmap.png")
        display.close()

    fig, ax = plt.subplots()
    ax.plot(event_time_offsets, max_zmaps)
    ax.set_xlabel('Event Time Offset')
    ax.set_ylabel('Max T-Statistic')
    fig.savefig(out_dir / "zmaptrend.png")
    plt.close(fig)

    # find best voxels
    zmap_data = np.array([np.array(zmap.dataobj).ravel() for zmap in zmaps])
    zmap_best_idxs = np.argsort(zmap_data.max(axis=0))[-n_best_vox:]
    zmean = zmap_data[:, zmap_best_idxs].mean(axis=1)
    zstd = zmap_data[:, zmap_best_idxs].std(axis=1)

    best_idx = np.argmax(np.abs(zmean))
    best_offset = event_time_offsets[best_idx]
    best_zmap = zmaps[best_idx]

    # plot intensity values over time shifts
    fig, ax = plt.subplots()
    ax.plot(event_time_offsets, zmap_data[:, zmap_best_idxs])
    ax.set_xlabel('Time Shift (s)')
    ax.set_ylabel('T-Statistic')
    fig.savefig(out_dir / 'best_voxel_time_shifts.png')

    copyfile(
        out_dir / 'time_shift' / f"eto-{best_offset}_zmap.png",
        out_dir / "best_zmap.png"
    )
    nib.save(best_zmap, out_dir / "best_zmap.nii.gz")

    return event_time_offsets, max_zmaps, best_offset
