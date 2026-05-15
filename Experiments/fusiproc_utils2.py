# ========================================================================
# fusiproc_utils.py      –‑ helper library for functional‑US processing
# ========================================================================
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.signal import clean
from nipype.algorithms.confounds import compute_dvars, TCompCor
import SimpleITK as sitk
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# ------------------------------------------------------------------------
# 0  General I/O
# ------------------------------------------------------------------------
def load_fusi_nifti(pwd_path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    nii = nib.load(pwd_path)
    return nii.get_fdata(), nii.affine, nii


def save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path,
               dtype = np.float32) -> Path:
    img = nib.Nifti1Image(arr.astype(dtype), affine)
    img.set_data_dtype(dtype)
    nib.save(img, path)
    return path


# ------------------------------------------------------------------------
# 1  Quick‑QC helpers
# ------------------------------------------------------------------------
def nan_mask(data: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(data), axis=-1)


def global_signal(data: np.ndarray) -> np.ndarray:
    return np.nanmean(data, axis=(0, 1, 2))


def plot_global_signal(sig: np.ndarray, out_png: Optional[Path] = None, y_lim: Optional[Tuple[float, float]] = None) -> None:
    fig, ax = plt.subplots()
    ax.plot(sig)
    ax.set_title("Global signal")
    ax.set_xlabel("Frame #")
    ax.set_ylabel("Average Intensity")
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()


def repetition_time(json_path: Path) -> float:
    with open(json_path) as f:
        frame_times = np.array(json.load(f)["VolumeTiming"])
    return float(np.median(np.diff(frame_times)))

def probe_temperature(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        hdr = json.load(f)

    # Accept a few alternative spellings in case the scanner software changes
    for key in ("ProbeTemperature", "InterposerTemperature",
                "probe_temperature", "interposer_temperature"):
        if key in hdr:
            return np.asarray(hdr[key], dtype=float)

    raise KeyError(
        f"No temperature array found in {json_path.name}. "
        "Looked for keys: ProbeTemperature / InterposerTemperature."
    )


def detect_motion_via_global(
        sig: np.ndarray, t_r: float, k: float = 3.0, y_lim: Optional[Tuple[float, float]] = None
) -> tuple[np.ndarray, np.ndarray, plt.Figure]:
    sig_detr = clean(sig.reshape(-1, 1), detrend=True, t_r=t_r).ravel()
    med = np.median(sig_detr)
    mad = np.median(np.abs(sig_detr - med))
    thr = med + k * mad
    bad = np.where(sig_detr > thr)[0]
    good = np.array([i for i in range(sig.size) if i not in bad], dtype=np.int64)

    fig, ax = plt.subplots()
    ax.plot(sig)
    for f in bad:
        ax.axvline(f, ls='--', c='r', alpha=.3)
    ax.set_title("Global signal with motion outliers")
    ax.set_xlabel("Frame #")
    ax.set_ylabel("Average Intensity")
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    return bad, good, fig


# ------------------------------------------------------------------------
# 2  Rigid registration   
# ------------------------------------------------------------------------
def register_3d_image_stack(
    image_stack: np.ndarray,
    reference: Union[int, str, None] = None,
    *,
    spacing: Optional[Sequence[float]] = None,
    skip_indices: Optional[Sequence[int]] = None,
    mask: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Rigidly register a 4‑D stack `(x, y, z, t)` to a reference volume.
    """
    # ------------------------------------------------------------------
    # 0. sanity checks --------------------------------------------------
    # ------------------------------------------------------------------
    if image_stack.ndim != 4:
        raise ValueError("image_stack must have shape (x, y, z, t)")
    num_frames = image_stack.shape[3]

    skip_set: set[int] = set(skip_indices or [])
    if any(idx < 0 or idx >= num_frames for idx in skip_set):
        raise ValueError("skip_indices contain out‑of‑range values")
    if len(skip_set) != len(skip_indices or []):
        raise ValueError("skip_indices must be unique")
    retained = [i for i in range(num_frames) if i not in skip_set]
    if not retained:
        raise ValueError("No valid frames left after applying skip_indices.")

    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != image_stack.shape[:3]:
            raise ValueError("mask must have same (x, y, z) shape as image_stack")

    # ------------------------------------------------------------------
    # 1. reference specification ---------------------------------------
    # ------------------------------------------------------------------
    ref_mode: str
    if reference is None:
        ref_idx = retained[len(retained) // 2]
        ref_mode = "index"
    elif isinstance(reference, int):
        ref_idx = reference
        ref_mode = "index"
    elif isinstance(reference, str):
        reference = reference.lower()
        if reference not in {"mean", "median"}:
            raise ValueError("reference must be 'mean' or 'median'")
        ref_idx = None
        ref_mode = reference
    else:
        raise TypeError("reference must be int, str, or None")

    if ref_mode == "index" and (ref_idx < 0 or ref_idx >= num_frames or ref_idx in skip_set):
        raise ValueError("Invalid reference frame index")

    # ------------------------------------------------------------------
    # 2. voxel spacing --------------------------------------------------
    # ------------------------------------------------------------------
    spacing_xyz = np.asarray(spacing if spacing is not None else (1., 1., 1.), float)
    if spacing_xyz.shape != (3,):
        raise ValueError("spacing must be an iterable of length 3 (x, y, z)")

    # ------------------------------------------------------------------
    # 3. build fixed image ---------------------------------------------
    # ------------------------------------------------------------------
    geom_source = sitk.GetImageFromArray(
        image_stack[..., retained[0]].astype(np.float32)
    )
    geom_source.SetSpacing(tuple(spacing_xyz))

    if ref_mode == "mean":
        fixed_image = sitk.GetImageFromArray(
            np.nanmean(image_stack[..., retained], axis=-1).astype(np.float32)
        )
    elif ref_mode == "median":
        fixed_image = sitk.GetImageFromArray(
            np.nanmedian(image_stack[..., retained], axis=-1).astype(np.float32)
        )
    else:
        fixed_image = sitk.GetImageFromArray(
            image_stack[..., ref_idx].astype(np.float32)
        )
    fixed_image.CopyInformation(geom_source)

    if mask is not None:
        mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_img.CopyInformation(geom_source)
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)

    # ------------------------------------------------------------------
    # 4. configure registration ----------------------------------------
    # ------------------------------------------------------------------
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1,
        minStep=1e-4,
        numberOfIterations=100,
        relaxationFactor=0.5,
    )
    if mask is not None:
        registration.SetMetricFixedMask(mask_img)

    # ------------------------------------------------------------------
    # 5. iterate through frames ----------------------------------------
    # ------------------------------------------------------------------
    kept_frames: List[np.ndarray] = []
    params: List[np.ndarray] = []
    metric_vals: List[float] = []
    stop_strings: List[str] = []

    for t in range(num_frames):
        if t in skip_set:
            continue
        if ref_mode == "index" and t == ref_idx:
            kept_frames.append(sitk.GetArrayFromImage(fixed_image))
            params.append(np.zeros(6, dtype=np.float32))
            metric_vals.append(np.nan)
            stop_strings.append("")
            if verbose:
                print(f"[frame {t:>4}] reference frame (copied as‑is)")
            continue

        moving = sitk.GetImageFromArray(image_stack[..., t].astype(np.float32))
        moving.CopyInformation(geom_source)

        init_tx = sitk.CenteredTransformInitializer(
            fixed_image, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        registration.SetInitialTransform(init_tx, inPlace=True)
        registration.SetOptimizerScalesFromPhysicalShift()
        final_tx = registration.Execute(fixed_image, moving)

        metric_vals.append(registration.GetMetricValue())
        stop_strings.append(registration.GetOptimizerStopConditionDescription())
        if verbose:
            print(f"[frame {t:>4}] metric = {metric_vals[-1]: .6e} "
                  f"| stop: {stop_strings[-1]}")

        params.append(
            np.array([*final_tx.GetTranslation(),
                      final_tx.GetAngleX(), final_tx.GetAngleY(), final_tx.GetAngleZ()],
                     dtype=np.float32)
        )

        corrected = sitk.Resample(
            moving, fixed_image, final_tx,
            sitk.sitkLinear, 0.0, moving.GetPixelID(),
        )
        kept_frames.append(sitk.GetArrayFromImage(corrected))
    retained = [i for i in range(num_frames) if i not in skip_set]
    registered_array  = np.stack(kept_frames, axis=-1)
    transform_params  = np.stack(params,  axis=1)
    metric_values     = np.asarray(metric_vals, dtype=np.float64)
    return registered_array, transform_params, metric_values, stop_strings, retained



# ------------------------------------------------------------------------
# 3  Motion & DVARS
# ------------------------------------------------------------------------
def framewise_displacement(rigid_params: np.ndarray,
                           radius_mm: float = 50.0) -> np.ndarray:
    diff = np.diff(rigid_params.astype(float), axis=1)
    diff[3:6] *= radius_mm
    return np.insert(np.abs(diff).sum(axis=0), 0, 0.0)


def calc_dvars(reg_path: Path, mask_path: Path
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return compute_dvars(
        in_file=str(reg_path), in_mask=str(mask_path),
        remove_zerovariance=True, intensity_normalization=0)


# ------------------------------------------------------------------------
# 4  tCompCor wrapper
# ------------------------------------------------------------------------
def run_tcompcor(realigned_path: Path, mask_path: Path, out_dir: Path, *,
                 n_components: int = 6, percentile_threshold: float = 0.02,
                 tr: float | None = None, ignore_n_vols: int = 0
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = TCompCor()
    tc.inputs.realigned_file = realigned_path.as_posix()
    tc.inputs.mask_files = mask_path.as_posix()
    tc.inputs.num_components = n_components
    tc.inputs.percentile_threshold = percentile_threshold
    if tr is not None:
        tc.inputs.repetition_time = float(tr)
    if ignore_n_vols:
        tc.inputs.ignore_initial_volumes = int(ignore_n_vols)
    tc.inputs.components_file = (out_dir/"tcompcor_components.tsv").as_posix()
    tc.inputs.save_metadata  = (out_dir/"tcompcor_metadata.tsv").as_posix()
    res = tc.run()
    return {"components": Path(res.outputs.components_file),
            "variance_mask": Path(res.outputs.high_variance_masks[0])}


# ------------------------------------------------------------------------
# 5  Temporal smoothing
# ------------------------------------------------------------------------
def boxcar_smooth(data: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        return data
    half = window // 2
    sm = np.empty_like(data)
    for t in range(data.shape[3]):
        lo, hi = max(0, t-half), min(data.shape[3], t+half+1)
        sm[..., t] = np.nanmean(data[..., lo:hi], axis=3)
    return sm


from scipy.signal import savgol_filter


def _interp_nans(ts: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs; returns *new* array so caller can keep original."""
    if np.isfinite(ts).all():
        return ts.copy()
    filled = ts.copy()
    valid  = np.isfinite(ts)
    filled[~valid] = np.interp(t_idx[~valid], t_idx[valid], ts[valid])
    return filled


def boxcar_smooth(data: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        return data
    half = window // 2
    out  = np.empty_like(data)
    for t in range(data.shape[3]):
        lo, hi = max(0, t-half), min(data.shape[3], t+half+1)
        out[..., t] = np.nanmean(data[..., lo:hi], axis=3)
    return out


def savgol_smooth(data: np.ndarray, *, window: int, polyorder: int = 2
                  ) -> np.ndarray:
    if window < 3 or window % 2 == 0 or polyorder >= window:
        raise ValueError("window must be odd ≥3 and > polyorder")
    *spatial, T = data.shape
    flat   = data.reshape(-1, T)
    t_idx  = np.arange(T)

    for i, ts in enumerate(flat):
        valid = np.isfinite(ts)
        if valid.sum() < polyorder + 2:        # not enough data
            continue                           # leave as‑is
        ts_f  = _interp_nans(ts, t_idx)
        filt  = savgol_filter(ts_f, window, polyorder,
                              mode="interp", axis=-1)
        filt[~valid] = np.nan
        flat[i] = filt

    return flat.reshape(*spatial, T)



# ------------------------------------------------------------------------
# 6  NaN‑padding after registration
# ------------------------------------------------------------------------
def reinstate_nans(reg_data: np.ndarray, rigid_params: np.ndarray,
                   reg_metrics: np.ndarray, reg_stop: list[str],
                   n_frames: int, retained: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    reg_full = np.full((*reg_data.shape[:3], n_frames), np.nan, dtype=reg_data.dtype)
    reg_full[..., retained] = reg_data
    rigid_full = np.full((6, n_frames), np.nan, dtype=rigid_params.dtype)
    rigid_full[:, retained] = rigid_params
    metric_full = np.full(n_frames, np.nan, dtype=reg_metrics.dtype)
    metric_full[retained] = reg_metrics
    stop_full = [""] * n_frames
    for i_good, t in enumerate(retained):
        stop_full[t] = reg_stop[i_good]
    return reg_full, rigid_full, metric_full, stop_full


# ------------------------------------------------------------------------
# 7  QC figures
# ------------------------------------------------------------------------
def plot_registration_parameters(rigid_full: np.ndarray, out_png: Path) -> None:
    n_frames = rigid_full.shape[1]
    x = np.arange(n_frames)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    ax_t = axes[0]; ax_r = axes[1]
    ax_t.plot(x, rigid_full[0], label="Translation X")
    ax_t.plot(x, rigid_full[1], label="Translation Y")
    ax_t.plot(x, rigid_full[2], label="Translation Z")
    ax_t.set_ylabel("Translation (mm)")
    ax_t.legend(); ax_t.grid(alpha=.3)
    ax_r.plot(x, rigid_full[3]*180/np.pi, label="Rotation X")
    ax_r.plot(x, rigid_full[4]*180/np.pi, label="Rotation Y")
    ax_r.plot(x, rigid_full[5]*180/np.pi, label="Rotation Z")
    ax_r.set_xlabel("Frame index"); ax_r.set_ylabel("Rotation (°)")
    ax_r.legend(); ax_r.grid(alpha=.3)
    fig.savefig(out_png, dpi=300)
    plt.show()


def plot_fd(fd_full: np.ndarray, threshold: float, out_png: Path
            ) -> np.ndarray:
    x = np.arange(fd_full.size)
    fig, ax = plt.subplots(figsize=(8,4), layout='constrained')
    ax.plot(x, fd_full, label="Framewise Displacement")
    ax.axhline(threshold, ls="--", label=f"{threshold} mm threshold")
    ax.set_xlabel("Frame index"); ax.set_ylabel("FD (mm)")
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out_png, dpi=300)
    plt.show()
    return np.where((fd_full > threshold) & np.isfinite(fd_full))[0]


def plot_dvars(dvars_stdz: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(dvars_stdz, label='Standardised DVARS')
    ax.set_xlabel("Frame #"); ax.set_ylabel("DVARS")
    ax.set_title("DVARS – rigid‑registered frames")
    ax.legend()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------------
# 8  Doppler movie
# ------------------------------------------------------------------------
def save_doppler_movie(reg_data: np.ndarray, out_mp4: Path,
                       vmax_plus: float = 7.0, step: int = 2
) -> HTML:
    data_db = 10*np.log10(reg_data)
    nnz = reg_data > np.finfo(reg_data.dtype).eps
    vmax = np.percentile(data_db[nnz], 90)
    vmin = np.percentile(data_db[nnz], 10)
    fig, ax = plt.subplots()
    slice_ = data_db[:, data_db.shape[1]//2, :, 0].T
    cax = ax.imshow(slice_, cmap="inferno", vmax=vmax+vmax_plus,
                    vmin=vmin, extent=[0,50,92,25])
    ax.set_xlabel("Y (mm)"); ax.set_ylabel("X (mm)")
    ax.set_title("0"); plt.colorbar(cax, ax=ax, label="Intensity (dB)")

    def update(frame):
        cax.set_data(data_db[:, data_db.shape[1]//2, :, frame].T)
        ax.set_title(f"Frame # {frame}")
        return (cax,)

    ani = FuncAnimation(fig, update,
                        frames=range(0, reg_data.shape[3], step),
                        interval=100, blit=True)
    ani.save(out_mp4, writer="ffmpeg")
    plt.close()
    return HTML(ani.to_jshtml())


# ------------------------------------------------------------------------
# 9  GLM helpers
# ------------------------------------------------------------------------
def build_design_matrix(glm, events, confounds, sample_mask, data):
    fitted = glm.fit(data, events=events,
                     confounds=confounds,
                     sample_masks=[sample_mask])
    return fitted, fitted.design_matrices_[0]


def localizer_contrasts(dm) -> dict[str, np.ndarray]:
    eye = np.eye(dm.shape[1])
    con = {c: eye[i] for i, c in enumerate(dm.columns)}
    # con["any_face minus closed"] = (con["face_fear"]
    #                                 + con["face_happy"]
    #                                 - con["eyes_closed"])
    # con["any_face"] = con["face_fear"] + con["face_happy"]
    return con
