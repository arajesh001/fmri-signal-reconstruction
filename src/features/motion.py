"""
motion.py

Motion features from motion.1D: raw params + first derivatives + FD,
interpolated onto the cardiac waveform's time base.

motion.1D column layout (confirmed via AFNI 3dvolreg -1Dfile):

    n  roll  pitch  yaw  dS  dL  dP  rmsold  rmsnew

    n      = sub-brick / TR index (not a motion feature)
    roll   = rotation about I-S axis, degrees
    pitch  = rotation about R-L axis, degrees
    yaw    = rotation about A-P axis, degrees
    dS     = displacement, Superior direction, mm
    dL     = displacement, Left direction, mm
    dP     = displacement, Posterior direction, mm
    rmsold = RMS diff, input brick vs base brick (registration QC,
             not a motion feature)
    rmsnew = RMS diff, output brick vs base brick (registration QC,
             not a motion feature)
"""

import json
import warnings

import numpy as np

from src.dataset.scan import Scan

MOTION_COLUMNS = [
    "n", "roll", "pitch", "yaw", "dS", "dL", "dP", "rmsold", "rmsnew",
]

ROTATION_COLUMNS = ["roll", "pitch", "yaw"]
TRANSLATION_COLUMNS = ["dS", "dL", "dP"]

# Order used everywhere in this file that stacks the 6 motion params
# into one array
MOTION_PARAM_COLUMNS = ROTATION_COLUMNS + TRANSLATION_COLUMNS

# Standard Power
HEAD_RADIUS_MM = 50.0

DEFAULT_TR = 1.5

# Final channel order out of build_features: 6 raw params, 6
# derivatives, 1 FD.
CHANNEL_NAMES = (
    MOTION_PARAM_COLUMNS
    + [f"d_{c}" for c in MOTION_PARAM_COLUMNS]
    + ["fd"]
)


def _get_tr(scan: Scan) -> float:
    """
    TR in seconds for this scan.

    Design choice: prefers reading RepetitionTime out of bold_json
    when a local copy exists, falls back to DEFAULT_TR otherwise.
    """

    if scan.bold_json is not None and scan.bold_json.exists():
        with open(scan.bold_json) as f:
            metadata = json.load(f)
        return metadata["RepetitionTime"]

    return DEFAULT_TR


# ==========================================================
# LOAD
# ==========================================================

def load_motion(scan: Scan) -> dict:
    """
    Load and parse motion.1D for one scan.

    Returns

    dict mapping each name in MOTION_COLUMNS to a 1D np.ndarray of
    length n_TRs.
    """

    data = np.loadtxt(scan.motion)
    return {name: data[:, i] for i, name in enumerate(MOTION_COLUMNS)}


# ==========================================================
# DERIVED FEATURES
# ==========================================================

def compute_derivatives(motion: dict) -> np.ndarray:
    """
    First frame-to-frame differences of roll/pitch/yaw/dS/dL/dP.

    Returns
    np.ndarray, shape (n_TRs, 6), columns in MOTION_PARAM_COLUMNS order.

    """

    params = np.column_stack([motion[c] for c in MOTION_PARAM_COLUMNS])
    return np.diff(params, axis=0, prepend=params[0:1])


def compute_fd(motion: dict) -> np.ndarray:
    """
    Framewise displacement

    Returns
    np.ndarray, shape (n_TRs,)
    """

    derivatives = compute_derivatives(motion)
    n_rot = len(ROTATION_COLUMNS)

    rotation_deg = derivatives[:, :n_rot]
    translation_mm = derivatives[:, n_rot:]

    # arc length: mm = radians * head radius
    rotation_mm = np.deg2rad(rotation_deg) * HEAD_RADIUS_MM

    return np.sum(np.abs(np.hstack([rotation_mm, translation_mm])), axis=1)


# ==========================================================
# RESAMPLING
# ==========================================================

def interpolate_to_rate(
    motion_features: np.ndarray,
    tr: float,
    target_rate: float,
    n_samples: int,
) -> np.ndarray:
    """
    Upsample motion features (n_TRs, n_channels) from TR resolution
    onto the cardiac waveform's time base.

    Returns
    np.ndarray, shape (n_samples, n_channels)
    """

    n_trs = motion_features.shape[0]
    t_source = np.arange(n_trs) * tr
    t_target = np.arange(n_samples) / target_rate

    # safety buffer
    gap = t_target[-1] - t_source[-1]
    if gap > 1.0:
        warnings.warn(
            f"target time axis extends {gap:.1f}s past motion.1D's "
            f"coverage ({t_source[-1]:.1f}s) -- the tail of this scan's "
            f"motion features will be constant-extrapolated from the "
            f"last known frame."
        )

    return np.column_stack([
        np.interp(t_target, t_source, motion_features[:, i])
        for i in range(motion_features.shape[1])
    ])


# ==========================================================
# PUBLIC API
# ==========================================================

def build_features(scan: Scan, target_rate: float, n_samples: int) -> np.ndarray:
    """
    Load motion.1D, compute derivatives + FD, interpolate everything
    onto (target_rate, n_samples).

    Parameters
    target_rate, n_samples : the exact time axis to align to.

    Returns
    np.ndarray, shape (n_samples, 13). Channel order is CHANNEL_NAMES
    (6 raw params, 6 derivatives, 1 FD).
    """

    motion = load_motion(scan)
    tr = _get_tr(scan)

    params = np.column_stack([motion[c] for c in MOTION_PARAM_COLUMNS])
    derivatives = compute_derivatives(motion)
    fd = compute_fd(motion)

    combined = np.column_stack([params, derivatives, fd])

    return interpolate_to_rate(combined, tr, target_rate, n_samples)
