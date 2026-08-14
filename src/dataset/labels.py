"""
labels.py

Loads the happy cardiac waveform and the hr.txt ground truth for one
scan, and aligns them onto a common time base.
"""

import json

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from config import SAMPLING_RATE
from .scan import Scan

# ==========================================================
# CONFIG 
# ==========================================================

# Which TSV column holds the cardiac waveform, per happy resolution.
# Only "stdres" has actually been used/tested end to end so far.

HAPPY_WAVEFORM_COLUMN = {
    "cardpulse": "pulsefromfmri",
    "sliceres": "cardiacfromfmri",
    "stdres": "cardiacfromfmri_25.0Hz",
}

# TBD
GROUND_TRUTH_COLUMN = 5

# Bandpass band applied to the ground truth column before use
GROUND_TRUTH_BANDPASS_HZ = (0.8, 3.0)


# ==========================================================
# HAPPY WAVEFORM
# ==========================================================

def load_happy_waveform(scan: Scan, resolution: str = "stdres") -> dict:
    """
    Load the happy cardiac waveform for one scan.

    Returns
    
    {"Waveform": np.ndarray, "Sampling Rate": float, "Start Time": float}
    """

    json_path = getattr(scan, f"{resolution}_json")
    tsv_path = getattr(scan, f"{resolution}_tsv")

    with open(json_path) as f:
        metadata = json.load(f)

    sampling_rate = metadata["SamplingFrequency"]
    start_time = metadata["StartTime"]
    columns = metadata["Columns"]

    tsv_file = pd.read_csv(tsv_path, sep="\t", header=None, names=columns)

    waveform_column = HAPPY_WAVEFORM_COLUMN[resolution]
    waveform = tsv_file[waveform_column].to_numpy()

    return {
        "Waveform": waveform,
        "Sampling Rate": sampling_rate,
        "Start Time": start_time,
    }


# ==========================================================
# GROUND TRUTH
# ==========================================================

def _bandpass_filter(signal: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass
    """

    nyquist = fs / 2
    low, high = band
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)


def load_ground_truth(scan: Scan) -> dict:
    """
    Load the hr.txt ground truth cardiac signal for one scan.

    Returns
    {"Waveform": np.ndarray, "Sampling Rate": float, "Start Time": float}
    """

    file = pd.read_csv(scan.ground_truth, sep="\t", header=None)
    waveform = file[GROUND_TRUTH_COLUMN].to_numpy(dtype=float)
    waveform = _bandpass_filter(waveform, SAMPLING_RATE, GROUND_TRUTH_BANDPASS_HZ)

    return {
        "Waveform": waveform,
        "Sampling Rate": SAMPLING_RATE,
        "Start Time": 0.0,
    }


# ==========================================================
# ALIGNMENT
# ==========================================================

def align_to_common_timebase(happy: dict, ground_truth: dict) -> dict:
    """
    Resample ground_truth onto happy's time axis.

    Design choice: resampling direction is ground_truth -> happy
    """

    happy_signal = np.asarray(happy["Waveform"])
    happy_rate = happy["Sampling Rate"]
    happy_start = happy["Start Time"]

    gt_signal = np.asarray(ground_truth["Waveform"])
    gt_rate = ground_truth["Sampling Rate"]
    gt_start = ground_truth["Start Time"]

    t_happy = happy_start + np.arange(len(happy_signal)) / happy_rate
    t_gt = gt_start + np.arange(len(gt_signal)) / gt_rate


    ## error check
    if t_happy[0] < t_gt[0] or t_happy[-1] > t_gt[-1]:
        raise ValueError(
            f"happy time range [{t_happy[0]:.2f}, {t_happy[-1]:.2f}]s "
            f"falls outside ground truth time range "
            f"[{t_gt[0]:.2f}, {t_gt[-1]:.2f}]s --≥ check start_time "
            f"assumptions before interpolating."
        )

    aligned_target = np.interp(t_happy, t_gt, gt_signal)

    return {
        "Waveform": happy_signal,
        "Target": aligned_target,
        "Sampling Rate": happy_rate,
        "Start Time": happy_start,
    }


# ==========================================================
# PUBLIC API
# ==========================================================

def build_labels(scan: Scan, resolution: str = "stdres") -> dict:
    """
    Load + align happy waveform and ground truth for one scan.

    Returns
    {
        "Scan ID": str,
        "Waveform": np.ndarray,  # happy's waveform --> model input
        "Target": np.ndarray,    # aligned hr.txt -- >what  model is trained for
        "Sampling Rate": float,
        "Start Time": float,
    }
    """

    happy = load_happy_waveform(scan, resolution)
    ground_truth = load_ground_truth(scan)
    aligned = align_to_common_timebase(happy, ground_truth)

    return {"Scan ID": scan.id, **aligned}
