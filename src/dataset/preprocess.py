"""
preprocess.py

Per-scan z-score normalization, applied before windowing.
"""

import numpy as np


# ==========================================================
# NORMALIZATION
# ==========================================================

def zscore(signal: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalize along axis 0.
    """

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # guard divide-by-zero on a constant channel

    return (signal - mean) / std


# ==========================================================
# PUBLIC API
# ==========================================================

def preprocess_scan(cardiac: np.ndarray, motion_features: np.ndarray, target: np.ndarray) -> dict:
    """
    Z-score the cardiac waveform, motion features, and target for one
    scan.
    """

    return {
        "Waveform": zscore(cardiac),
        "Motion Features": zscore(motion_features),
        "Target": zscore(target),
    }
