"""
signal_quality.py

Signal-quality-index (SQI) features per window: skewness and
kurtosis.
"""

import numpy as np
from scipy.stats import kurtosis, skew


def skewness_sqi(windows: np.ndarray) -> np.ndarray:
    """
    Skewness per window.

    Returns:  shape (n_windows,).
    """

    return skew(windows, axis=1)


def kurtosis_sqi(windows: np.ndarray) -> np.ndarray:
    """
    Excess kurtosis per window

    Returns: shape (n_windows,).
    """

    return kurtosis(windows, axis=1, fisher=True)


def build_signal_quality_features(cardiac_windows: np.ndarray) -> np.ndarray:
    """
    Skewness + kurtosis, stacked into one feature block.

    Returns: shape (n_windows, 2).
    """

    return np.column_stack([
        skewness_sqi(cardiac_windows),
        kurtosis_sqi(cardiac_windows),
    ])


# ==========================================================
# RELIABILITY WEIGHTING
# ==========================================================

## ADD THIS LATER!!! 
