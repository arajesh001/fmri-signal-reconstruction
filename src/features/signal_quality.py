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

# via abs(skew - median)
GOOD_PERCENTILE = 50
OK_PERCENTILE = 80

def quality_weight(
    cardiac_windows: np.ndarray,
    good_percentile: float = GOOD_PERCENTILE,
    ok_percentile: float = OK_PERCENTILE,
) -> np.ndarray:
    """
    Per-window reliability weight heuristic in {0.0, 0.5, 1.0}.

    "quality" is defined as closeness to this
    dataset's median skewness.

    Returns shape (n_windows,), values in {0.0, 0.5, 1.0}.
    """

    skewness = skewness_sqi(cardiac_windows)
    deviation = np.abs(skewness - np.median(skewness))

    good_cutoff, ok_cutoff = np.percentile(deviation, [good_percentile, ok_percentile])

    weights = np.zeros(len(skewness))
    weights[deviation <= ok_cutoff] = 0.5
    weights[deviation <= good_cutoff] = 1.0

    return weights
