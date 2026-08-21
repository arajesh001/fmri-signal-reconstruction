"""
windows.py

Sliding-window extraction over the aligned (cardiac, motion, ground
truth) arrays for one scan. Produces two representations from the
same underlying windows:

    - flattened per-window stats, for XGBoost
    - raw multi-channel arrays, for the 1D CNN

Window size and stride: 10s windows, 50% overlap @ 25 Hz (stdres)
that's window_size=250, stride=125.
"""

import numpy as np

from src.features.frequency import build_frequency_features
from src.features.signal_quality import build_signal_quality_features


# ==========================================================
# CORE SLIDING WINDOW
# ==========================================================

def make_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Slice a 1D (n,) or 2D (n, channels) array into overlapping windows.

    Returns
    (n_windows, window_size) for 1D input, or
    (n_windows, window_size, channels) for 2D input.
    """

    n = signal.shape[0]

    if n < window_size:
        raise ValueError(
            f"signal length {n} is shorter than window_size {window_size}"
        )

    starts = range(0, n - window_size + 1, stride)
    return np.stack([signal[start:start + window_size] for start in starts])


# ==========================================================
# XGBOOST REPRESENTATION
# ==========================================================

def flatten_window_stats(windows: np.ndarray) -> np.ndarray:
    """
    Convert a batch of windows into flattened summary stats per
    channel, for XGBoost.

    Params:
    windows: (n_windows, window_size) or (n_windows, window_size, channels)

    Returns
    (n_windows, channels * 5) -- mean, std, min, max, rms per channel,
    in that order. channels=1 for a plain 1D input (e.g. cardiac).
    """

    if windows.ndim == 2:
        windows = windows[:, :, np.newaxis]

    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    minimum = windows.min(axis=1)
    maximum = windows.max(axis=1)
    rms = np.sqrt((windows ** 2).mean(axis=1))

    return np.concatenate([mean, std, minimum, maximum, rms], axis=1)


# ==========================================================
# CNN REPRESENTATION
# ==========================================================

def stack_window_channels(
    cardiac_windows: np.ndarray,
    motion_windows: np.ndarray,
) -> np.ndarray:
    """
    Stack cardiac waveform windows with motion feature windows into
    one multi-channel array for the CNN.

    Params:
    cardiac_windows: (n_windows, window_size)
    motion_windows: (n_windows, window_size, n_motion_channels)

    Returns
    (n_windows, window_size, 1 + n_motion_channels)
    """

    cardiac_expanded = cardiac_windows[:, :, np.newaxis]
    return np.concatenate([cardiac_expanded, motion_windows], axis=2)


# ==========================================================
# PUBLIC API
# ==========================================================

def build_windows(
    cardiac: np.ndarray,
    motion_features: np.ndarray,
    target: np.ndarray,
    window_size: int,
    stride: int,
    fs: float,
) -> dict:
    """
    Build both representations (XGBoost-flattened, CNN-stacked) plus
    the aligned target windows, for one scan.

    Params:
    cardiac: (n_samples,)
    motion_features: (n_samples, 13)
    target: (n_samples,)
    fs: sampling rate of cardiac/target (e.g. 25.0 for stdres) --
        needed by frequency.build_frequency_features, not used
        anywhere else in this function.

    Returns
    {
        "X_xgb": (n_windows, 5 + 65 + 2 + 2),  # cardiac stats, motion
                                                # stats, frequency, SQI
        "X_cnn": (n_windows, window_size, 14),
        "y": (n_windows, window_size),
    }

    """

    if not (len(cardiac) == len(motion_features) == len(target)):
        raise ValueError(
            f"cardiac ({len(cardiac)}), motion_features "
            f"({len(motion_features)}), and target ({len(target)}) must "
            f"be the same length -- check labels.py/features.py alignment "
            f"upstream of this call."
        )

    cardiac_windows = make_windows(cardiac, window_size, stride)
    motion_windows = make_windows(motion_features, window_size, stride)
    target_windows = make_windows(target, window_size, stride)

    X_cnn = stack_window_channels(cardiac_windows, motion_windows)

    cardiac_stats = flatten_window_stats(cardiac_windows)
    motion_stats = flatten_window_stats(motion_windows)
    frequency_features = build_frequency_features(cardiac_windows, fs)
    signal_quality_features = build_signal_quality_features(cardiac_windows)

    X_xgb = np.concatenate(
        [cardiac_stats, motion_stats, frequency_features, signal_quality_features],
        axis=1,
    )

    return {"X_xgb": X_xgb, "X_cnn": X_cnn, "y": target_windows}
