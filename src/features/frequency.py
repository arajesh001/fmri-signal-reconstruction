"""
frequency.py

Frequency-domain features per window: dominant frequency + band power
within the plausible cardiac band. See project discussion (PPG/HR
feature-engineering research) for why these are the standard first
additions beyond flattened time-domain stats -- mean/std/min/max/rms
can't distinguish where in the cardiac cycle a window sits, frequency
content can.

Operates on already-windowed arrays -- same shape convention as
windows.flatten_window_stats -- so this plugs into the same place
X_xgb gets assembled, e.g.:

    X_xgb = np.concatenate(
        [cardiac_stats, motion_stats, frequency_features], axis=1
    )

Independent of signal_quality.py on purpose -- no shared imports or
constants between the two files, so either can be added, removed, or
changed without touching the other.
"""

import numpy as np
from scipy.signal import periodogram

# Matches labels.py's GROUND_TRUTH_BANDPASS_HZ -- not imported from
# there on purpose (independence). check_alignment.ipynb's spectral
# check cell used (0.5, 3.0) for a broader exploratory look; this
# defaults to the tighter band labels.py actually filters to.
CARDIAC_BAND_HZ = (0.8, 3.0)


# ==========================================================
# SHARED HELPER
# ==========================================================

def _band_periodogram(
    windows: np.ndarray,
    fs: float,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Periodogram of every window, restricted to `band`.

    Returns: (band_freqs, band_psd)
    """

    freqs, psd = periodogram(windows, fs=fs, axis=1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[mask], psd[:, mask]


# ==========================================================
# PUBLIC API
# ==========================================================

def dominant_frequency(
    windows: np.ndarray,
    fs: float,
    band: tuple[float, float] = CARDIAC_BAND_HZ,
) -> np.ndarray:
    """
    Peak frequency within 'band', per window.

    Returns shape (n_windows,).
    """

    band_freqs, band_psd = _band_periodogram(windows, fs, band)
    peak_idx = np.argmax(band_psd, axis=1)
    return band_freqs[peak_idx]


def band_power(
    windows: np.ndarray,
    fs: float,
    band: tuple[float, float] = CARDIAC_BAND_HZ,
) -> np.ndarray:
    """
    Total power within 'band', per window
    Returns: --> shape (n_windows,).
    """

    band_freqs, band_psd = _band_periodogram(windows, fs, band)
    return np.trapz(band_psd, x=band_freqs, axis=1)


def build_frequency_features(cardiac_windows: np.ndarray, fs: float) -> np.ndarray:
    """
    Dominant frequency + band power, stacked into one feature block.

    Returns: shape (n_windows, 2).
    """

    band_freqs, band_psd = _band_periodogram(cardiac_windows, fs, CARDIAC_BAND_HZ)

    peak_idx = np.argmax(band_psd, axis=1)
    dom_freq = band_freqs[peak_idx]
    power = np.trapz(band_psd, x=band_freqs, axis=1)

    return np.column_stack([dom_freq, power])
