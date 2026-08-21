"""
frequency.py

Frequency-domain features per window: dominant frequency + band power
within the plausible cardiac band. 
"""

import numpy as np
from scipy.signal import periodogram

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
