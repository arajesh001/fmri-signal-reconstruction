"""
builder.py

High-level dataset construction --> discovery, filtering,
per-scan labels/features/windowing, and concatenation into a final
X, y dataset with subject IDs preserved for grouped train/test
splitting (split by subject, not by window)
"""

import numpy as np

from src.features.motion import build_features

from .discovery import discover_scans
from .labels import build_labels
from .preprocess import preprocess_scan
from .scan import Scan
from .windows import build_windows


# ==========================================================
# FILTER
# ==========================================================

def filter_scans(scans: list[Scan]) -> list[Scan]:
    """Keep only scans that pass Scan.validate()."""
    return [s for s in scans if s.validate()]


# ==========================================================
# SUMMARY
# ==========================================================

def summarize_dataset(scans: list[Scan]) -> None:
    """Print which scans/subjects are going into the dataset."""

    subjects = sorted({s.subject for s in scans})
    print(f"{len(scans)} scans across {len(subjects)} subjects: {subjects}")
    for s in scans:
        print(f"  {s.id}")


# ==========================================================
# PUBLIC API
# ==========================================================

def build_dataset(
    resolution: str = "stdres",
    target_rate: float = 25.0,
    window_size: int = 250,
    stride: int = 125,
) -> dict:
    """
    Main entry point. Builds the full ML dataset from every valid scan.

    Returns
    {
        "X_xgb": (n_windows_total, 74),  # cardiac stats, motion stats,
                                          # frequency, signal-quality
        "X_cnn": (n_windows_total, window_size, 14),
        "y": (n_windows_total, window_size),
        "subject_ids": (n_windows_total,)  -- e.g. "sub-20", for grouped splitting
    }
    """

    scans = discover_scans()
    scans = filter_scans(scans)
    summarize_dataset(scans)

    X_xgb_parts, X_cnn_parts, y_parts, subject_id_parts = [], [], [], []
    skipped = []

    for scan in scans:
        try:
            labels = build_labels(scan, resolution)
            features = build_features(scan, labels["Sampling Rate"], len(labels["Waveform"]))
            preprocessed = preprocess_scan(labels["Waveform"], features, labels["Target"])
            windows = build_windows(
                preprocessed["Waveform"], preprocessed["Motion Features"],
                preprocessed["Target"], window_size, stride,
                fs=labels["Sampling Rate"],
            )
        except ValueError as e:
            skipped.append((scan.id, str(e).splitlines()[0]))
            continue

        n_windows = windows["X_xgb"].shape[0]

        X_xgb_parts.append(windows["X_xgb"])
        X_cnn_parts.append(windows["X_cnn"])
        y_parts.append(windows["y"])
        subject_id_parts.append(np.full(n_windows, scan.subject))

    if skipped:
        print(f"\nSkipped {len(skipped)} scan(s) that failed to build:")
        for scan_id, msg in skipped:
            print(f"  {scan_id}: {msg}")

    X_xgb = np.concatenate(X_xgb_parts, axis=0)
    X_cnn = np.concatenate(X_cnn_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subject_ids = np.concatenate(subject_id_parts, axis=0)

    print(f"Built dataset: X_xgb={X_xgb.shape}, X_cnn={X_cnn.shape}, "
          f"y={y.shape}, subject_ids={subject_ids.shape}")

    return {"X_xgb": X_xgb, "X_cnn": X_cnn, "y": y, "subject_ids": subject_ids}
