"""
save.py
"""

import json
from pathlib import Path

import numpy as np

from config import DATASET_ROOT

DATASET_FILENAME = "dataset.npz"
METADATA_FILENAME = "metadata.json"


# ==========================================================
# PUBLIC API
# ==========================================================

def save_dataset(dataset: dict, path: Path = DATASET_ROOT, **metadata) -> None:
    """
    Save a dataset dict to path/dataset.npz, plus path/metadata.json.

    Takes whole dict (X_xgb, X_cnn, y, subject_ids, sample_weight,
    ...)

    **metadata: anything worth recording  like:
        save_dataset(dataset, window_size=250, stride=125,
        resolution="stdres")
    """

    path.mkdir(parents=True, exist_ok=True)

    np.savez(path / DATASET_FILENAME, **dataset)

    with open(path / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)
