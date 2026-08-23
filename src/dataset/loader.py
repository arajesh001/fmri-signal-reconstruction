"""
loader.py

Load a previously cached dataset back from disk (use w/
save.py).
"""

import json
from pathlib import Path

import numpy as np

from config import DATASET_ROOT

from .save import DATASET_FILENAME, METADATA_FILENAME


# ==========================================================
# PUBLIC API
# ==========================================================

def load_dataset(path: Path = DATASET_ROOT) -> dict:
    """
    Load a dataset cached by save.save_dataset.
    """

    dataset_path = path / DATASET_FILENAME

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No cached dataset at {dataset_path} -- run "
            f"builder.build_dataset() and save.save_dataset() first."
        )

    with np.load(dataset_path) as npz:
        dataset = {key: npz[key] for key in npz.files}

    metadata_path = path / METADATA_FILENAME
    if metadata_path.exists():
        with open(metadata_path) as f:
            dataset["metadata"] = json.load(f)

    return dataset
