"""
dataset.py

LOSO splitting + DataLoader construction for CNN
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import ndarray_to_tensor
# ==========================================================
# PUBLICAPI
# ==========================================================

def get_loso_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    test_subject_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Boolean-mask LOSO split -> all windows from test_subject_id go -->
    test, everything else to train. 


    Returns: X_train, X_test, y_train, y_test.
    """

    test_mask = subjects == test_subject_id
    train_mask = subjects != test_subject_id

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Returns the torch dataloader given the X and y arrays.
    """

    X_transposed = np.transpose(X, (0, 2, 1))  
    # (n, window_size, 14) ---> (n, 14, window_size)

    X_tensor = ndarray_to_tensor(X_transposed)
    y_tensor = ndarray_to_tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
