"""
model.py

1D CNN architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# TENSOR / UTILs 
# ==========================================================

def ndarray_to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X).float()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ==========================================================
# MODEL
# ==========================================================

class CardiacCNN(nn.Module):
    pass
