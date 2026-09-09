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
# BASELINE MODEL
# ==========================================================

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    IN:  (batch, 14, window_size)
            [raw signal, 6 motion params, 6 motion derivatives, FD]
    OUTS: (batch, window_size) --> the cleaned waveform
    """

    def __init__(
        self,
        in_channels: int = 14,   # raw  + 6 motion + 6 d/dx + FD
        num_filters: int = 16,   # width o/e hidden layer
        kernel_size: int = 5,    # num timesteps each filter looks at
        num_layers: int = 5,     # total conv layers; -->match happy's default
        dropout_rate: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act_layer = nn.ReLU if activation == "relu" else nn.Tanh

        layers = []

        # input block: 14 in -> num_filters out
        layers += [
            nn.Conv1d(in_channels, num_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(dropout_rate),
            act_layer(),
        ]

        # middle blocks: num_filters -> num_filters 
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv1d(num_filters, num_filters, kernel_size, padding="same"),
                nn.BatchNorm1d(num_filters),
                nn.Dropout(dropout_rate),
                act_layer(),
            ]

        #output block: num_filters -> 1 (the cleaned waveform)
        layers.append(nn.Conv1d(num_filters, 1, kernel_size, padding="same"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 14, window_size)
        returns: (batch, window_size) --> channel dim squeezed out
        """
        out = self.net(x)          
        # (batch, 1, window_size)
        return out.squeeze(1)      
        # (batch, window_size)



# ==========================================================
# MODEL
# ==========================================================


class CardiacCNN(nn.Module):
    """
    Real Cardiac CNN, taking into account the baseline. 
    """

    def __init__(self, n_channels: int = 14, window_size: int = 250):
        super().__init__()
        pass

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return (batch, window_size) !!!!
        pass


if __name__ == "__main__":
    model = CardiacCNN()
    dummy = torch.randn(8, 14, 250)
    out = model(dummy)
    print(out.shape)
