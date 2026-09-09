"""
losses.py

Custom loss:

L_total = L_MSE + lambda * L_Spec
"""

import torch
import torch.nn as nn


class SpectralMSELoss(nn.Module):

    def __init__(self, mse_weight: float = 1.0, spectral_weight: float = 0.01):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight  # !!!!!!! TUNE THIS LATER; chose approx !!!!!!

    def _spectral_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Inputs --> (Prediction, Target)
        Outputs --> Spectal Loss Value
        """
        # whole spectrum for now -->> could restrict to cardiac band later
        freq_pred = torch.abs(torch.fft.rfft(pred, dim=-1))
        freq_target = torch.abs(torch.fft.rfft(target, dim=-1))
        spec_loss = torch.mean((freq_target - freq_pred) ** 2)
        return spec_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((target - pred) ** 2)
        spec = self._spectral_loss(pred, target)
        return self.mse_weight * mse + self.spectral_weight * spec

    def components(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Inputs --> (Prediction, Target)
        Outputs --> {"mse", "spectral", "total"}
        """
        with torch.no_grad():
            mse = torch.mean((target - pred) ** 2)
            spec = self._spectral_loss(pred, target)
            total = self.mse_weight * mse + self.spectral_weight * spec

        return {"mse": mse.item(), "spectral": spec.item(), "total": total.item()}
