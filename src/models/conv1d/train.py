"""
train.py

Training loop + LOSO evaluation for the CNN.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.builder import build_dataset
from src.models.xgboost_baseline import log_results_to_wandb, summarize_results

from .dataset import get_loso_split, make_dataloader
from .losses import SpectralMSELoss
from .model import BaselineCNN, CardiacCNN, get_device


# ==========================================================
# ONE EPOCH --> direct import from eeg-wheelchair-control; loss swapped
# =========================================================

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    One training epoch.
    """

    model.train()
    running_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """
    Avg loss over a dataloader, no gradient tracking --> for
    val split ( early stopping).
    """

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """
    Predicts over full dataloader and scores rmse/corr.
    Returns {"rmse": ..., "corr": ...}.
    """

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    corr = float(np.corrcoef(preds, targets)[0, 1])

    return {"rmse": rmse, "corr": corr}


# ==========================================================
# PUBLIC API
# ==========================================================

def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 32,
    checkpoint_dir: Path = Path("checkpoints"),
) -> list[dict]:
    pass
