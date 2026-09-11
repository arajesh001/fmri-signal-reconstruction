"""
train.py

Training loop + LOSO evaluation for the CNN.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset

from src.dataset.builder import build_dataset
from src.models.wandb_logging import log_results_to_wandb
from src.models.xgboost_baseline import summarize_results

from .dataset import get_loso_split, make_dataloader
from .losses import SpectralMSELoss
from .model import BaselineCNN, CardiacCNN, get_device, ndarray_to_tensor


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
    model_type: str = "baseline",  # "baseline" -> BaselineCNN, "cardiac" -> CardiacCNN
    n_epochs: int = 50,
    batch_size: int = 32,
    checkpoint_dir: Path = Path("checkpoints"),
) -> list[dict]:
    """
    Returns fold_results: lst of {"subject", "rmse", "corr"} dicts 
    ---> same shape xgboost_baseline.run_loso_cv !!!!
    ***IMPORTANT** give straight --> summarize_results/log_results_to_wandb.
    """

    fold_results = []
    device = get_device()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # eeg-wheelchair saved to cwd directly, we use a subdir

    for test_subject_id in np.unique(subjects):
        # tts fold
        X_train_full, X_test, y_train_full, y_test = get_loso_split(X, y, subjects, test_subject_id)

        # val split --> (X_train_full/y_train_full) --> (~85/15)
        X_train_full_t = ndarray_to_tensor(np.transpose(X_train_full, (0, 2, 1)))
        y_train_full_t = ndarray_to_tensor(y_train_full)
        full_train_data = TensorDataset(X_train_full_t, y_train_full_t)

        train_subset, val_subset = random_split(
            full_train_data, [0.85, 0.15],
            generator=torch.Generator().manual_seed(42),
        )

        # build dataloaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

        #this needs transpose --> use fn
        test_loader = make_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

        # model + optimizer
        model = (BaselineCNN() if model_type == "baseline" else CardiacCNN()).to(device)

        # TUNE LATER !!!!!
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  
        loss_fn = SpectralMSELoss()

        # 6. train loop w/ early stopping 
        best_loss = float("inf")
        counter = 0
        patience = 15 # might change
        tol = 0.001
        warmup_epochs = 15
        weights = checkpoint_dir / f"best_model_subject_{test_subject_id}.pt"

        # delete if its there from previous runs
        if weights.exists():
            weights.unlink()

        for epoch in range(n_epochs):
            avg_loss = train(model, train_loader, optimizer, loss_fn, device)
            val_loss = evaluate_loss(model, val_loader, loss_fn, device)
            print(f"[Subject {test_subject_id}] Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss - tol:
                best_loss = val_loss
                torch.save(model.state_dict(), weights)
                counter = 0
            else:
                counter += 1
            if epoch + 1 > warmup_epochs and counter >= patience:
                print(f"[Subject {test_subject_id}] Early stopping at epoch {epoch+1}")
                break

        # 7. load in the best weights instead o/using most recent
        if weights.exists():
            state_dict = torch.load(weights, weights_only=True)
            model.load_state_dict(state_dict)
        # just use most recent in case the file DNE
        else:
            pass

        # score on the held-out sub
        metrics = evaluate(model, test_loader, device)
        print(f"[Subject {test_subject_id}] RMSE: {metrics['rmse']:.4f} | Corr: {metrics['corr']:.4f}\n")

        #record
        fold_results.append({"subject": test_subject_id, **metrics})

    return fold_results


# script to run 
if __name__ == "__main__":
    dataset = build_dataset()

    # just a start, idk --> change later
    n_epochs = 50
    batch_size = 32

    # run
    fold_results = run_loso(
        dataset["X_cnn"], dataset["y"], dataset["subject_ids"],
        model_type="baseline",
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # print loss at each epoch 
    summarize_results(fold_results)

    # w&b logging
    log_results_to_wandb(fold_results, config={
        "model": "baseline",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "mse_weight": 1.0,       
        "spectral_weight": 0.01,
    })
