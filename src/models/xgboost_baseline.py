"""
xgboost_baseline.py

XGBoost baseline on the sliding-window dataset.

Uses X_xgb (flattened per-window stats) and a collapsed single-point
target per window.

Evaluated with LOSOS CV (subject_ids from
build_dataset), macro-averaged across folds.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

from src.dataset.builder import build_dataset

# ==========================================================
# LOSO
# ==========================================================

def extract_center_target(y: np.ndarray) -> np.ndarray:
    """
    Collapse windows.py's full-window target (n_windows, window_size)
    down to one scalar per window, for XGBoost.
    """

    center_idx = y.shape[1] // 2
    return y[:, center_idx]


def make_loso_splits(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray):
    """
    Yield (train_idx, test_idx) pairs, one per subject (4 pairs total
    for this dataset).
    """

    logo = LeaveOneGroupOut()
    return logo.split(X, y, groups=subject_ids)

# ==========================================================
# TRAIN
# ==========================================================

def train_fold(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """
    Fit one XGBRegressor on one LOSO fold's training data.
    """
    model = XGBRegressor(max_depth=3, 
                         n_estimators=75, 
                         learning_rate=0.05,
                         random_state=42)
    model.fit(X_train, y_train)

    return model


def evaluate_fold(model: XGBRegressor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Predict on one held-out subject and score it.

    Returns: a dict {"rmse": ..., "corr": ...}
    """
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))

    corr_matrix = np.corrcoef(preds, y_test)
    corr = float(corr_matrix[0, 1])

    return {
    "rmse": rmse,
    "corr": corr
    }


# ==========================================================
# PUBLIC API
# ==========================================================

def run_loso_cv(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> list[dict]:
    """
    Run the full LOSO loop: split, train, evaluate, once per subject.

    Returns a list of per-fold result dicts, each tagged with the
    held-out subject --> preds never pooled across folds; so
    summarize_results can macro-average correct.
    """

    fold_results = []

    for train_idx, test_idx in make_loso_splits(X, y, subject_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # LOSO -->guarantees every row in test_idx belongs to same
        # subject --> [0] reads off which one.
        held_out_subject = subject_ids[test_idx][0]

        model = train_fold(X_train, y_train)
        metrics = evaluate_fold(model, X_test, y_test)

        fold_results.append({"subject": held_out_subject, **metrics})

    return fold_results


def summarize_results(fold_results: list[dict]) -> None:
    """
    Print per-fold metrics and the macro-averaged result across folds.
    """

    print("Per-fold LOSO results:")
    for result in fold_results:
        print(f"  held out {result['subject']}: "
              f"rmse={result['rmse']:.4f}, corr={result['corr']:.4f}")

    rmses = [r["rmse"] for r in fold_results]
    corrs = [r["corr"] for r in fold_results]

    print()
    print(f"Macro-average across {len(fold_results)} folds:")
    print(f"rmse: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
    print(f"corr: {np.mean(corrs):.4f} (+/- {np.std(corrs):.4f})")


