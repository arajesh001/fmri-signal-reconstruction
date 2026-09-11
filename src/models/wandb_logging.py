"""
wandb_logging.py

W&B logging, shared by both models --> moved to seperate file cuz of weird errors w/imports
"""

import numpy as np
import wandb

# Same project for every model
WANDB_PROJECT = "fmri-cardiac-reconstruction"


# ==========================================================
# PUBLIC API
# ==========================================================

def log_results_to_wandb(fold_results: list[dict], config: dict) -> None:
    """
    Log one run's per-fold + macro-averaged results to W&B.

    config: hyperparameters/settings for this run --> shown alongside
        the results on the run's W&B page.
    """

    run = wandb.init(project=WANDB_PROJECT, config=config, reinit=True)

    rmses = [r["rmse"] for r in fold_results]
    corrs = [r["corr"] for r in fold_results]

    # log
    run.log({
        "macro_rmse": float(np.mean(rmses)),
        "macro_rmse_std": float(np.std(rmses)),
        "macro_corr": float(np.mean(corrs)),
        "macro_corr_std": float(np.std(corrs)),
    })

    # log desired metrics in a table per run
    table = wandb.Table(columns=["subject", "rmse", "corr"])
    for result in fold_results:
        table.add_data(result["subject"], result["rmse"], result["corr"])
    run.log({"per_fold_results": table})

    run.finish()
