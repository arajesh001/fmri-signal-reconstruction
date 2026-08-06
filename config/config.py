"""
config.py

Loads config.yaml and exposes project-wide settings.

Usage
from config import (
    HAND_ROOT,
    HAPPY_ROOT,
    MOTION_ROOT,
    GROUND_TRUTH_ROOT,
    DATASET_ROOT,
    TASK,
    ECHO,
)
"""

from pathlib import Path
import yaml

# ==========================================================
# PROJECT ROOT
# ==========================================================

# project/
# ├── config/
# │   ├── config.py
# │   └── config.yaml
# ├── data/
# └── src/

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================================
# LOAD YAML
# ==========================================================

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_FILE, "r") as f:
    CONFIG = yaml.safe_load(f)

# ==========================================================
# PATHS
# ==========================================================

HAND_ROOT = PROJECT_ROOT / CONFIG["paths"]["hand"]
HAPPY_ROOT = PROJECT_ROOT / CONFIG["paths"]["happy"]
MOTION_ROOT = PROJECT_ROOT / CONFIG["paths"]["motion"]
GROUND_TRUTH_ROOT = PROJECT_ROOT / CONFIG["paths"]["ground_truth"]

RAW_DATA_ROOT = PROJECT_ROOT / CONFIG["paths"]["raw_data"]
OUTPUT_ROOT = PROJECT_ROOT / CONFIG["paths"]["output"]
DATASET_ROOT = PROJECT_ROOT / CONFIG["paths"]["dataset"]

# ==========================================================
# DATASET SETTINGS
# ==========================================================

TASK = CONFIG["dataset"]["task"]
ECHO = CONFIG["dataset"]["echo"]

# ==========================================================
# PHYSIO SETTINGS
# ==========================================================

SAMPLING_RATE = CONFIG["physio"]["sampling_rate"]
CARDIAC_COLUMN = CONFIG["physio"]["cardiac_column"]