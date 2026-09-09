"""
build_dataset.py

Runs the full dataset pipeline (discovery -> labels -> features ->
preprocess -> windows) across all valid scans && prints the result.

"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

from src.dataset.builder import build_dataset

result = build_dataset()

print()
print("Shapes:")
for key, value in result.items():
    print(f"  {key}: {value.shape}")
