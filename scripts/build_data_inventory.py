"""
build_data_inventory.py

One row per subject/session; show which files are present
vs missing
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

import pandas as pd

from config import OUTPUT_ROOT
from src.dataset.discovery import discover_scans

scans = discover_scans()

rows = []
for scan in scans:
    rows.append({
        "Subject": scan.subject,
        "Session": scan.session,
        "Bold (local copy)": scan.has_bold(),
        "Bold JSON": scan.bold_json is not None and scan.bold_json.exists(),
        "Motion": scan.has_motion(),
        "Ground Truth (hr.txt)": scan.has_ground_truth(),
        "Cardpulse TSV": scan.cardpulse_tsv is not None and scan.cardpulse_tsv.exists(),
        "Cardpulse JSON": scan.cardpulse_json is not None and scan.cardpulse_json.exists(),
        "Sliceres TSV": scan.sliceres_tsv is not None and scan.sliceres_tsv.exists(),
        "Sliceres JSON": scan.sliceres_json is not None and scan.sliceres_json.exists(),
        "Stdres TSV": scan.stdres_tsv is not None and scan.stdres_tsv.exists(),
        "Stdres JSON": scan.stdres_json is not None and scan.stdres_json.exists(),
        "Valid (ready for ML dataset)": scan.validate(),
        "Missing": ", ".join(scan.missing_files()),
    })

df = pd.DataFrame(rows).sort_values(["Subject", "Session"]).reset_index(drop=True)

OUTPUT_ROOT.mkdir(exist_ok=True)
out_path = OUTPUT_ROOT / "data_inventory.csv"
df.to_csv(out_path, index=False)

print(f"Wrote {len(df)} rows to {out_path}")
print(f"{df['Valid (ready for ML dataset)'].sum()} valid / {len(df)} total")
