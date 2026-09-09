"""
check_scans.py

Tells you exactly how many subject/session pairs discover_scans() finds,
and which ones are fully valid 
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

from src.dataset.discovery import discover_scans

scans = discover_scans()

print(f"Found {len(scans)} subject/session pairs under data/_staging.\n")

valid = [s for s in scans if s.validate()]
invalid = [s for s in scans if not s.validate()]

print(f"{len(valid)} are fully valid (ready to build the ML dataset from):")
for s in valid:
    print(f"  {s.id}")

print(f"\n{len(invalid)} are missing something:")
for s in invalid:
    print(f"  {s.id}: missing {s.missing_files()}")