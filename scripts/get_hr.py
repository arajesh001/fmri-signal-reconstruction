from pathlib import Path
import shutil
import re
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

SOURCE_DIR = Path(
    "/Users/ashvath/Library/CloudStorage/OneDrive-NorthwesternUniversity/ground_truth"
)

DEST_DIR = Path.cwd().parent / "data" / "hand"

# ==========================================================
# Regex
# ==========================================================

pattern = re.compile(
    r"(sub-\d+).*(ses-\d+).*\.txt$",
    re.IGNORECASE
)

rows = []
copied = 0

# ==========================================================
# Copy HR recordings
# ==========================================================

for file_path in SOURCE_DIR.rglob("*.txt"):

    match = pattern.search(file_path.name)

    if match is None:
        continue

    subject = match.group(1)
    session = match.group(2)

    save_dir = DEST_DIR / subject / session
    save_dir.mkdir(parents=True, exist_ok=True)

    destination = save_dir / "hr.txt"

    shutil.copy2(file_path, destination)

    copied += 1

    rows.append({
        "Subject": subject,
        "Session": session,
        "Original Path": str(file_path),
        "Saved Path": str(destination)
    })

# ==========================================================
# Save inventory
# ==========================================================

inventory = pd.DataFrame(rows)

OUTPUT = Path.cwd().parent / "output"
OUTPUT.mkdir(exist_ok=True)

inventory.to_csv(
    OUTPUT / "hr_inventory.csv",
    index=False
)

print("=" * 50)
print(f"Copied {copied} HR recordings.")
print("=" * 50)

print()
print(inventory.head())