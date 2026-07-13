from pathlib import Path
import pandas as pd
import re

# Local Path:
ROOT = ROOT = Path.home() / "Library" / "CloudStorage" / "OneDrive-NorthwesternUniversity" / "HAPPY DATA"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# FORMAT TO FOLLOW:
# sub-01_ses-01_task-hand_echo-1_bold.nii.gz

pattern = re.compile(
    r'(?P<subject>sub-[^_]+)_'
    r'(?P<session>ses-[^_]+)_'
    r'task-(?P<task>[^_]+)_'
    r'echo-(?P<echo>\d+)_'
    r'(?P<datatype>.+)'
)

rows = []

for file in ROOT.iterdir():

    if not file.is_file():
        continue

    match = pattern.match(file.name)

    if match is None:
        rows.append({
            "filename": file.name,
            "subject": None,
            "session": None,
            "task": None,
            "echo": None,
            "datatype": "UNKNOWN",
            "extension": "".join(file.suffixes),
            "size_MB": round(file.stat().st_size / 1e6, 2)
        })
        continue

    rows.append({
        "filename": file.name,
        "subject": match.group("subject"),
        "session": match.group("session"),
        "task": match.group("task"),
        "echo": int(match.group("echo")),
        "datatype": match.group("datatype"),
        "extension": "".join(file.suffixes),
        "size_MB": round(file.stat().st_size / 1e6, 2)
    })

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_DIR / "inventory.csv", index=False)

print(df.head())

print("\nSubjects found:")
print(df["subject"].value_counts())

print("\nFile types:")
print(df["datatype"].value_counts())