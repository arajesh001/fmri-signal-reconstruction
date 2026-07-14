# from pathlib import Path
# import pandas as pd
# import shutil
# import re

# ROOT = Path("/Users/ashvath/Library/CloudStorage/OneDrive-NorthwesternUniversity/HAPPY DATA")

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR = PROJECT_ROOT / "data"
# DATA_DIR.mkdir(exist_ok=True)

# pattern = re.compile(
#     r'(?P<subject>sub-[^_]+)_'
#     r'(?P<session>ses-[^_]+)_'
#     r'task-(?P<task>[^_]+)_'
#     r'echo-(?P<echo>\d+)_'
#     r'(?P<datatype>.+)'
# )

# rows = []

# for file in ROOT.iterdir():

#     if not file.is_file():
#         continue

#     match = pattern.match(file.name)

#     if match is None:
#         continue

#     subject = match.group("subject")
#     session = match.group("session")
#     task = match.group("task")
#     echo = match.group("echo")
#     datatype = match.group("datatype")

#     destination = (
#         DATA_DIR
#         / task
#         / subject
#         / session
#     )

#     destination.mkdir(parents=True, exist_ok=True)

#     # using a symbolic link rather than the actual file copy
#     target = destination / file.name
#     if not target.exists():
#         target.symlink_to(file)

#     rows.append({
#         "task": task,
#         "subject": subject,
#         "session": session,
#         "echo": echo,
#         "datatype": datatype,
#         "filename": file.name
#     })

# df = pd.DataFrame(rows)

# df.to_csv(DATA_DIR / "dataset_inventory.csv", index=False)

# print("=" * 50)
# print(f"Copied {len(df)} files.")
# print("=" * 50)

# print("\nTasks found:")
# print(df["task"].value_counts())

# print("\nSubjects found:")
# print(df["subject"].nunique())

# print("\nSessions found:")
# print(df["session"].value_counts())

# print("\nFile types:")
# print(df["datatype"].value_counts())

############################ BREAK #################################
# ABOVE IS THE PRELIMINARY VERSION TO SORT DATA, BELOW IS ONLY WHAT HAPPY NEEDS


from pathlib import Path
import shutil
import re

ROOT = Path("/Users/ashvath/Library/CloudStorage/OneDrive-NorthwesternUniversity/HAPPY DATA")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUT = PROJECT_ROOT / "data"
OUT.mkdir(exist_ok=True)

pattern = re.compile(
    r'(?P<subject>sub-[^_]+)_'
    r'(?P<session>ses-[^_]+)_'
    r'task-(?P<task>[^_]+)_'
    r'echo-(?P<echo>\d+)_'
    r'(?P<datatype>.+)'
)

copied = 0

for file in ROOT.iterdir():

    if not file.is_file():
        continue

    m = pattern.match(file.name)

    if not m:
        continue

    if m.group("task") != "hand":
        continue

    if m.group("echo") != "1":
        continue

    datatype = m.group("datatype")

    if datatype not in (
        "bold.nii.gz",
        "bold.json"
    ):
        continue

    destination = (
        OUT
        / m.group("subject")
        / m.group("session")
    )

    destination.mkdir(parents=True, exist_ok=True)

    shutil.copy2(file, destination / file.name)

    copied += 1

print(f"Copied {copied} files.")