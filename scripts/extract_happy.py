from pathlib import Path
import shutil

# ==========================================================
# PATHS
# ==========================================================

SOURCE = Path(
    "/Users/ashvath/Library/CloudStorage/OneDrive-NorthwesternUniversity/happy_outputs"
)

DEST = Path(
    "/Users/ashvath/fmri-signal-reconstruction/data/happy"
)

# ==========================================================
# Files to keep
# ==========================================================

KEEP = {
    "cardpulse": [
        "desc-cardpulsefromfmri_timeseries.tsv.gz",
        "desc-cardpulsefromfmri_timeseries.json",
    ],

    "sliceres": [
        "desc-slicerescardfromfmri_timeseries.tsv.gz",
        "desc-slicerescardfromfmri_timeseries.json",
    ],

    "stdres": [
        "desc-stdrescardfromfmri_timeseries.tsv.gz",
        "desc-stdrescardfromfmri_timeseries.json",
    ],

    "vessel": [
        "desc-vessels_mask.nii.gz",
        "desc-vessels_mask.json",
        "desc-vessels_map.nii.gz",
    ]
}

copied = 0

# ==========================================================
# Search all HAPPY outputs
# ==========================================================

for file in SOURCE.rglob("*"):

    if not file.is_file():
        continue

    name = file.name

    subject = None
    session = None

    for part in file.parts:
        if part.startswith("sub-"):
            subject = part
        elif part.startswith("ses-"):
            session = part

    if subject is None or session is None:
        continue

    for folder, suffixes in KEEP.items():

        for suffix in suffixes:

            if name.endswith(suffix):

                out_dir = DEST / subject / session / folder
                out_dir.mkdir(parents=True, exist_ok=True)

                if suffix.endswith(".tsv.gz"):
                    new_name = "timeseries.tsv.gz"

                elif suffix.endswith(".json"):

                    if "mask" in suffix:
                        new_name = "mask.json"
                    else:
                        new_name = "metadata.json"

                elif suffix.endswith("mask.nii.gz"):
                    new_name = "mask.nii.gz"

                elif suffix.endswith("map.nii.gz"):
                    new_name = "map.nii.gz"

                else:
                    new_name = file.name

                shutil.copy2(file, out_dir / new_name)

                copied += 1
                break

print(f"Copied {copied} files.")