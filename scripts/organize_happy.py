from pathlib import Path
import shutil

ROOT = Path("data/happy")

# Which HAPPY outputs we actually care about
CATEGORY_MAP = {
    "cardpulsefromfmri_timeseries": "cardpulse",
    "slicerescardfromfmri_timeseries": "sliceres",
    "stdrescardfromfmri_timeseries": "stdres",
    "vessels_map": "vessels",
    "vessels_mask": "vessels",
}


def get_new_name(filename: str):
    """Rename files to something consistent."""

    if filename.endswith(".json"):
        return "metadata.json"

    if filename.endswith(".tsv.gz"):
        return "timeseries.tsv.gz"

    if "vessels_map" in filename and filename.endswith(".nii.gz"):
        return "map.nii.gz"

    if "vessels_mask" in filename and filename.endswith(".nii.gz"):
        return "mask.nii.gz"

    return None


# -------------------------------------------------------
# Iterate through each subject
# -------------------------------------------------------

for subject_dir in ROOT.glob("sub-*"):

    if not subject_dir.is_dir():
        continue

    print(f"\nProcessing {subject_dir.name}")

    # Only process files sitting directly inside the subject folder
    for file in list(subject_dir.iterdir()):

        if file.is_dir():
            continue

        filename = file.name

        # Determine which output type it is
        category = None
        for key, folder in CATEGORY_MAP.items():
            if key in filename:
                category = folder
                break

        if category is None:
            continue

        # Extract session (ses-01, ses-02, ...)
        session = next(
            (part for part in filename.split("_") if part.startswith("ses-")),
            None,
        )

        if session is None:
            continue

        # Destination directory
        destination = subject_dir / session / category
        destination.mkdir(parents=True, exist_ok=True)

        # Rename file nicely
        new_name = get_new_name(filename)

        if new_name is None:
            new_name = filename

        print(f"  {filename}  ->  {destination/new_name}")

        shutil.move(file, destination / new_name)

print("\nDone!")