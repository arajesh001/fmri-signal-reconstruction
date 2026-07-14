from pathlib import Path
import pandas as pd
import nibabel as nib
import json
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)

DATA_DIR = PROJECT_ROOT / config["paths"]["raw_data"]
OUTPUT_DIR = PROJECT_ROOT / config["paths"]["output"]
task = config["dataset"]["task"]
echo = config["dataset"]["echo"]

# ==========================================================
# STORAGE
# One dict --> one row in the dataframe
# ==========================================================

rows = []

for subject_dir in DATA_DIR.iterdir():

    # SKIP EVERYTHING ELSE
    if not subject_dir.is_dir():
        continue
    if subject_dir.name == "hand":
        continue
    if subject_dir.name == "processed":
        continue
    if subject_dir.name == "raw":
        continue

    for session_dir in subject_dir.iterdir():

        if not session_dir.is_dir():
            continue

        bold_file = next(session_dir.glob("*bold.nii.gz"), None)
        json_file = next(session_dir.glob("*bold.json"), None)

        # incomplete --> SKIP
        if bold_file is None or json_file is None:
            continue

        # BOLD 
        img = nib.load(bold_file)
        shape = img.shape

        # TO FIX ERROR FROM BEFORE WITH non 4D files
        if len(shape) != 4:
            print(f"Unexpected shape: {shape} ({bold_file})")
            continue

        zooms = img.header.get_zooms()

        x = shape[0]
        y = shape[1]
        z = shape[2]
        volumes = shape[3]
        voxel_x = zooms[0]
        voxel_y = zooms[1]
        voxel_z = zooms[2]

        # JSON
        with open(json_file) as f:
            metadata = json.load(f)

        tr = metadata.get("RepetitionTime")
        echo_time = metadata.get("EchoTime")
        echo_number = metadata.get("EchoNumber")
        flip_angle = metadata.get("FlipAngle")
        slice_thickness = metadata.get("SliceThickness")
        phase_encoding = metadata.get("PhaseEncodingDirection")
        manufacturer = metadata.get("Manufacturer")
        scanner_model = metadata.get("ManufacturersModelName")
        magnetic_field = metadata.get("MagneticFieldStrength")

        # Store one row
        rows.append({

            "Subject": subject_dir.name,
            "Session": session_dir.name,

            "X": x,
            "Y": y,
            "Z": z,
            "Volumes": volumes,
            "Shape": str(shape),

            "Voxel X": voxel_x,
            "Voxel Y": voxel_y,
            "Voxel Z": voxel_z,

            "TR": tr,
            "Echo Time": echo_time,
            "Echo Number": echo_number,
            "Flip Angle": flip_angle,
            "Slice Thickness": slice_thickness,
            "Phase Encoding": phase_encoding,

            "Manufacturer": manufacturer,
            "Scanner Model": scanner_model,
            "Magnetic Field": magnetic_field,

            "BOLD Path": str(bold_file.relative_to(PROJECT_ROOT)),
            "JSON Path": str(json_file.relative_to(PROJECT_ROOT)),

            "Status": "Ready"
        })

# Convert to df and save
df = pd.DataFrame(rows)
print(df.head())

OUTPUT = PROJECT_ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
df.to_csv(OUTPUT / "dataset_metadata.csv", index=False)