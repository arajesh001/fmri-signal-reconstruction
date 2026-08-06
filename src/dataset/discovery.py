"""
discovery.py

Searches the filesystem and creates Scan objects.
"""

from pathlib import Path

from config import (
    HAND_ROOT,
    HAPPY_ROOT,
    MOTION_ROOT,
    GROUND_TRUTH_ROOT,
)

from .scan import Scan

# ==========================================================
# PUBLIC API
# ==========================================================

def discover_scans() -> list[Scan]:
    """
    Discover every valid scan in the dataset.
    """

    scans = []

    for subject_dir in sorted(HAND_ROOT.glob("sub-*")):

        if not subject_dir.is_dir():
            continue

        for session_dir in sorted(subject_dir.glob("ses-*")):

            if not session_dir.is_dir():
                continue

            try:
                scan = build_scan(subject_dir, session_dir)

                if scan.validate():
                    scans.append(scan)

            except FileNotFoundError:
                # Missing required raw inputs
                continue

    return scans


# ==========================================================
# BUILD SCAN
# ==========================================================

def build_scan(
    subject_dir: Path,
    session_dir: Path,
) -> Scan:
    """
    Build one Scan object from one subject/session.
    """

    subject = subject_dir.name
    session = session_dir.name

    # --------------------------------------------------
    # Raw inputs
    # --------------------------------------------------

    bold = find_required(session_dir, "*echo-1_bold.nii.gz")
    bold_json = find_required(session_dir, "*echo-1_bold.json")

    # --------------------------------------------------
    # Optional files
    # --------------------------------------------------

    motion = find_file(
        MOTION_ROOT / subject / session,
        "*.1D",
    )

    ground_truth = find_file(
        GROUND_TRUTH_ROOT,
        f"{subject}_{session}*.txt",
    )

    # --------------------------------------------------
    # HAPPY outputs
    # --------------------------------------------------

    happy_session = HAPPY_ROOT / subject / session

    cardpulse_json, cardpulse_tsv = find_happy_pair(
        happy_session / "cardpulse"
    )

    sliceres_json, sliceres_tsv = find_happy_pair(
        happy_session / "sliceres"
    )

    stdres_json, stdres_tsv = find_happy_pair(
        happy_session / "stdres"
    )

    vessels_dir = happy_session / "vessels"

    vessel_map = find_file(vessels_dir, "map.nii.gz")
    vessel_mask = find_file(vessels_dir, "mask.nii.gz")
    vessel_metadata = find_file(vessels_dir, "metadata.json")

    # --------------------------------------------------
    # Construct Scan
    # --------------------------------------------------

    return Scan(
        subject=subject,
        session=session,
        bold=bold,
        bold_json=bold_json,
        motion=motion,
        ground_truth=ground_truth,
        cardpulse_tsv=cardpulse_tsv,
        cardpulse_json=cardpulse_json,
        sliceres_tsv=sliceres_tsv,
        sliceres_json=sliceres_json,
        stdres_tsv=stdres_tsv,
        stdres_json=stdres_json,
        vessel_map=vessel_map,
        vessel_mask=vessel_mask,
        vessel_metadata=vessel_metadata,
    )


# ==========================================================
# HELPERS
# ==========================================================

def find_file(directory: Path, pattern: str) -> Path | None:
    """
    Return the first file matching a pattern.
    Returns None if nothing matches.
    """

    if not directory.exists():
        return None

    matches = sorted(directory.glob(pattern))

    return matches[0] if matches else None


def find_happy_pair(folder: Path) -> tuple[Path | None, Path | None]:
    """
    Return

        metadata.json
        timeseries.tsv.gz
    """

    metadata = find_file(folder, "metadata.json")
    timeseries = find_file(folder, "timeseries.tsv.gz")

    return metadata, timeseries


def find_required(directory: Path, pattern: str) -> Path:
    """
    Find a required file.
    Raises FileNotFoundError if missing.
    """

    path = find_file(directory, pattern)

    if path is None:
        raise FileNotFoundError(
            f"Missing required file: {pattern}"
        )

    return path