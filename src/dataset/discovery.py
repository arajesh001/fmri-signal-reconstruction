"""
discovery.py

Searches the filesystem and creates Scan objects.

"""

from pathlib import Path

from config import HAND_ROOT, HAPPY_ROOT

from .scan import Scan

# ==========================================================
# PUBLIC API
# ==========================================================

def discover_scans() -> list[Scan]:
    """
    Discover every subject/session pair under HAND_ROOT and build a
    Scan object for each one.
    """

    scans = []

    for subject_dir in sorted(HAND_ROOT.glob("sub-*")):

        if not subject_dir.is_dir():
            continue

        for session_dir in sorted(subject_dir.glob("ses-*")):

            if not session_dir.is_dir():
                continue

            scans.append(build_scan(subject_dir, session_dir))

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
    # Raw bold + physio (all together in session_dir)
    # --------------------------------------------------

    bold = find_file(session_dir, "*_bold.nii.gz")
    bold_json = find_file(session_dir, "*_bold.json")
    motion = find_file(session_dir, "motion.1D")
    ground_truth = find_file(session_dir, "hr.txt")

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

    vessel_dir = happy_session / "vessel"

    vessel_map = find_file(vessel_dir, "map.nii.gz")
    vessel_mask = find_file(vessel_dir, "mask.nii.gz")

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
    )


# ==========================================================
# HELPERS
# ==========================================================

def find_file(directory: Path, pattern: str) -> Path | None:
    """
    Return the first file matching a pattern.
    Returns None if nothing matches (never raises).
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