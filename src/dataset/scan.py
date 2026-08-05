"""
scan.py

Represents one fMRI scan (one subject + one session).

This class only knows where files are located.

"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scan:
    """
    Represents one complete scan.

    Example
    -------
    Subject: sub-01
    Session: ses-03
    """

    # --------------------------------------------------
    # Identity
    # --------------------------------------------------

    subject: str
    session: str

    # --------------------------------------------------
    # Raw fMRI inputs
    # --------------------------------------------------

    bold: Path
    bold_json: Path

    motion: Path | None
    ground_truth: Path | None

    # --------------------------------------------------
    # HAPPY outputs
    # --------------------------------------------------

    cardpulse_tsv: Path
    cardpulse_json: Path

    sliceres_tsv: Path
    sliceres_json: Path

    stdres_tsv: Path
    stdres_json: Path

    vessel_map: Path
    vessel_mask: Path
    vessel_metadata: Path

    # ==================================================
    # FUNCTIONS
    # ==================================================

    def missing_files(self) -> list[Path]:
        """
        Return a list of missing required files.
        """

        required = [
            self.bold,
            self.bold_json,
            self.cardpulse_tsv,
            self.cardpulse_json,
            self.sliceres_tsv,
            self.sliceres_json,
            self.stdres_tsv,
            self.stdres_json,
            self.vessel_map,
            self.vessel_mask,
            self.vessel_metadata,
        ]

        return [path for path in required if not path.exists()]

    def validate(self) -> bool:
        """
        Returns True if all required files exist.
        """
        return len(self.missing_files()) == 0

    def has_motion(self) -> bool:
        """
        Returns True if motion regressors exist.
        """
        return self.motion is not None and self.motion.exists()

    def has_ground_truth(self) -> bool:
        """
        Returns True if ground truth heart-rate data exists.
        """
        return self.ground_truth is not None and self.ground_truth.exists()

    def summary(self) -> str:
        """
        Return a readable summary of the scan.
        """

        motion = "Yes" if self.has_motion() else "No"
        ground_truth = "Yes" if self.has_ground_truth() else "No"

        return (
            f"Subject      : {self.subject}\n"
            f"Session      : {self.session}\n"
            f"Motion       : {motion}\n"
            f"Ground Truth : {ground_truth}\n"
            f"Valid Scan   : {self.validate()}"
        )

    def __repr__(self) -> str:
        """
        Short representation for debugging.
        """
        return f"Scan(subject='{self.subject}', session='{self.session}')"

    @property
    def id(self) -> str:
        return f"{self.subject}_{self.session}"