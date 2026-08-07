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
    # Raw fMRI inputs --> optional
    # happy already consumed these on Quest, so 
    # NOT required for building dataset.
    # --------------------------------------------------

    bold: Path | None
    bold_json: Path | None

    # --------------------------------------------------
    # Physio --> required for dataset
    # --------------------------------------------------

    motion: Path | None
    ground_truth: Path | None

    # --------------------------------------------------
    # HAPPY outputs --> required for dataset
    # --------------------------------------------------

    cardpulse_tsv: Path | None
    cardpulse_json: Path | None

    sliceres_tsv: Path | None
    sliceres_json: Path | None

    stdres_tsv: Path | None
    stdres_json: Path | None

    # --------------------------------------------------
    # Vessel outputs --> not used by the cardiac-cleaning
    # model--> kept for completeness
    # --------------------------------------------------

    vessel_map: Path | None
    vessel_mask: Path | None

    # ==================================================
    # FUNCTIONS
    # ==================================================

    def missing_files(self) -> list[str]:
        """
        Return the names of files that are REQUIRED to build the ML
        dataset but are missing for this scan. bold/bold_json/vessel_*
        are intentionally excluded -- they're optional provenance, not
        required for training.
        """

        required = {
            "motion": self.motion,
            "ground_truth": self.ground_truth,
            "cardpulse_tsv": self.cardpulse_tsv,
            "cardpulse_json": self.cardpulse_json,
            "sliceres_tsv": self.sliceres_tsv,
            "sliceres_json": self.sliceres_json,
            "stdres_tsv": self.stdres_tsv,
            "stdres_json": self.stdres_json,
        }

        return [
            name for name, path in required.items()
            if path is None or not path.exists()
        ]



    def validate(self) -> bool:
        """
        Returns True if every file required to build the ML dataset
        exists for this scan.
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


    def has_bold(self) -> bool:
        """
        Returns True if the local raw bold copy exists (provenance
        only -- not required for the ML dataset).
        """
        return self.bold is not None and self.bold.exists()

    def summary(self) -> str:
        """
        Return a readable summary of the scan.
        """

        return (
            f"Subject      : {self.subject}\n"
            f"Session      : {self.session}\n"
            f"Motion       : {'Yes' if self.has_motion() else 'No'}\n"
            f"Ground Truth : {'Yes' if self.has_ground_truth() else 'No'}\n"
            f"Bold (local) : {'Yes' if self.has_bold() else 'No'}\n"
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

    @property
    def happy_dir(self) -> Path | None:
        if self.cardpulse_tsv is None:
            return None
        return self.cardpulse_tsv.parent.parent