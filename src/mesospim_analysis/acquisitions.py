"""Locating acquisitions laid out as yyyy-mm-dd/MOUSE_ID/IMAGING-NUM/stitched.h5."""

import datetime
from dataclasses import dataclass
from importlib.metadata import metadata
from pathlib import Path
import re

import numpy as np

from mesospim_analysis.constants import AFCORR_DIRNAME, DATA_ROOT, RESULTS_DIR
from mesospim_analysis.utils import read_stitched_metadata

STITCHED_FILENAME = "stitched.h5"


@dataclass(frozen=True)
class Acquisition:
    date: datetime.date
    mouse_id: str
    imaging_number: str | None
    """None when the imaging-number level is absent (yyyy-mm-dd/MOUSE_ID/stitched.h5)."""
    h5_path: Path

    @property
    def name(self) -> str:
        parts = [self.date.isoformat(), self.mouse_id]
        if self.imaging_number is not None:
            parts.append(self.imaging_number)
        return "_".join(parts)


def resolve_h5(path: Path) -> Path:
    """Accept either stitched.h5 itself or the directory containing it."""
    return path / STITCHED_FILENAME if path.is_dir() else path


def parse_acquisition(h5_path: Path) -> Acquisition:
    parent = h5_path.parent
    imaging_number: str | None = None
    if parent.name.isdigit():
        imaging_number = parent.name
        parent = parent.parent
    mouse_id = parent.name
    try:
        date = datetime.date.fromisoformat(parent.parent.name)
    except ValueError as e:
        raise ValueError(
            f"{h5_path}: expected yyyy-mm-dd/MOUSE_ID/IMAGING-NUM/{STITCHED_FILENAME}"
        ) from e
    return Acquisition(date, mouse_id, imaging_number, h5_path)


def find_acquisitions(root: Path, mouse_id: str | None = None) -> list[Acquisition]:
    """All acquisitions under `root`, sorted by date, mouse and imaging number."""
    candidates = [
        *root.glob(f"*/*/*/{STITCHED_FILENAME}"),
        *root.glob(f"*/*/{STITCHED_FILENAME}"),
    ]
    acquisitions: list[Acquisition] = []
    for path in candidates:
        try:
            acquisition = parse_acquisition(path)
        except ValueError:
            continue
        if mouse_id is None or acquisition.mouse_id == mouse_id:
            acquisitions.append(acquisition)
    return sorted(
        acquisitions, key=lambda a: (a.date, a.mouse_id, a.imaging_number or "")
    )


def build_aquisition_list() -> list[Path]:
    all_mice = [f"N{n:03d}" for n in range(1, 66)]

    to_analyse: list[Path] = []

    for mouse in all_mice:
        acquisitions = find_acquisitions(DATA_ROOT, mouse_id=mouse)
        for acquisition in acquisitions:
            metadata = read_stitched_metadata(acquisition.h5_path)
            if "561 nm" in metadata.channel_setups:
                to_analyse.append(acquisition.h5_path)

    return to_analyse


def copy_results() -> None:

    import shutil

    dates = DATA_ROOT.glob("*/")

    all_csvs = []
    for date in dates:
        mice = date.glob("*/")
        for mouse in mice:
            imaging_numbers = mouse.glob("*/")
            for imaging_number in imaging_numbers:
                if not (imaging_number / AFCORR_DIRNAME).exists():
                    continue
                csvs = imaging_number.glob(f"{AFCORR_DIRNAME}/*.csv")
                all_csvs.extend(list(csvs))

    destination = RESULTS_DIR
    for csv in all_csvs:
        destination_path = destination / csv.name
        # A run truncates its output at startup and buffers, so a file that is still being
        # written (or whose run died early) is empty. Copying that over a finished result
        # loses it, which has happened.
        if csv.stat().st_size == 0:
            print(f"skipping empty {csv.name} (run in progress or failed)")
            continue
        if (
            destination_path.exists()
            and destination_path.stat().st_size > csv.stat().st_size
        ):
            print(
                f"skipping {csv.name}: destination is larger "
                f"({destination_path.stat().st_size} > {csv.stat().st_size} bytes); "
                "delete it by hand if the new run really is the one you want"
            )
            continue
        shutil.copy2(csv, destination_path)


if __name__ == "__main__":
    copy_results()
