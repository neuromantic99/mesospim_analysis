"""Edit the settings below, then run:  uv run main.py"""

from pathlib import Path
from typing import List

from mesospim_analysis.acquisitions import (
    build_aquisition_list,
    find_acquisitions,
    parse_acquisition,
    resolve_h5,
)
from mesospim_analysis.constants import DATA_ROOT
from mesospim_analysis.pipeline import SlabSummary
from mesospim_analysis.plotting import plot_depth_profiles
from mesospim_analysis.run import run_brain

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

THICKNESS_UM = 50.0
PYRAMID_LEVEL = 0  # count cells at level 0 only
SIGNAL_CHANNEL = "638 nm"
AF_CHANNEL = "561 nm"
"""561, not 488. The label is AF647 and 488 sits too far from it to predict its
autofluorescence: as a reference it left 234 false cells in a brain given no antibody at all,
where 561 left none, and no cutoff recovered it. Anything processed against 488 is not
comparable with the rest of the cohort."""

Z_RANGE_UM: tuple[float, float] | None = None  # e.g. (3000.0, 4000.0); None = whole brain
SAVE_BGSUB = False  # also write the background-subtracted (not AF-corrected) stack
SAVE_OBJECTS = True  # one csv row per detected object (roughly doubles runtime)
OUT_DIR: Path | None = None  # None = afcorr/ next to each stitched.h5
PLOT_PATH: Path | None = Path("depth_profiles.png")  # summary figure; None = skip

# ---------------------------------------------------------------------------------------------


def main(acquisitions: List[Path]) -> None:
    summaries: dict[str, list[SlabSummary]] = {}
    for acquisition in acquisitions:
        summaries[parse_acquisition(resolve_h5(acquisition)).name] = run_brain(
            acquisition,
            thickness_um=THICKNESS_UM,
            pyramid_level=PYRAMID_LEVEL,
            signal=SIGNAL_CHANNEL,
            af=AF_CHANNEL,
            out_dir=OUT_DIR,
            z_range_um=Z_RANGE_UM,
            save_bgsub=SAVE_BGSUB,
            save_objects=SAVE_OBJECTS,
        )
    if PLOT_PATH is not None:
        print(f"wrote {plot_depth_profiles(summaries, PLOT_PATH)}")


if __name__ == "__main__":
    acquisitions = [
        DATA_ROOT / "2026-05-18/N027/001",  # real, stain worked (1185/MP on 561)
        DATA_ROOT / "2026-05-18/N030/001",  # no-antibody control (5/MP on 561)
    ]

    # acquisitions = build_aquisition_list()
    main(acquisitions)
