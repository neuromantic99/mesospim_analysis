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
from mesospim_analysis.run import run_brain, run_slab

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

THICKNESS_UM = 50.0
PYRAMID_LEVEL = 0  # count cells at level 0 only
SIGNAL_CHANNEL = "638 nm"
AF_CHANNEL = "488 nm"
Z_RANGE_UM: tuple[float, float] | None = (
    None  # e.g. (3000.0, 4000.0); None = whole brain
)
SAVE_BGSUB = False  # also write the background-subtracted (not AF-corrected) stack
SAVE_OBJECTS = True  # one csv row per detected object (roughly doubles runtime)
# OUT_DIR: Path | None = None  # None = afcorr/ next to each stitched.h5
OUT_DIR = DATA_ROOT / "afcorr_488"
# PLOT_PATH: Path | None = Path(
#     "depth_profiles.png"

# )  # summary figure of all brains; None = skip
PLOT_PATH = Path("depth_profiles_488.png")

# ---------------------------------------------------------------------------------------------


def main(acquisitions: List[Path]) -> None:
    summaries: dict[str, list[SlabSummary]] = {}
    for acquisition in acquisitions:
        run_slab(
            acquisition,
            z_start_um=3800.0,
            af="488 nm",
            out_dir=DATA_ROOT / "afcorr_488",
        )
        # summaries[parse_acquisition(resolve_h5(acquisition)).name] = run_brain(
        #     acquisition,
        #     thickness_um=THICKNESS_UM,
        #     pyramid_level=PYRAMID_LEVEL,
        #     signal=SIGNAL_CHANNEL,
        #     af=AF_CHANNEL,
        #     out_dir=OUT_DIR,
        #     z_range_um=Z_RANGE_UM,
        #     save_bgsub=SAVE_BGSUB,
        #     save_objects=SAVE_OBJECTS,
        # )
    # if PLOT_PATH is not None:
    #     print(f"wrote {plot_depth_profiles(summaries, PLOT_PATH)}")


if __name__ == "__main__":
    acquisitions = [
        DATA_ROOT / "2026-05-18/N027/001",  # real, stain worked (1185/MP on 561)
        DATA_ROOT / "2026-05-18/N030/001",  # no-antibody control (5/MP on 561)
    ]

    # acquisitions = build_aquisition_list()
    main(acquisitions)
