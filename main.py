"""Edit the settings below, then run:  uv run main.py"""

from pathlib import Path

from mesospim_analysis.run import run_brain

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

DATA_ROOT = Path("/home/james/mnt/MarcBusche/James/Mesospim")

# Acquisition folders (yyyy-mm-dd/MOUSE_ID/IMAGING-NUM) or stitched.h5 files to process.
# To process every acquisition of a mouse instead:
#   from mesospim_analysis.acquisitions import find_acquisitions
#   ACQUISITIONS = [a.h5_path for a in find_acquisitions(DATA_ROOT, mouse_id="N027")]
ACQUISITIONS = [
    DATA_ROOT / "2026-05-18/N027/001",  # stained
    DATA_ROOT / "2026-05-18/N030/001",  # secondary-only control
]

THICKNESS_UM = 50.0
PYRAMID_LEVEL = 0  # count cells at level 0 only
SIGNAL_CHANNEL = "638 nm"
AF_CHANNEL = "561 nm"
Z_RANGE_UM: tuple[float, float] | None = None  # e.g. (3000.0, 4000.0); None = whole brain
SAVE_BGSUB = False  # also write the background-subtracted (not AF-corrected) stack
OUT_DIR: Path | None = None  # None = afcorr/ next to each stitched.h5

# ---------------------------------------------------------------------------------------------


def main() -> None:
    for acquisition in ACQUISITIONS:
        run_brain(
            acquisition,
            thickness_um=THICKNESS_UM,
            pyramid_level=PYRAMID_LEVEL,
            signal=SIGNAL_CHANNEL,
            af=AF_CHANNEL,
            out_dir=OUT_DIR,
            z_range_um=Z_RANGE_UM,
            save_bgsub=SAVE_BGSUB,
        )


if __name__ == "__main__":
    main()
