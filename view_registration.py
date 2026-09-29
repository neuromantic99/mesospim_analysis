"""Edit the settings below, then run:  uv run view_registration.py

Opens atlas regions over the volume they were projected onto, with the voxel scale set so the
brain is not rendered five times too deep.
"""

from pathlib import Path

from mesospim_analysis.viewing import RegistrationView, open_registration, region_file

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

DRIVE = Path("/Volumes/JR-cache/Mesospim")

# The best N027 registration to date: princeton at bending energy 0.5. It beat the same atlas
# at 0.95 on dice (0.9530 vs 0.9462), surface distance (0.119 vs 0.131 mm) and olfactory bulb
# coverage (57% vs 38%). Those are all boundary measures though, so they say little about where
# internal structures land -- judge those by eye.
RUN = DRIVE / "brainreg_bakeoff" / "princeton_mouse_20um"
RAW = DRIVE / "brainreg" / "downsampled_488 nm_pyramid_3.tif"

# Regions to draw, by acronym, numbered in this order. Descendants are included, which matters:
# princeton labels CA1 with one id while the Allen labels its layers and barely uses the parent,
# so asking for a bare id gives a full field in one atlas and nothing in the other.
#   ["CA1"]                 one region, for comparing against another atlas
#   ["CP", "HPF", "CB"]     three unmistakable ones, for a quick sanity check
# HPF is the hippocampal formation, including subiculum and entorhinal cortex. HIP alone stops
# at CA1/CA3/DG and looks, correctly but confusingly, like half the structure is missing.
REGIONS = ["CA1"]

PYRAMID_LEVEL = 3
CONTRAST: tuple[float, float] | None = (0.0, 1300.0)  # None = let napari choose

# ---------------------------------------------------------------------------------------------


def main() -> None:
    import napari

    labels, values = region_file(RUN, REGIONS)
    print(f"{labels.name}: " + ", ".join(f"{k}={v}" for k, v in values.items()))

    viewer = open_registration(
        RegistrationView(
            raw=RAW, labels=labels, pyramid_level=PYRAMID_LEVEL, contrast=CONTRAST
        )
    )
    # The volume is (z=dorsoventral, y=anteroposterior, x=mediolateral), so scrolling axis 1
    # gives coronal sections. Default order scrolls axis 0, which gives horizontal ones.
    viewer.dims.order = (1, 0, 2)
    napari.run()


if __name__ == "__main__":
    main()
