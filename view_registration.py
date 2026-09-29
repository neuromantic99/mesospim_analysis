"""Edit the settings below, then run:  uv run view_registration.py

Opens the atlas labels over the volume they were projected onto, with the voxel scale set so
the brain is not rendered five times too deep.
"""

from pathlib import Path

from mesospim_analysis.viewing import RegistrationView, open_registration

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

DRIVE = Path("/Volumes/JR-cache/Mesospim")

# The best N027 registration: princeton atlas at bending energy 0.5. It beat the same atlas at
# 0.95 on dice (0.9530 vs 0.9462), on surface distance (0.119 vs 0.131 mm) and on olfactory
# bulb coverage (57% vs 38%), for 0.38% folded tissue against 0.13%.
RUN = DRIVE / "brainreg_bakeoff" / "princeton_mouse_20um"

RAW = DRIVE / "brainreg" / "downsampled_488 nm_pyramid_3.tif"
LABELS = RUN / "atlas_check_3regions.tif"  # or atlas_in_original_space.tif for all regions
PYRAMID_LEVEL = 3
CONTRAST: tuple[float, float] | None = (0.0, 1300.0)  # None = let napari choose

# ---------------------------------------------------------------------------------------------


def main() -> None:
    import napari  # noqa: PLC0415

    open_registration(
        RegistrationView(
            raw=RAW, labels=LABELS, pyramid_level=PYRAMID_LEVEL, contrast=CONTRAST
        )
    )
    napari.run()


if __name__ == "__main__":
    main()
