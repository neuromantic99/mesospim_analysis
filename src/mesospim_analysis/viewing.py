"""Opening a registration in napari with the right voxel scale.

The volume is strongly anisotropic: 5 um in z against 26.08 um in plane at pyramid level 3.
Loaded without a scale, napari treats the voxels as cubic and renders the brain roughly five
times too deep. Thin curved structures suffer most -- the hippocampus looks badly misplaced and
the registration looks broken when it is not, which cost an afternoon.

napari is not a dependency of this package. It is imported inside the function so that
everything else keeps working without it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

Z_STEP_UM = 5.0
PIXEL_SIZE_UM = 3.26
"""Pyramid level 0. Level 3 is 8x in plane and leaves z untouched, so 26.08 / 26.08 / 5.0."""


def level_scale(pyramid_level: int) -> tuple[float, float, float]:
    """(z, y, x) voxel size in microns at a pyramid level, for napari's `scale`.

    BigStitcher's levels here downsample in plane only, so z keeps its step at every level.
    """
    if pyramid_level < 0:
        raise ValueError(f"pyramid level must be >= 0, got {pyramid_level}")
    in_plane = PIXEL_SIZE_UM * 2**pyramid_level
    return (Z_STEP_UM, in_plane, in_plane)


@dataclass(frozen=True)
class RegistrationView:
    raw: Path
    """The channel that was registered, in its original geometry."""
    labels: Path
    """Atlas labels projected back onto that geometry."""
    pyramid_level: int = 3
    contrast: tuple[float, float] | None = None
    label_opacity: float = 0.4
    z_step: int = 4
    """Load every nth plane. z is sampled at 5 um against 26.08 um in plane, so 4 gives roughly
    isotropic 20 um: the right trade for looking at a registration, and a quarter the memory."""
    in_memory: bool = True
    """Read the volumes rather than memory-mapping them.

    A tiff stores its pages along the first axis, so a coronal view -- which napari gives by
    slicing axis 1 -- touches every page to build one section. Memory-mapped from an external
    disk that is unusably slow; in memory it is instant. With z_step 4 the pair is ~280 MB.
    """

    def check(self) -> tuple[int, ...]:
        """Both volumes must exist and share a shape, or the overlay is meaningless."""
        for path in (self.raw, self.labels):
            if not path.exists():
                raise FileNotFoundError(path)
        with tifffile.TiffFile(self.raw) as handle:
            raw_shape = tuple(handle.series[0].shape)
        with tifffile.TiffFile(self.labels) as handle:
            label_shape = tuple(handle.series[0].shape)
        if raw_shape != label_shape:
            raise ValueError(
                f"shapes differ: {self.raw.name} is {raw_shape} but "
                f"{self.labels.name} is {label_shape}; the labels were projected onto a "
                "different volume"
            )
        if len(raw_shape) != 3:
            raise ValueError(f"expected a 3D volume, got {raw_shape}")
        return raw_shape


def _load(path: Path, step: int, in_memory: bool) -> Any:
    """Every `step`th plane, materialised only if asked for.

    Memory-mapping first, because whether a stack is one contiguous 3D page or one page per
    plane depends on how it was written, and only the memmap presents the right shape either
    way. tifffile's `key` argument indexes the series rather than the pages, so passing a slice
    there quietly returns the whole volume.
    """
    volume: Any
    try:
        volume = tifffile.memmap(path)
    except (ValueError, MemoryError):
        volume = tifffile.imread(path)  # compressed or otherwise not mappable
    subsampled = volume[::step]
    return np.array(subsampled) if in_memory else subsampled


def open_registration(view: RegistrationView) -> Any:
    """Show the labels over the volume they were projected onto, correctly scaled."""
    import napari  # noqa: PLC0415  -- optional, and slow to import

    view.check()
    z, y, x = level_scale(view.pyramid_level)
    scale = (z * view.z_step, y, x)
    raw = _load(view.raw, view.z_step, view.in_memory)
    labels = _load(view.labels, view.z_step, view.in_memory)
    print(f"loaded {raw.shape} at {scale[0]:.1f} / {scale[1]:.2f} / {scale[2]:.2f} um "
          f"({(raw.nbytes + labels.nbytes) / 1e6:.0f} MB)")
    viewer = napari.Viewer()
    viewer.add_image(
        raw,
        name=view.raw.stem,
        scale=scale,
        colormap="gray",
        contrast_limits=view.contrast,
    )
    viewer.add_labels(
        labels,
        name=view.labels.stem,
        scale=scale,
        opacity=view.label_opacity,
    )
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    return viewer


def region_file(
    run: Path, regions: Sequence[str], full_labels: str = "atlas_in_original_space.tif"
) -> tuple[Path, dict[str, int]]:
    """A small label volume holding just `regions`, built once and cached beside the run.

    Streaming a few regions out of the 1.5 GB full projection takes minutes on an external
    disk, so the result is kept: asking for the same regions again is instant.
    """
    from mesospim_analysis.registration import region_ids, write_region_subset

    if not regions:
        raise ValueError("name at least one region")
    structures = run / "structures.csv"
    values = {name: index for index, name in enumerate(regions, start=1)}
    out = run / f"atlas_subset_{'_'.join(regions)}.tif"
    if not out.exists():
        groups = {
            value: region_ids(structures, name) for name, value in values.items()
        }
        write_region_subset(run / full_labels, groups, out)
    return out, values
