"""Atlas registration: preparing a volume for brainreg, and reading its output back.

brainreg registers a downsampled autofluorescence channel to an atlas. Three things about
this data had to be established empirically and are encoded here.

Masking the brain is not optional. In N027 only 33% of the frame is brain and 59% is the
stitched tile footprint and light-sheet striping, at 22% of brain brightness. NiftyReg's
mutual information is computed over the whole frame, so unmasked it aligns the field of view
rather than the brain: Dice against the atlas was 0.813 unmasked and 0.921 masked.

Clipping matters for the same reason. NiftyReg spreads 128 histogram bins over the full
intensity range, so a few hot voxels collapse the tissue into a couple of bins. In N027, 90%
of tissue sat in 14 of 128 bins; clipping at the 99.9th percentile spread it over 71.

brainreg works in the atlas's orientation at the atlas's resolution, so its outputs have to be
mapped back before they can be used with objects detected in the original volume.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from mesospim_analysis.types import BoolImage, FloatImage

IntImage = npt.NDArray[np.integer]

CLIP_PERCENTILE = 99.9
"""Chosen against the histogram-bin occupancy above, not tuned for detection."""

MASK_DILATION = 1
"""One voxel, to keep the dim pial surface that the Otsu threshold cuts into."""

MIN_MASK_FRACTION = 0.05
MAX_MASK_FRACTION = 0.90
"""A brain filled 34% of the frame in the validated data. Outside this range the threshold has
found something other than the brain and the caller should look before registering."""


@dataclass(frozen=True)
class PreparedVolume:
    volume: IntImage
    """Masked and clipped, ready to write out and hand to brainreg."""
    mask: BoolImage
    clip_value: float
    threshold: float

    @property
    def mask_fraction(self) -> float:
        return float(self.mask.mean())

    @property
    def looks_like_a_brain(self) -> bool:
        return MIN_MASK_FRACTION <= self.mask_fraction <= MAX_MASK_FRACTION


def _otsu(values: FloatImage) -> float:
    return float(threshold_otsu(values))  # type: ignore[no-untyped-call]


def _mask_and_threshold(
    volume: FloatImage, subsample: int, clip_percentile: float = CLIP_PERCENTILE
) -> tuple[BoolImage, float]:
    """The largest solid bright object, i.e. the brain rather than the illuminated background.

    Thresholding is done on a subsampled copy because the morphology is the slow part and the
    mask is smooth at this scale; the result is expanded back to full size nearest-neighbour.

    Otsu sees a percentile-capped copy. It maximises between-class variance, so a handful of
    very bright voxels pull the threshold up towards them and the mask collapses to those
    voxels alone. The real data survived this only because its outliers were mild.
    """
    cube = np.ones((3, 3, 3), dtype=np.bool_)
    small = np.asarray(volume[::subsample, ::subsample, ::subsample], dtype=np.float32)
    finite = small[np.isfinite(small) & (small > 0)]
    if finite.size == 0:
        raise ValueError("volume is empty; nothing to mask")
    threshold = _otsu(np.minimum(finite, np.percentile(finite, clip_percentile)))

    mask = small > threshold
    mask = ndi.binary_opening(mask, cube)
    labels, count = ndi.label(mask)
    if count == 0:
        raise ValueError("no object survived opening; the threshold found no brain")
    if count > 1:
        sizes = ndi.sum(mask, labels, range(1, count + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)
    mask = ndi.binary_closing(mask, np.ones((5, 5, 5), dtype=np.bool_))
    mask = ndi.binary_fill_holes(mask)
    if MASK_DILATION:
        mask = ndi.binary_dilation(mask, cube, iterations=MASK_DILATION)

    if subsample == 1:
        return np.asarray(mask, dtype=np.bool_), threshold
    grown = ndi.zoom(mask.astype(np.uint8), subsample, order=0).astype(bool)
    full = np.zeros(volume.shape, dtype=np.bool_)
    z, y, x = (min(a, b) for a, b in zip(volume.shape, grown.shape, strict=True))
    full[:z, :y, :x] = grown[:z, :y, :x]
    return full, threshold


def brain_mask(volume: FloatImage, subsample: int = 2) -> BoolImage:
    """The brain, excluding the illuminated background and the stitched tile footprint."""
    mask, _ = _mask_and_threshold(volume, subsample)
    return mask


def prepare_for_registration(
    volume: IntImage, clip_percentile: float = CLIP_PERCENTILE, subsample: int = 2
) -> PreparedVolume:
    """Mask the brain and clip outliers, leaving everything else untouched.

    The output keeps the input's dtype and geometry so brainreg is given the same voxel sizes
    as the original.
    """
    as_float: FloatImage = np.asarray(volume, dtype=np.float32)
    mask, threshold = _mask_and_threshold(as_float, subsample)
    inside = as_float[mask]
    if inside.size == 0:
        raise ValueError("brain mask is empty")
    clip_value = float(np.percentile(inside, clip_percentile))
    prepared = np.where(mask, np.minimum(volume, clip_value), 0).astype(volume.dtype)
    return PreparedVolume(
        volume=prepared,
        mask=mask,
        clip_value=clip_value,
        threshold=threshold,
    )


def histogram_occupancy(volume: IntImage, mask: BoolImage, bins: int = 128) -> int:
    """Bins holding 90% of the tissue, out of `bins`. NiftyReg's similarity sees this many.

    A diagnostic for whether clipping was worth it: 14 before and 71 after, in N027.
    """
    inside = np.asarray(volume[mask], dtype=np.float64)
    if inside.size == 0 or inside.max() <= 0:
        return 0
    counts = np.bincount(
        np.floor(inside / inside.max() * (bins - 1)).astype(np.int64), minlength=bins
    )
    ordered = np.sort(counts)[::-1]
    return int(np.searchsorted(np.cumsum(ordered) / counts.sum(), 0.90) + 1)


@dataclass(frozen=True)
class SourceGeometry:
    """How a brainreg run maps back onto the volume it was given.

    brainreg reorients the input to the atlas's orientation and resamples it to the atlas's
    resolution, so its outputs are in neither the orientation nor the sampling of the original.
    `data_orientation` "ial" with an atlas in "asr" -- the case validated here -- means axis 1
    of the source becomes the atlas's first axis, and the source's other two axes are reversed.
    """

    shape: tuple[int, int, int]
    """The original volume's shape, (z, y, x)."""

    def plane_from_atlas_space(self, registered: IntImage, z: int) -> IntImage:
        """The atlas labels for plane `z` of the original volume.

        `registered` must already be resampled in-plane to the original's y and x.
        """
        nz = self.shape[0]
        forward = np.linspace(0, nz - 1, registered.shape[1]).round().astype(np.int64)
        source = int(np.abs(forward - (nz - 1 - z)).argmin())
        return np.ascontiguousarray(registered[:, source, ::-1])


def resample_in_plane(registered: IntImage, shape: tuple[int, int, int]) -> IntImage:
    """Atlas labels resampled to the original's in-plane size, nearest-neighbour.

    Region ids are categorical, so interpolating them would invent regions that do not exist.
    """
    _, ny, nx = shape
    zoomed: IntImage = ndi.zoom(
        registered,
        (ny / registered.shape[0], 1.0, nx / registered.shape[2]),
        order=0,
        prefilter=False,
    )
    return zoomed


def region_subset(
    labels: IntImage, groups: dict[int, npt.NDArray[np.integer]]
) -> IntImage:
    """Collapse a label volume to a handful of regions, numbered 1..n.

    For checking a registration by eye: a few large, unmistakable regions read far more clearly
    than several hundred overlapping ones.
    """
    out = np.zeros(labels.shape, dtype=np.uint8)
    for value, ids in groups.items():
        out[np.isin(labels, ids)] = value
    return out


@dataclass(frozen=True)
class RegionScore:
    acronym: str
    atlas_intensity: float
    sample_intensity: float
    covered: float

    @property
    def ratio(self) -> float:
        return self.sample_intensity / self.atlas_intensity


def score_regions(
    warped: FloatImage,
    annotation: IntImage,
    reference: FloatImage,
    regions: dict[str, npt.NDArray[np.integer]],
) -> list[RegionScore]:
    """Compare each region's autofluorescence with the same region in the atlas's reference.

    Both are autofluorescence, so a correctly placed label should show the same brightness
    relative to the whole brain as the atlas does, and a ratio near 1 is what a good
    registration looks like.

    What this cannot do is separate alignment from tissue. Two runs of the same brain with very
    different deformation freedom gave the same median deviation of 0.15, and the run-to-run
    spread on identical settings is 0.013 -- larger than the gap between two candidate atlases.
    Most of that deviation is the atlas's clearing protocol differing from this one. Use it for
    gross failures, such as a region with almost no coverage, and for ranking runs of the same
    brain; do not read a 10% deviation as a 10% misalignment.
    """
    inside = annotation > 0
    got = inside & (warped > 0)
    if not got.any():
        raise ValueError("the warped volume and the annotation do not overlap")
    reference_norm = reference / reference[inside].mean()
    warped_norm = warped / warped[got].mean()

    scores = []
    for acronym, ids in regions.items():
        region = np.isin(annotation, ids)
        present = region & (warped > 0)
        if not region.any() or not present.any():
            continue
        scores.append(
            RegionScore(
                acronym=acronym,
                atlas_intensity=float(reference_norm[region].mean()),
                sample_intensity=float(warped_norm[present].mean()),
                covered=float(present.sum() / region.sum()),
            )
        )
    return scores


def surface_distance_mm(
    sample: BoolImage, atlas: BoolImage, resolution_um: float
) -> tuple[float, float]:
    """Symmetric mean and 95th-percentile distance between the two surfaces, in millimetres.

    Reported in millimetres rather than as voxel overlap because Dice rises with coarser
    voxels: a 25 um atlas scored a better Dice than a 20 um one that was in fact closer.
    """
    sample_edge = sample ^ ndi.binary_erosion(sample)
    atlas_edge = atlas ^ ndi.binary_erosion(atlas)
    if not sample_edge.any() or not atlas_edge.any():
        raise ValueError("one of the masks has no surface")
    to_atlas = ndi.distance_transform_edt(~atlas_edge) * resolution_um / 1000.0
    to_sample = ndi.distance_transform_edt(~sample_edge) * resolution_um / 1000.0
    both = np.concatenate([to_atlas[sample_edge], to_sample[atlas_edge]])
    return float(both.mean()), float(np.percentile(both, 95))


def dice(a: BoolImage, b: BoolImage) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2 * float((a & b).sum()) / total if total else float("nan")


def region_ids(structures_csv: Path, acronym: str) -> npt.NDArray[np.integer]:
    """Every id belonging to a region, including its descendants.

    Atlases differ in how finely they subdivide. Princeton labels CA1 with the single id 382,
    while the Allen labels its layers and leaves 382 itself almost unused -- so asking for one
    id returns a full field in one atlas and nearly nothing in the other. Always take the
    descendants.
    """
    with open(structures_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    parents = [row for row in rows if row["acronym"] == acronym]
    if not parents:
        raise KeyError(f"{acronym} is not in {structures_csv.name}")
    parent_id = parents[0]["id"]
    ids = {int(parent_id)} | {
        int(row["id"]) for row in rows if f"/{parent_id}/" in row["structure_id_path"]
    }
    return np.array(sorted(ids))


def write_region_subset(
    full_labels: Path, groups: dict[int, npt.NDArray[np.integer]], out: Path
) -> dict[int, int]:
    """Stream a few regions out of a full label volume, numbered 1..n.

    Plane by plane, because the full volume is 1.5 GB of uint32 and the point is to end up
    with something small enough to page through in a viewer.
    """
    source = tifffile.memmap(full_labels)
    counts = dict.fromkeys(groups, 0)
    with tifffile.TiffWriter(out, bigtiff=True) as writer:
        for index in range(source.shape[0]):
            plane = np.asarray(source[index])
            selected = np.zeros(plane.shape, dtype=np.uint8)
            for value, ids in groups.items():
                hit = np.isin(plane, ids)
                selected[hit] = value
                counts[value] += int(hit.sum())
            writer.write(selected, contiguous=True)
    return counts
