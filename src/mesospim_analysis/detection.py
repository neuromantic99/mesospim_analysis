from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from mesospim_analysis.correction import REFERENCE_PIXEL_SIZE_UM, local_background, to_pixels
from mesospim_analysis.types import BoolImage, FloatImage

Labels = npt.NDArray[np.int32]

DARK_SMOOTHING_UM = 15 * REFERENCE_PIXEL_SIZE_UM
DARK_PERCENTILE = 5.0
"""Dark structures (vessels, ventricles, cracks) are the dimmest few percent of smoothed
tissue. Nonspecific antibody accumulates on their surfaces."""

CONTEXT_CONTRAST = 1.4
"""Objects are also described by the structure containing them at this permissive contrast:
a nucleus sits in a compact blob, antibody on a vessel sits in a long thin one."""

NUCLEUS_AREA_RANGE_UM2 = (3 * REFERENCE_PIXEL_SIZE_UM**2, 400 * REFERENCE_PIXEL_SIZE_UM**2)
"""Keeps nucleus-sized objects and rejects single-pixel noise and large debris."""


def object_size_range(pixel_size_um: float) -> tuple[int, int]:
    """NUCLEUS_AREA_RANGE_UM2 in pixels."""
    pixel_area = pixel_size_um**2
    low, high = NUCLEUS_AREA_RANGE_UM2
    return max(1, round(low / pixel_area)), max(1, round(high / pixel_area))


def label_objects(
    mask: BoolImage, min_size: int, max_size: int
) -> tuple[Labels, npt.NDArray[np.int64]]:
    """Label connected components and return the ids of those within the size range (pixels)."""
    labels, n = ndi.label(mask)
    if n == 0:
        return labels, np.array([], dtype=np.int64)
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero((sizes >= min_size) & (sizes <= max_size)) + 1
    return labels, keep.astype(np.int64)


def count_objects(
    image: FloatImage,
    threshold: float,
    mask: BoolImage,
    pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
) -> int:
    """Number of nucleus-sized objects above `threshold` inside `mask`."""
    _, keep = label_objects((image > threshold) & mask, *object_size_range(pixel_size_um))
    return len(keep)


def contrast_ratio(
    image: FloatImage, pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM
) -> FloatImage:
    """Image divided by its local background, so thresholds mean the same in bright and dim tissue."""
    background = local_background(image, pixel_size_um)
    ratio: FloatImage = (image / np.maximum(background, 1.0)).astype(np.float32)
    return ratio


AF_RATIO_CUTOFF = 2.0
"""Signal/AF peak ratio separating autofluorescence from label. Autofluorescent granules sit
near 0.7 (IQR ~0.5-0.9); AF647 bleeding into the 561 nm channel at ~7% puts labelled cells
near 14. Results were identical for cutoffs from 1.5 to 3 on real data. Assumes unchanged
laser powers and exposures: if these change, the autofluorescence ratio moves with them."""


@dataclass(frozen=True)
class SpatialContext:
    """Where each object sits. Nonspecific antibody hugs vessel and tissue boundaries."""

    distance_to_dark_um: npt.NDArray[np.float64]
    context_length_um: npt.NDArray[np.float64]
    """Major axis of the structure containing the object at CONTEXT_CONTRAST."""
    context_area_px: npt.NDArray[np.float64]


def _major_axis_length(mask: BoolImage) -> float:
    """Major axis in pixels, matching skimage's axis_major_length, for a single component."""
    rows, columns = np.nonzero(mask)
    if len(rows) < 2:
        return 1.0
    covariance = np.cov(np.stack([rows.astype(np.float64), columns.astype(np.float64)]))
    largest = float(np.linalg.eigvalsh(covariance)[-1])
    return float(4.0 * np.sqrt(max(largest, 0.0)))


def spatial_context(
    signal: FloatImage,
    tissue: BoolImage,
    centroids: npt.NDArray[np.float64],
    pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
) -> SpatialContext:
    """Distance from each object to the nearest dark structure, and the shape of its surroundings."""
    if len(centroids) == 0:
        empty = np.array([], dtype=np.float64)
        return SpatialContext(empty, empty, empty)
    rows = centroids[:, 0].astype(int)
    columns = centroids[:, 1].astype(int)

    smoothed = ndi.uniform_filter(signal, to_pixels(DARK_SMOOTHING_UM, pixel_size_um))
    dark = smoothed < np.percentile(signal[tissue], DARK_PERCENTILE)
    distance = ndi.distance_transform_edt(~(dark | ~tissue)) * pixel_size_um

    ratio = signal / np.maximum(local_background(signal, pixel_size_um), 1.0)
    context, _ = ndi.label((ratio > CONTEXT_CONTRAST) & tissue)
    boxes = ndi.find_objects(context)
    at_object = context[rows, columns]
    lengths: dict[int, float] = {}
    areas: dict[int, float] = {}
    for component in np.unique(at_object):  # only the components that contain an object
        if component == 0:
            continue
        patch = context[boxes[component - 1]] == component
        lengths[int(component)] = _major_axis_length(patch) * pixel_size_um
        areas[int(component)] = float(patch.sum())
    return SpatialContext(
        distance_to_dark_um=np.asarray(distance[rows, columns], dtype=np.float64),
        context_length_um=np.array([lengths.get(int(c), 0.0) for c in at_object]),
        context_area_px=np.array([areas.get(int(c), 0.0) for c in at_object]),
    )


@dataclass(frozen=True)
class Colocalisation:
    centroids: npt.NDArray[np.float64]
    """(n, 2) row/column centroids of objects detected in the signal channel."""
    signal_contrast: npt.NDArray[np.float64]
    """Peak contrast (image / local background) in the signal channel."""
    area_px: npt.NDArray[np.float64]
    signal_peak: npt.NDArray[np.float64]
    af_peak: npt.NDArray[np.float64]
    """Peak background-subtracted intensity in each channel within each signal-channel object."""
    ratio: npt.NDArray[np.float64]
    """signal_peak / af_peak."""
    is_autofluorescent: npt.NDArray[np.bool_]


def classify_by_autofluorescence(
    signal: FloatImage,
    af: FloatImage,
    tissue: BoolImage,
    detection_contrast: float = 2.0,
    af_ratio_cutoff: float = AF_RATIO_CUTOFF,
    pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
) -> Colocalisation:
    """Detect objects in the signal channel and flag those whose signal/AF ratio is autofluorescent.

    Classifying on the ratio rather than on whether an object is visible in the AF channel
    matters because the label bleeds into the AF channel: bright labelled cells are visible
    there too, but with a ratio ~20x higher than autofluorescence.
    """
    signal_background = local_background(signal, pixel_size_um)
    af_background = local_background(af, pixel_size_um)
    signal_ratio = signal / np.maximum(signal_background, 1.0)
    detected = (signal_ratio > detection_contrast) & tissue
    labels, keep = label_objects(detected, *object_size_range(pixel_size_um))
    if len(keep) == 0:
        empty = np.array([], dtype=np.float64)
        return Colocalisation(
            np.empty((0, 2)), empty, empty, empty, empty, empty, np.array([], dtype=np.bool_)
        )

    signal_peak = np.asarray(
        ndi.maximum(signal - signal_background, labels, keep), dtype=np.float64
    )
    af_peak = np.asarray(ndi.maximum(af - af_background, labels, keep), dtype=np.float64)
    # an object with no AF-channel signal at all gets a large ratio, i.e. is specific
    ratio = signal_peak / np.maximum(af_peak, 1.0)
    return Colocalisation(
        centroids=np.asarray(ndi.center_of_mass(detected, labels, keep), dtype=np.float64),
        signal_contrast=np.asarray(ndi.maximum(signal_ratio, labels, keep), dtype=np.float64),
        area_px=np.asarray(ndi.sum(detected, labels, keep), dtype=np.float64),
        signal_peak=signal_peak,
        af_peak=af_peak,
        ratio=ratio,
        is_autofluorescent=ratio < af_ratio_cutoff,
    )
