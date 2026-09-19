from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from mesospim_analysis.correction import REFERENCE_PIXEL_SIZE_UM, local_background
from mesospim_analysis.types import BoolImage, FloatImage

Labels = npt.NDArray[np.int32]

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
class Colocalisation:
    centroids: npt.NDArray[np.float64]
    """(n, 2) row/column centroids of objects detected in the signal channel."""
    signal_contrast: npt.NDArray[np.float64]
    """Peak contrast (image / local background) in the signal channel."""
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
            np.empty((0, 2)), empty, empty, empty, empty, np.array([], dtype=np.bool_)
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
        signal_peak=signal_peak,
        af_peak=af_peak,
        ratio=ratio,
        is_autofluorescent=ratio < af_ratio_cutoff,
    )
