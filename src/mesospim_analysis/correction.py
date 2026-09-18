from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from mesospim_analysis.types import BoolImage, FloatImage

MAD_TO_SIGMA = 1.4826

REFERENCE_PIXEL_SIZE_UM = 3.26
"""mesoSPIM 2x at pyramid level 0. Spatial defaults were validated at this pixel size and are
defined as (validated pixel count) x this, so they convert back to exactly those counts here."""

BACKGROUND_WINDOW_UM = 30 * REFERENCE_PIXEL_SIZE_UM
TISSUE_SMOOTHING_UM = 51 * REFERENCE_PIXEL_SIZE_UM
TISSUE_EROSION_UM = 10 * REFERENCE_PIXEL_SIZE_UM


def to_pixels(length_um: float, pixel_size_um: float) -> int:
    return max(1, round(length_um / pixel_size_um))


def tissue_mask(
    image: FloatImage,
    pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
    percentile: float = 40.0,
) -> BoolImage:
    """Mask of tissue, excluding the frame surround and a margin at the tissue edge.

    Thresholds a heavily smoothed image at a percentile. Tested down to 0.8% tissue
    coverage of the frame, where it admits some flat background but no false objects.
    """
    smoothed = ndi.uniform_filter(image, to_pixels(TISSUE_SMOOTHING_UM, pixel_size_um))
    mask = smoothed > np.percentile(smoothed, percentile)
    eroded: BoolImage = ndi.binary_erosion(
        mask, iterations=to_pixels(TISSUE_EROSION_UM, pixel_size_um)
    )
    return eroded


def local_background(
    image: FloatImage, pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM
) -> FloatImage:
    """Median-filter background over a ~100 um window, computed at half resolution for speed.

    The window must be well above nucleus size so puncta do not leak into the estimate.
    """
    size = to_pixels(BACKGROUND_WINDOW_UM, 2 * pixel_size_um)
    background_half = ndi.median_filter(image[::2, ::2], size=size)
    background = np.kron(background_half, np.ones((2, 2), dtype=np.float32))
    return background[: image.shape[0], : image.shape[1]].astype(np.float32)


def robust_sigma(values: FloatImage) -> float:
    """Noise standard deviation from the median absolute deviation."""
    return MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))


@dataclass(frozen=True)
class AutofluorescenceCorrection:
    background_subtracted: FloatImage
    """Signal channel minus local background, clipped at zero. Still contains autofluorescence."""
    corrected: FloatImage
    """Background-subtracted signal with the scaled autofluorescence channel removed."""
    alpha: float
    """Ratio of signal-channel to AF-channel intensity for autofluorescent objects."""
    n_fit_pixels: int
    signal_sigma: float
    af_sigma: float
    tissue: BoolImage


def correct_autofluorescence(
    signal: FloatImage,
    af: FloatImage,
    pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
    af_threshold_sigma: float = 3.0,
    min_af_for_fit: float = 60.0,
) -> AutofluorescenceCorrection:
    """Remove autofluorescence from a signal channel using a registered off-target channel.

    Both channels are background-subtracted. The AF channel is soft-thresholded at
    `af_threshold_sigma` so that only real autofluorescent objects are subtracted, not
    the AF channel's own shot noise. `alpha` is fitted per brain as the median
    signal/AF ratio over pixels with strong autofluorescence, because it absorbs
    relative laser power and each brain's AF spectrum and so does not transfer
    between brains.
    """
    if signal.shape != af.shape:
        raise ValueError(f"channel shapes differ: {signal.shape} vs {af.shape}")

    tissue = tissue_mask(signal, pixel_size_um)
    if not tissue.any():
        raise ValueError("no tissue found")
    signal_hp = signal - local_background(signal, pixel_size_um)
    af_hp = af - local_background(af, pixel_size_um)

    signal_sigma = robust_sigma(signal_hp[tissue])
    af_sigma = robust_sigma(af_hp[tissue])
    af_clean = np.clip(af_hp - af_threshold_sigma * af_sigma, 0, None)

    fit_pixels = tissue & (af_clean > min_af_for_fit)
    n_fit_pixels = int(fit_pixels.sum())
    if n_fit_pixels == 0:
        raise ValueError("no autofluorescent pixels above min_af_for_fit; cannot fit alpha")
    alpha = float(np.median(signal_hp[fit_pixels] / af_clean[fit_pixels]))

    return AutofluorescenceCorrection(
        background_subtracted=np.clip(signal_hp, 0, None),
        corrected=np.clip(signal_hp - alpha * af_clean, 0, None),
        alpha=alpha,
        n_fit_pixels=n_fit_pixels,
        signal_sigma=signal_sigma,
        af_sigma=af_sigma,
        tissue=tissue,
    )
