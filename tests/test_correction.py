import numpy as np
import pytest

from mesospim_analysis.correction import correct_autofluorescence, local_background, robust_sigma
from mesospim_analysis.detection import classify_by_autofluorescence, count_objects
from mesospim_analysis.types import FloatImage

SHAPE = (400, 400)
BACKGROUND = 200.0


def _add_spots(image: FloatImage, centres: list[tuple[int, int]], amplitude: float) -> None:
    for r, c in centres:
        image[r - 1 : r + 2, c - 1 : c + 2] += amplitude


def _synthetic_brain(seed: int = 0) -> tuple[FloatImage, FloatImage, int, int]:
    """Two registered channels: AF spots appear in both, specific spots only in the signal."""
    rng = np.random.default_rng(seed)
    signal = np.full(SHAPE, BACKGROUND, dtype=np.float32) + rng.normal(0, 5, SHAPE).astype(np.float32)
    af = np.full(SHAPE, BACKGROUND, dtype=np.float32) + rng.normal(0, 5, SHAPE).astype(np.float32)
    grid = [(r, c) for r in range(40, 360, 20) for c in range(40, 360, 20)]
    af_spots, specific_spots = grid[::2], grid[1::2]
    _add_spots(af, af_spots, 400.0)
    _add_spots(signal, af_spots, 300.0)  # AF seen at 0.75x strength in the signal channel
    _add_spots(signal, specific_spots, 300.0)
    return signal, af, len(af_spots), len(specific_spots)


def test_local_background_ignores_puncta() -> None:
    image = np.full(SHAPE, BACKGROUND, dtype=np.float32)
    _add_spots(image, [(100, 100), (200, 250)], 1000.0)
    assert np.allclose(local_background(image), BACKGROUND)


def test_robust_sigma_matches_gaussian_noise() -> None:
    values = np.random.default_rng(1).normal(0, 7, 100_000).astype(np.float32)
    assert robust_sigma(values) == pytest.approx(7, rel=0.02)


def test_correction_recovers_alpha_and_removes_only_autofluorescence() -> None:
    signal, af, n_af, n_specific = _synthetic_brain()
    result = correct_autofluorescence(signal, af)
    # soft-thresholding the AF channel shrinks it slightly, so alpha sits a little above 0.75
    assert 0.75 < result.alpha < 0.9
    tissue = np.ones(SHAPE, dtype=np.bool_)
    assert count_objects(result.background_subtracted, 100, tissue) == n_af + n_specific
    assert count_objects(result.corrected, 100, tissue) == n_specific


def test_classification_separates_af_from_specific() -> None:
    signal, af, n_af, n_specific = _synthetic_brain()
    coloc = classify_by_autofluorescence(signal, af, np.ones(SHAPE, dtype=np.bool_))
    assert int(coloc.is_autofluorescent.sum()) == n_af
    assert int((~coloc.is_autofluorescent).sum()) == n_specific


def test_mismatched_channels_rejected() -> None:
    a = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="shapes differ"):
        correct_autofluorescence(a, np.zeros((10, 11), dtype=np.float32))
