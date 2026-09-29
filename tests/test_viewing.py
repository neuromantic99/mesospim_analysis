"""The scale and the shape check, which are what make an overlay trustworthy."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from mesospim_analysis.viewing import (
    PIXEL_SIZE_UM,
    Z_STEP_UM,
    RegistrationView,
    level_scale,
)


@pytest.mark.parametrize(
    ("level", "expected"),
    [(0, (5.0, 3.26, 3.26)), (1, (5.0, 6.52, 6.52)), (3, (5.0, 26.08, 26.08))],
)
def test_levels_downsample_in_plane_only(
    level: int, expected: tuple[float, float, float]
) -> None:
    """z keeps its step at every level, which is why the volume is so anisotropic."""
    assert level_scale(level) == pytest.approx(expected)


def test_the_scale_is_anisotropic_enough_to_matter() -> None:
    z, y, _ = level_scale(3)
    assert y / z > 5, "if this ratio were near 1 the scale would not need setting"
    assert (z, y) == (Z_STEP_UM, PIXEL_SIZE_UM * 8)


def test_a_negative_level_is_refused() -> None:
    with pytest.raises(ValueError, match="pyramid level"):
        level_scale(-1)


def _write(path: Path, shape: tuple[int, int, int]) -> Path:
    tifffile.imwrite(path, np.zeros(shape, dtype=np.uint8))
    return path


def test_mismatched_volumes_are_refused_before_anything_is_displayed(
    tmp_path: Path,
) -> None:
    """Labels projected onto a different volume would overlay silently and wrongly."""
    view = RegistrationView(
        raw=_write(tmp_path / "raw.tif", (10, 8, 8)),
        labels=_write(tmp_path / "labels.tif", (10, 8, 9)),
    )
    with pytest.raises(ValueError, match="shapes differ"):
        view.check()


def test_matching_volumes_pass_and_report_the_shape(tmp_path: Path) -> None:
    view = RegistrationView(
        raw=_write(tmp_path / "raw.tif", (10, 8, 8)),
        labels=_write(tmp_path / "labels.tif", (10, 8, 8)),
    )
    assert view.check() == (10, 8, 8)


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    view = RegistrationView(
        raw=_write(tmp_path / "raw.tif", (4, 4, 4)),
        labels=tmp_path / "absent.tif",
    )
    with pytest.raises(FileNotFoundError, match="absent.tif"):
        view.check()
