import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mesospim_analysis.filters import (
    DEFAULT_FILTER,
    GEOMETRY_FEATURES,
    NonspecificFilter,
    fit_nonspecific_filter,
    read_objects_csv,
    write_scored_objects,
)

HEADER = [
    "slab", "z_start_um", "z_end_um", "y_px", "x_px", "y_um", "x_um", "area_px", "area_um2",
    "signal_peak", "af_peak", "ratio", "signal_contrast", "is_autofluorescent",
    "distance_to_dark_um", "context_length_um", "context_area_px",
]


def _object(distance: float, length: float, area: float, autofluorescent: int = 0) -> dict[str, str]:
    row = dict.fromkeys(HEADER, "0")
    row["is_autofluorescent"] = str(autofluorescent)
    row["distance_to_dark_um"] = str(distance)
    row["context_length_um"] = str(length)
    row["context_area_px"] = str(area)
    return row


def _populations() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Cells sit far from dark structures in compact contexts; nonspecific objects do neither."""
    rng = np.random.default_rng(0)
    def pos(loc: float, sd: float, n: int) -> npt.NDArray[np.float64]:
        return np.clip(rng.normal(loc, sd, n), 1.0, None)

    cells = [_object(d, l, a) for d, l, a in
             zip(pos(800, 150, 300), pos(18, 4, 300), pos(14, 4, 300))]
    junk = [_object(d, l, a) for d, l, a in
            zip(pos(120, 60, 300), pos(70, 20, 300), pos(90, 30, 300))]
    return cells, junk


def test_fitted_filter_separates_the_populations() -> None:
    cells, junk = _populations()
    fitted = fit_nonspecific_filter(cells, junk, provenance="synthetic")
    assert fitted.features == GEOMETRY_FEATURES
    assert fitted.score(cells).mean() > 0.9
    assert fitted.score(junk).mean() < 0.1
    assert fitted.brain_score(cells) > fitted.brain_score(junk)


def test_threshold_for_recall_keeps_that_fraction() -> None:
    cells, junk = _populations()
    fitted = fit_nonspecific_filter(cells, junk)
    for recall in (0.9, 0.95):
        threshold = fitted.threshold_for_recall(cells, recall)
        assert (fitted.score(cells) >= threshold).mean() == pytest.approx(recall, abs=0.02)
    with pytest.raises(ValueError, match="recall"):
        fitted.threshold_for_recall(cells, 0.0)


def test_scoring_needs_no_sklearn_and_round_trips(tmp_path: Path) -> None:
    cells, junk = _populations()
    fitted = fit_nonspecific_filter(cells, junk, provenance="synthetic")
    path = tmp_path / "filter.json"
    fitted.to_json(path)
    reloaded = NonspecificFilter.from_json(path)
    assert reloaded == fitted
    np.testing.assert_allclose(reloaded.score(cells), fitted.score(cells))


def test_empty_input() -> None:
    assert len(DEFAULT_FILTER.score([])) == 0
    assert np.isnan(DEFAULT_FILTER.brain_score([]))


def test_write_scored_objects(tmp_path: Path) -> None:
    cells, junk = _populations()
    source = tmp_path / "objects.csv"
    with open(source, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows([*cells, *junk, _object(800, 18, 14, autofluorescent=1)])

    fitted = fit_nonspecific_filter(cells, junk)
    out = tmp_path / "filtered.csv"
    kept, removed = write_scored_objects(source, out, threshold=0.5, filter_=fitted)
    assert kept + removed == len(cells) + len(junk)  # the autofluorescent row is not scored
    assert kept == pytest.approx(len(cells), abs=20)

    written = list(csv.DictReader(open(out)))
    assert len(written) == kept
    assert all(float(r["cell_score"]) >= 0.5 for r in written)
    assert read_objects_csv(source, specific_only=False)[-1]["is_autofluorescent"] == "1"
