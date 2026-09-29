"""The registration helpers, on synthetic volumes with a known answer."""

from pathlib import Path

import numpy as np
import tifffile
import numpy.typing as npt
import pytest

from mesospim_analysis.registration import (
    CLIP_PERCENTILE,
    region_ids,
    write_region_subset,
    PreparedVolume,
    RegionScore,
    SourceGeometry,
    brain_mask,
    dice,
    histogram_occupancy,
    prepare_for_registration,
    region_subset,
    resample_in_plane,
    score_regions,
    surface_distance_mm,
)

IntImage = npt.NDArray[np.integer]


def _brain_in_a_lit_frame(
    shape: tuple[int, int, int] = (60, 80, 80),
    background: int = 200,
    tissue: int = 900,
    hot_voxels: int = 5,
) -> tuple[IntImage, npt.NDArray[np.bool_]]:
    """A bright ellipsoid in a dimmer, striped frame: the situation brainreg gets."""
    rng = np.random.default_rng(0)
    z, y, x = np.indices(shape)
    centre = [s / 2 for s in shape]
    radii = [s / 3.5 for s in shape]
    truth = (
        ((z - centre[0]) / radii[0]) ** 2
        + ((y - centre[1]) / radii[1]) ** 2
        + ((x - centre[2]) / radii[2]) ** 2
    ) < 1.0
    stripes = background + 40 * np.sin(x / 3.0)
    volume = np.where(truth, tissue, stripes) + rng.normal(0, 8, shape)
    volume[:, :2, :] = 0  # the black edge left where no tile covered
    flat = volume.reshape(-1)
    flat[rng.choice(np.flatnonzero(truth.reshape(-1)), hot_voxels, replace=False)] = 60000
    return volume.reshape(shape).astype(np.uint16), truth


def test_the_mask_finds_the_brain_not_the_lit_background() -> None:
    """Losing brain is the costly error; a voxel of background at the edge is not.

    The mask is dilated deliberately to keep the dim pial surface, so it is expected to be
    slightly larger than the truth and dice alone would penalise that.
    """
    volume, truth = _brain_in_a_lit_frame()
    mask = brain_mask(np.asarray(volume, dtype=np.float32), subsample=1)
    recall = (mask & truth).sum() / truth.sum()
    precision = (mask & truth).sum() / mask.sum()
    assert recall > 0.995, f"the mask cut into the brain: kept only {recall:.3f} of it"
    assert precision > 0.75, f"the mask took in background: only {precision:.3f} is brain"
    assert mask.sum() >= truth.sum(), "the mask should keep the whole brain, dilated"


def test_preparation_zeroes_the_background_and_clips_the_hot_voxels() -> None:
    volume, truth = _brain_in_a_lit_frame()
    prepared = prepare_for_registration(volume, subsample=1)
    assert isinstance(prepared, PreparedVolume)
    assert prepared.volume.dtype == volume.dtype
    assert prepared.volume.shape == volume.shape
    outside = prepared.volume[~prepared.mask]
    assert not outside.any(), "background survived the mask"
    assert prepared.volume.max() <= prepared.clip_value
    assert prepared.clip_value < 60000, "the hot voxels were not clipped"
    assert prepared.looks_like_a_brain


def test_clipping_is_what_opens_up_the_histogram() -> None:
    """The reason to clip: niftyreg's similarity only sees the bins the tissue occupies."""
    volume, _ = _brain_in_a_lit_frame()
    prepared = prepare_for_registration(volume, subsample=1)
    before = histogram_occupancy(volume, prepared.mask)
    after = histogram_occupancy(prepared.volume, prepared.mask)
    assert after > before, f"clipping did not help: {before} -> {after} bins"


def test_an_empty_volume_is_refused_rather_than_masked() -> None:
    with pytest.raises(ValueError, match="empty"):
        prepare_for_registration(np.zeros((20, 20, 20), dtype=np.uint16), subsample=1)


def test_the_projection_reverses_brainregs_reorientation() -> None:
    """ial -> asr is a transpose and two flips, so the inverse must return the original."""
    shape = (40, 12, 14)
    source = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    # what brainreg does to the source: transpose(1,0,2) then reverse axes 1 and 2
    registered = source.transpose(1, 0, 2)[:, ::-1, ::-1]
    geometry = SourceGeometry(shape=shape)
    recovered = np.stack(
        [geometry.plane_from_atlas_space(registered, z) for z in range(shape[0])]
    )
    assert recovered.shape == shape
    assert np.array_equal(recovered, source)


def test_resampling_labels_never_invents_a_region() -> None:
    labels = np.zeros((10, 5, 5), dtype=np.uint32)
    labels[:, 1:4, 1:4] = 7
    labels[:, 2, 2] = 13
    # axis 0 of an atlas-space volume is the source's y, axis 2 its x; axis 1 stays untouched
    out = resample_in_plane(labels, (10, 20, 30))
    assert (out.shape[0], out.shape[2]) == (20, 30)
    assert out.shape[1] == labels.shape[1]
    assert set(np.unique(out).tolist()) <= {0, 7, 13}


def test_region_subset_numbers_the_groups() -> None:
    labels = np.zeros((4, 4, 4), dtype=np.uint32)
    labels[0] = 100
    labels[1] = 200
    labels[2] = 999  # not requested
    out = region_subset(labels, {1: np.array([100]), 2: np.array([200, 201])})
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) == {0, 1, 2}
    assert (out[0] == 1).all() and (out[1] == 2).all() and (out[2] == 0).all()


def test_a_region_on_the_right_tissue_scores_near_one() -> None:
    annotation = np.zeros((10, 10, 10), dtype=np.uint32)
    annotation[:, :5, :] = 1
    annotation[:, 5:, :] = 2
    reference = np.where(annotation == 1, 2.0, 1.0).astype(np.float32)
    aligned = reference.copy()
    scores = {s.acronym: s for s in score_regions(
        aligned, annotation, reference, {"A": np.array([1]), "B": np.array([2])}
    )}
    assert scores["A"].ratio == pytest.approx(1.0, abs=1e-5)
    assert scores["B"].ratio == pytest.approx(1.0, abs=1e-5)
    assert scores["A"].covered == pytest.approx(1.0)


def test_a_region_on_the_wrong_tissue_does_not() -> None:
    annotation = np.zeros((10, 10, 10), dtype=np.uint32)
    annotation[:, :5, :] = 1
    annotation[:, 5:, :] = 2
    reference = np.where(annotation == 1, 2.0, 1.0).astype(np.float32)
    swapped = np.where(annotation == 1, 1.0, 2.0).astype(np.float32)  # labels on the wrong half
    scores = {s.acronym: s for s in score_regions(
        swapped, annotation, reference, {"A": np.array([1]), "B": np.array([2])}
    )}
    assert scores["A"].ratio < 0.8
    assert scores["B"].ratio > 1.2
    assert isinstance(scores["A"], RegionScore)


def test_surface_distance_is_in_mm_and_independent_of_voxel_size() -> None:
    """The measure that corrected the atlas comparison: a coarser grid must not flatter a run."""
    fine = np.zeros((60, 60, 60), dtype=bool)
    fine[10:50, 10:50, 10:50] = True
    shifted = np.zeros_like(fine)
    shifted[12:52, 10:50, 10:50] = True  # 2 voxels at 20 um = 40 um
    mean_fine, _ = surface_distance_mm(shifted, fine, resolution_um=20.0)

    coarse = fine[::2, ::2, ::2]
    coarse_shifted = shifted[::2, ::2, ::2]  # 1 voxel at 40 um = the same 40 um
    mean_coarse, _ = surface_distance_mm(coarse_shifted, coarse, resolution_um=40.0)

    assert mean_fine == pytest.approx(mean_coarse, abs=0.02), (
        f"the same physical shift measured {mean_fine:.3f} mm and {mean_coarse:.3f} mm"
    )


def test_dice_handles_empty_masks() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert np.isnan(dice(empty, empty))
    full = np.ones((4, 4, 4), dtype=bool)
    assert dice(full, full) == pytest.approx(1.0)


def _structures_csv(path: Path) -> Path:
    """A miniature atlas table: CA1 subdivided into layers, CP not subdivided."""
    path.write_text(
        "id,name,acronym,structure_id_path\n"
        "382,Field CA1,CA1,/997/1080/382/\n"
        "391,CA1 stratum oriens,CA1so,/997/1080/382/391/\n"
        "399,CA1 pyramidal layer,CA1sp,/997/1080/382/399/\n"
        "672,Caudoputamen,CP,/997/477/672/\n"
    )
    return path


def test_region_ids_include_descendants(tmp_path: Path) -> None:
    """The Allen labels CA1's layers and barely uses the parent id; princeton uses only the
    parent. Taking descendants is what makes the two comparable."""
    csv_path = _structures_csv(tmp_path / "structures.csv")
    assert region_ids(csv_path, "CA1").tolist() == [382, 391, 399]
    assert region_ids(csv_path, "CP").tolist() == [672]


def test_an_unknown_acronym_is_named_in_the_error(tmp_path: Path) -> None:
    csv_path = _structures_csv(tmp_path / "structures.csv")
    with pytest.raises(KeyError, match="NOTAREGION"):
        region_ids(csv_path, "NOTAREGION")


def test_write_region_subset_streams_only_the_requested_regions(tmp_path: Path) -> None:
    labels = np.zeros((6, 4, 4), dtype=np.uint32)
    labels[0] = 391      # a CA1 layer
    labels[1] = 672      # CP
    labels[2] = 999      # something not asked for
    source = tmp_path / "full.tif"
    tifffile.imwrite(source, labels)

    out = tmp_path / "subset.tif"
    counts = write_region_subset(
        source, {1: np.array([382, 391, 399]), 2: np.array([672])}, out
    )
    written = tifffile.imread(out)
    assert written.dtype == np.uint8
    assert written.shape == labels.shape
    assert set(np.unique(written).tolist()) == {0, 1, 2}
    assert (written[0] == 1).all() and (written[1] == 2).all() and (written[2] == 0).all()
    assert counts == {1: 16, 2: 16}
