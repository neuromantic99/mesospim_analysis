"""Object-to-region assignment, on a synthetic atlas where the answer is known."""

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from mesospim_analysis.regions import (
    AtlasGrid,
    ancestors,
    assign_regions,
    object_positions,
    region_table,
    region_volumes,
    roll_up,
)


def _structures(path: Path) -> Path:
    """CA1 and DG under HIP under HPF; CP under STR. Leaves only, as atlases label them."""
    path.write_text(
        "id,name,acronym,structure_id_path\n"
        "1089,Hippocampal formation,HPF,/997/1089/\n"
        "1080,Hippocampal region,HIP,/997/1089/1080/\n"
        "382,Field CA1,CA1,/997/1089/1080/382/\n"
        "726,Dentate gyrus,DG,/997/1089/1080/726/\n"
        "477,Striatum,STR,/997/477/\n"
        "672,Caudoputamen,CP,/997/477/672/\n"
    )
    return path


def _atlas(path: Path) -> Path:
    """10 planes; plane 0-4 are CA1, 5-7 DG, 8-9 CP. 4x4 in plane."""
    volume = np.zeros((10, 4, 4), dtype=np.uint32)
    volume[0:5] = 382
    volume[5:8] = 726
    volume[8:10] = 672
    tifffile.imwrite(path, volume)
    return path


GRID = AtlasGrid(pixel_size_um=10.0, z_step_um=2.0)


def test_positions_use_the_slab_midpoint() -> None:
    rows = [{"z_start_um": "100.0", "z_end_um": "150.0", "y_um": "20.0", "x_um": "30.0"}]
    assert object_positions(rows) == [(125.0, 20.0, 30.0)]


def test_objects_are_assigned_to_the_region_they_sit_in(tmp_path: Path) -> None:
    atlas = _atlas(tmp_path / "atlas.tif")
    positions = [
        (2.0, 10.0, 10.0),    # plane 1 -> CA1
        (12.0, 10.0, 10.0),   # plane 6 -> DG
        (18.0, 10.0, 10.0),   # plane 9 -> CP
    ]
    assigned = assign_regions(positions, atlas, GRID)
    assert assigned.tolist() == [382, 726, 672]


def test_objects_outside_the_volume_are_unassigned_not_clamped(tmp_path: Path) -> None:
    """Clamping would silently pile every stray object into the edge regions."""
    atlas = _atlas(tmp_path / "atlas.tif")
    assigned = assign_regions([(2.0, 10.0, 10.0), (2.0, 10_000.0, 10.0)], atlas, GRID)
    assert assigned.tolist() == [382, 0]


def test_counts_roll_up_to_every_containing_region(tmp_path: Path) -> None:
    """A CA1 object is also a HIP object and an HPF object; without this a parent reads zero."""
    paths = ancestors(_structures(tmp_path / "s.csv"))
    totals = roll_up(np.array([382, 382, 726, 672], dtype=np.int64), paths)
    assert totals[382] == 2
    assert totals[726] == 1
    assert totals[1080] == 3      # HIP = CA1 + DG
    assert totals[1089] == 3      # HPF
    assert totals[672] == 1
    assert totals[477] == 1       # STR
    assert totals[997] == 4       # root


def test_unassigned_objects_are_not_counted(tmp_path: Path) -> None:
    paths = ancestors(_structures(tmp_path / "s.csv"))
    assert sum(roll_up(np.array([0, 0], dtype=np.int64), paths).values()) == 0


def test_volumes_measure_this_brain_not_the_atlas(tmp_path: Path) -> None:
    atlas = _atlas(tmp_path / "atlas.tif")
    volumes = region_volumes(atlas, GRID)
    voxel_mm3 = 10.0 * 10.0 * 2.0 / 1e9
    assert volumes[382] == pytest.approx(5 * 16 * voxel_mm3)
    assert volumes[726] == pytest.approx(3 * 16 * voxel_mm3)
    assert 0 not in volumes, "background must not be reported as a region"


def test_a_parent_region_gets_its_descendants_volume(tmp_path: Path) -> None:
    """Otherwise a parent has a large count over a near-zero volume and an absurd density."""
    atlas = _atlas(tmp_path / "atlas.tif")
    structures = _structures(tmp_path / "s.csv")
    assigned = np.array([382, 382, 726, 672], dtype=np.int64)
    rows = {row.acronym: row for row in region_table(assigned, region_volumes(atlas, GRID), structures)}
    assert rows["HIP"].cells == 3
    assert rows["HIP"].volume_mm3 == pytest.approx(
        rows["CA1"].volume_mm3 + rows["DG"].volume_mm3
    )
    assert rows["HIP"].cells_per_mm3 == pytest.approx(
        rows["HIP"].cells / rows["HIP"].volume_mm3
    )
