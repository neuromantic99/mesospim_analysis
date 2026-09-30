"""Assigning detected objects to atlas regions.

Objects are detected at pyramid level 0 and their coordinates are recorded in microns, while
the projected atlas sits on the level-3 grid: 26.08 um in plane, 5 um in z. Working in microns
rather than pixels means the two never have to agree about which pyramid level they are on.

Each object carries the slab it came from rather than a z position, so its depth is the slab's
midpoint. A 50 um slab is ten atlas planes thick, so an object's region is the region at the
middle of the slab it was found in -- fine for regions far larger than that, and unreliable for
anything thinner.
"""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tifffile

UNASSIGNED = 0
"""Atlas id for a voxel outside the brain, or an object outside the volume."""


@dataclass(frozen=True)
class AtlasGrid:
    """The geometry the projected atlas sits on."""

    pixel_size_um: float = 26.08
    z_step_um: float = 5.0
    shape: tuple[int, int, int] | None = None

    def index(self, z_um: float, y_um: float, x_um: float) -> tuple[int, int, int] | None:
        """Voxel holding a point, or None if it falls outside the volume."""
        index = (
            int(round(z_um / self.z_step_um)),
            int(round(y_um / self.pixel_size_um)),
            int(round(x_um / self.pixel_size_um)),
        )
        if self.shape is None:
            return index
        if any(i < 0 or i >= n for i, n in zip(index, self.shape, strict=True)):
            return None
        return index


def object_positions(rows: list[dict[str, str]]) -> list[tuple[float, float, float]]:
    """(z, y, x) in microns. z is the midpoint of the slab the object was found in."""
    return [
        (
            (float(row["z_start_um"]) + float(row["z_end_um"])) / 2.0,
            float(row["y_um"]),
            float(row["x_um"]),
        )
        for row in rows
    ]


def assign_regions(
    positions: list[tuple[float, float, float]], atlas_path: Path, grid: AtlasGrid
) -> npt.NDArray[np.int64]:
    """The atlas id under each object, in the order given.

    Objects are grouped by plane so each plane of the atlas is read once. Read object by object
    instead and this is thousands of random seeks into a 1.5 GB volume on an external disk.
    """
    volume = tifffile.memmap(atlas_path)
    resolved = AtlasGrid(grid.pixel_size_um, grid.z_step_um, tuple(volume.shape))

    by_plane: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    out = np.zeros(len(positions), dtype=np.int64)
    for order, (z_um, y_um, x_um) in enumerate(positions):
        located = resolved.index(z_um, y_um, x_um)
        if located is not None:
            by_plane[located[0]].append((order, located[1], located[2]))

    for plane_index in sorted(by_plane):
        plane = np.asarray(volume[plane_index])
        entries = by_plane[plane_index]
        rows = np.fromiter((e[1] for e in entries), dtype=np.int64, count=len(entries))
        cols = np.fromiter((e[2] for e in entries), dtype=np.int64, count=len(entries))
        orders = np.fromiter((e[0] for e in entries), dtype=np.int64, count=len(entries))
        out[orders] = plane[rows, cols]
    return out


def region_volumes(atlas_path: Path, grid: AtlasGrid) -> dict[int, float]:
    """Volume in mm3 of every region present, in the sample's own space.

    Densities need the volume actually covered in this brain, not the atlas's nominal volume:
    a region the registration only partly covers would otherwise look sparsely labelled.
    """
    volume = tifffile.memmap(atlas_path)
    voxel_mm3 = grid.pixel_size_um**2 * grid.z_step_um / 1e9
    counts: Counter[int] = Counter()
    for index in range(volume.shape[0]):
        ids, found = np.unique(np.asarray(volume[index]), return_counts=True)
        counts.update(dict(zip(ids.tolist(), found.tolist(), strict=True)))
    return {int(i): n * voxel_mm3 for i, n in counts.items() if i != UNASSIGNED}


def region_names(structures_csv: Path) -> dict[int, tuple[str, str]]:
    """id -> (acronym, name)."""
    with open(structures_csv, newline="") as handle:
        return {
            int(row["id"]): (row["acronym"], row["name"])
            for row in csv.DictReader(handle)
        }


def ancestors(structures_csv: Path) -> dict[int, list[int]]:
    """id -> its path from the root, so counts can be rolled up to coarser regions."""
    with open(structures_csv, newline="") as handle:
        return {
            int(row["id"]): [
                int(part) for part in row["structure_id_path"].strip("/").split("/") if part
            ]
            for row in csv.DictReader(handle)
        }


def roll_up(assigned: npt.NDArray[np.int64], paths: dict[int, list[int]]) -> Counter[int]:
    """Count each object against every region containing it, not just its leaf.

    An object in CA1 is also in the hippocampal region, the hippocampal formation and the
    cortex. Without this, asking for a parent region's count returns almost nothing, because
    voxels are labelled at the finest level the atlas defines.
    """
    totals: Counter[int] = Counter()
    for region_id in assigned.tolist():
        if region_id == UNASSIGNED:
            continue
        for ancestor in paths.get(region_id, [region_id]):
            totals[ancestor] += 1
    return totals


@dataclass(frozen=True)
class RegionRow:
    region_id: int
    acronym: str
    name: str
    cells: int
    volume_mm3: float

    @property
    def cells_per_mm3(self) -> float | None:
        return self.cells / self.volume_mm3 if self.volume_mm3 > 0 else None


def region_table(
    assigned: npt.NDArray[np.int64], volumes: dict[int, float], structures_csv: Path
) -> list[RegionRow]:
    """One row per region, counts and volumes both rolled up the hierarchy.

    A parent's volume has to be the sum of its descendants' for the same reason its count does:
    voxels are labelled at the finest level the atlas defines, so a parent queried directly
    holds almost nothing and its density would be meaningless.
    """
    paths = ancestors(structures_csv)
    names = region_names(structures_csv)
    counts = roll_up(assigned, paths)

    rolled: dict[int, float] = {}
    for region_id, mm3 in volumes.items():
        for ancestor in paths.get(region_id, [region_id]):
            rolled[ancestor] = rolled.get(ancestor, 0.0) + mm3

    rows = [
        RegionRow(
            region_id=region_id,
            acronym=names.get(region_id, ("?", "?"))[0],
            name=names.get(region_id, ("?", "?"))[1],
            cells=count,
            volume_mm3=rolled.get(region_id, 0.0),
        )
        for region_id, count in counts.items()
    ]
    rows.sort(key=lambda row: -row.cells)
    return rows


def write_region_table(rows: list[RegionRow], out: Path) -> None:
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "acronym", "name", "cells", "volume_mm3", "cells_per_mm3"])
        for row in rows:
            density = row.cells_per_mm3
            writer.writerow([
                row.region_id, row.acronym, row.name, row.cells,
                round(row.volume_mm3, 4),
                "" if density is None else round(density, 1),
            ])
