"""Edit the settings below, then run:  uv run map_cells_to_regions.py

Assigns each specifically-labelled object to an atlas region and writes a per-region table.
"""

import csv
from pathlib import Path

from mesospim_analysis.pipeline import read_summary_csv
from mesospim_analysis.regions import (
    AtlasGrid,
    ancestors,
    assign_regions,
    object_positions,
    region_names,
    region_volumes,
    roll_up,
)
from mesospim_analysis.verdict import select_specific

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

DRIVE = Path("/Volumes/JR-cache/Mesospim")
RUN = DRIVE / "brainreg_bakeoff" / "princeton_mouse_20um"
ACQUISITION = "2026-05-18_N027_001"

OBJECTS = DRIVE / "afcorr_results" / f"{ACQUISITION}_50um_objects.csv"
SUMMARY = DRIVE / "afcorr_results" / f"{ACQUISITION}_50um_summary.csv"
ATLAS = RUN / "atlas_in_original_space.tif"
STRUCTURES = RUN / "structures.csv"
OUT = DRIVE / "afcorr_results" / f"{ACQUISITION}_regions.csv"

GRID = AtlasGrid(pixel_size_um=26.08, z_step_um=5.0)

# ---------------------------------------------------------------------------------------------


def main() -> None:
    with open(OBJECTS, newline="") as handle:
        objects = list(csv.DictReader(handle))
    summaries = read_summary_csv(SUMMARY)
    specific = select_specific(objects, summaries)
    print(f"{len(objects)} detected, {len(specific)} specific", flush=True)

    assigned = assign_regions(object_positions(specific), ATLAS, GRID)
    outside = int((assigned == 0).sum())
    print(f"{len(assigned) - outside} assigned, {outside} outside the atlas", flush=True)

    print("measuring region volumes in this brain...", flush=True)
    volumes = region_volumes(ATLAS, GRID)
    names = region_names(STRUCTURES)
    totals = roll_up(assigned, ancestors(STRUCTURES))

    # A parent's volume is the sum of everything beneath it, for the same reason counts roll up.
    paths = ancestors(STRUCTURES)
    rolled_volume: dict[int, float] = {}
    for region_id, mm3 in volumes.items():
        for ancestor in paths.get(region_id, [region_id]):
            rolled_volume[ancestor] = rolled_volume.get(ancestor, 0.0) + mm3

    rows = []
    for region_id, count in totals.items():
        acronym, name = names.get(region_id, ("?", "?"))
        mm3 = rolled_volume.get(region_id, 0.0)
        rows.append({
            "id": region_id,
            "acronym": acronym,
            "name": name,
            "cells": count,
            "volume_mm3": round(mm3, 4),
            "cells_per_mm3": round(count / mm3, 1) if mm3 > 0 else "",
        })
    rows.sort(key=lambda r: -int(r["cells"]))

    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} regions)\n")

    print(f"{'acronym':10s} {'cells':>8s} {'mm3':>8s} {'per mm3':>9s}  name")
    for row in rows[:25]:
        print(f"{row['acronym']:10s} {row['cells']:8d} {row['volume_mm3']:8.2f} "
              f"{str(row['cells_per_mm3']):>9s}  {row['name'][:44]}")


if __name__ == "__main__":
    main()
