"""Edit the settings below, then run:  uv run threshold_sweep.py

How many cell-sized objects each brain yields as the detection contrast threshold is lowered,
split by depth from the tissue surface.

Detection requires signal/local_background > detection_contrast. That ratio depends on how
bright the surroundings are, so it is not comparable between regions: in N027 the isocortex
yielded 5 objects at the default 2.0 and 343 at 1.3, while the ventricles went only from 179 to
562. Cortical cells sit just below the cutoff; ventricular debris sits well above it.

Lowering a threshold always finds more of everything, so the question is whether the extra
objects are cells or noise. A brain given no antibody answers that: whatever it yields at a
threshold is the floor, and only the excess above it is signal.

No registration needed -- the outer shell of the tissue stands in for cortex, and it is applied
identically to every brain, so the comparison holds even though the proxy is rough.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

from mesospim_analysis.acquisitions import resolve_h5
from mesospim_analysis.correction import local_background, tissue_mask
from mesospim_analysis.detection import label_objects, object_size_range
from mesospim_analysis.pipeline import open_volumes
from mesospim_analysis.projection import iter_slab_projections, planes_per_slab

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

DATA_ROOT = Path("/Volumes/MarcBusche/James/Mesospim")

ACQUISITIONS = [
    (DATA_ROOT / "2026-05-18/N027/001", "N027 real"),
    (DATA_ROOT / "2026-05-18/N030/001", "N030 no antibody"),
]

Z_START_UM = 4750.0  # the depth inspected in N027; use the same in every brain
THICKNESS_UM = 50.0
PYRAMID_LEVEL = 0
SIGNAL_CHANNEL = "638 nm"
AF_CHANNEL = "561 nm"

THRESHOLDS = [2.0, 1.8, 1.6, 1.5, 1.4, 1.3]
SHELL_UM = 1200.0
"""Depth from the tissue surface counted as "outer". Mouse cortex is roughly 1-1.5 mm thick."""

# ---------------------------------------------------------------------------------------------


def sweep(path: Path, label: str) -> None:
    h5_path = resolve_h5(path)
    with h5py.File(h5_path, "r") as handle:
        volumes = open_volumes(
            handle, h5_path, PYRAMID_LEVEL, SIGNAL_CHANNEL, AF_CHANNEL
        )
        scale = volumes.scale
        planes = planes_per_slab(THICKNESS_UM, scale.z_step_um)
        start = round(Z_START_UM / scale.z_step_um)
        [slab] = list(
            iter_slab_projections(
                volumes.signal, planes, start, start + planes, volumes.read_block
            )
        )
        raw = np.asarray(slab.projection, dtype=np.float32)

    tissue = tissue_mask(raw, scale.pixel_size_um)
    depth_um = ndi.distance_transform_edt(tissue) * scale.pixel_size_um
    outer = tissue & (depth_um <= SHELL_UM)
    inner = tissue & (depth_um > SHELL_UM)

    ratio = raw / np.maximum(local_background(raw, scale.pixel_size_um), 1.0)
    low, high = object_size_range(scale.pixel_size_um)

    outer_mm2 = float(outer.sum()) * scale.pixel_size_um**2 / 1e6
    inner_mm2 = float(inner.sum()) * scale.pixel_size_um**2 / 1e6
    print(f"\n{label}   outer shell {outer_mm2:.1f} mm2, interior {inner_mm2:.1f} mm2", flush=True)
    print(f"  {'thr':>5s} {'outer':>7s} {'inner':>7s} {'outer/mm2':>10s} {'inner/mm2':>10s}",
          flush=True)
    for threshold in THRESHOLDS:
        counts = []
        for mask in (outer, inner):
            _, keep = label_objects((ratio > threshold) & mask, low, high)
            counts.append(len(keep))
        print(f"  {threshold:5.1f} {counts[0]:7d} {counts[1]:7d} "
              f"{counts[0] / max(outer_mm2, 1e-9):10.1f} {counts[1] / max(inner_mm2, 1e-9):10.1f}",
              flush=True)


def main() -> None:
    for path, label in ACQUISITIONS:
        try:
            sweep(path, label)
        except Exception as exc:  # one unreadable brain should not stop the rest
            print(f"\n{label}: FAILED -- {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
