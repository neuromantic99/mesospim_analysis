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

Both steps of the pipeline have to run at each threshold, not just detection. A brain with no
antibody still detects thousands of autofluorescent objects; what removes them is the ratio
test against the 561 channel. Counting detections alone compares autofluorescence between
brains and says nothing about label.

No registration needed -- the outer shell of the tissue stands in for cortex, and it is applied
identically to every brain, so the comparison holds even though the proxy is rough.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

from mesospim_analysis.acquisitions import resolve_h5
from mesospim_analysis.correction import correct_autofluorescence
from mesospim_analysis.detection import AF_RATIO_MULTIPLE, classify_by_autofluorescence
from mesospim_analysis.pipeline import open_volumes
from mesospim_analysis.projection import iter_slab_projections, planes_per_slab

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

DATA_ROOT = Path("/Volumes/MarcBusche/James/Mesospim")

# Each of these answers something the N027/N030 pair cannot. Their alphas, and so their ratio
# cutoffs, span 0.46 to 1.61, and three have 8.0 MP frames against N027's 15.3.
ACQUISITIONS = [
    # done already, kept so every run is self-contained
    (DATA_ROOT / "2026-05-18/N027/001", "N027 real, alpha 0.59, 1185/MP"),
    (DATA_ROOT / "2026-05-18/N030/001", "N030 no antibody at all, alpha 0.84"),
    # the realistic floor: secondary only, and already carrying 222/MP of nonspecific antibody.
    # N030 has no secondary in it at all, so it cannot test what lowering the threshold does to
    # the nonspecific component that every working brain contains.
    (DATA_ROOT / "2026-08-27/N041/001", "N041 secondary-only control, 222/MP"),
    # the most permissive cutoff in the cohort at 3 x 0.46 = 1.37, and from N041's own round and
    # frame size, so floor and signal are measured under the same conditions.
    (DATA_ROOT / "2026-08-27/N049/001", "N049 real, alpha 0.46, 463/MP"),
    # densely labelled: does a lower threshold recover cortical cells, or merge neighbours in
    # the 50 um projection? Occlusion would show as counts rising more slowly than in N027.
    (DATA_ROOT / "2026-07-08/N045/001", "N045 real, best brain, 3442/MP"),
    # a control whose alpha is four times N027's, to check that N030's zero floor is not just
    # its own harsh calibration.
    (DATA_ROOT / "2026-05-21/N029/001", "N029 secondary-only control, alpha 1.61"),
]

Z_START_UM = 4750.0
"""The depth inspected in N027. A fixed z is not the same anatomical level in every brain --
they sit at different heights in the chamber and the 8 MP frames differ again -- so check the
printed tissue areas before comparing. A slab with much less tissue than its neighbours is at
the edge of the brain and its counts mean something different."""
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
        [signal_slab] = list(
            iter_slab_projections(
                volumes.signal, planes, start, start + planes, volumes.read_block
            )
        )
        [af_slab] = list(
            iter_slab_projections(
                volumes.af, planes, start, start + planes, volumes.read_block
            )
        )
        raw = np.asarray(signal_slab.projection, dtype=np.float32)
        af = np.asarray(af_slab.projection, dtype=np.float32)

    result = correct_autofluorescence(raw, af, scale.pixel_size_um)
    tissue = result.tissue
    cutoff = AF_RATIO_MULTIPLE * result.alpha

    depth_um = ndi.distance_transform_edt(tissue) * scale.pixel_size_um
    outer_mm2 = float((tissue & (depth_um <= SHELL_UM)).sum()) * scale.pixel_size_um**2 / 1e6
    inner_mm2 = float((tissue & (depth_um > SHELL_UM)).sum()) * scale.pixel_size_um**2 / 1e6

    print(f"\n{label}   alpha {result.alpha:.2f}, cutoff {cutoff:.2f}, "
          f"outer {outer_mm2:.1f} mm2, interior {inner_mm2:.1f} mm2", flush=True)
    print(f"  {'thr':>5s} {'detected':>9s} {'specific':>9s} "
          f"{'outer/mm2':>10s} {'inner/mm2':>10s}", flush=True)

    for threshold in THRESHOLDS:
        coloc = classify_by_autofluorescence(
            raw, af, tissue,
            detection_contrast=threshold,
            af_ratio_cutoff=cutoff,
            pixel_size_um=scale.pixel_size_um,
        )
        specific = ~coloc.is_autofluorescent
        if not specific.any():
            print(f"  {threshold:5.1f} {len(coloc.is_autofluorescent):9d} {0:9d}", flush=True)
            continue
        points = coloc.centroids[specific].astype(int)
        depths = depth_um[points[:, 0], points[:, 1]]
        outer = int((depths <= SHELL_UM).sum())
        inner = int((depths > SHELL_UM).sum())
        print(f"  {threshold:5.1f} {len(coloc.is_autofluorescent):9d} {int(specific.sum()):9d} "
              f"{outer / max(outer_mm2, 1e-9):10.1f} {inner / max(inner_mm2, 1e-9):10.1f}",
              flush=True)


def main() -> None:
    for path, label in ACQUISITIONS:
        try:
            sweep(path, label)
        except Exception as exc:  # one unreadable brain should not stop the rest
            print(f"\n{label}: FAILED -- {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
