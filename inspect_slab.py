"""Edit the settings below, then run:  uv run inspect_slab.py

Compares slab thicknesses at one depth, because the projection may be the reason the cortex
looks empty.

Detection thresholds signal/local_background. A max projection over n planes takes the maximum
of n noise draws at every pixel, so the local background rises with thickness while a cell's
peak does not: thickening the slab lowers the very ratio the detector tests. In bright, uniform
cortex the margin is already thin, whereas in the dark ventricle lumen there is no background
to inflate. That would explain why the isocortex yielded 5 objects at contrast 2.0 against the
ventricles' 179, and why the counts are so sensitive to the threshold.

Thickness also sets how well an object is localised in depth. At 50 um every object is placed
at its slab midpoint, so its depth is known to +/-25 um -- ten atlas planes, which is wider
than the CA1 pyramidal layer.

Objects per mm3 is the comparable number across thicknesses; raw counts are not, since a
thinner slab contains less tissue.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi

from mesospim_analysis.acquisitions import resolve_h5
from mesospim_analysis.correction import correct_autofluorescence, local_background
from mesospim_analysis.detection import AF_RATIO_MULTIPLE, classify_by_autofluorescence
from mesospim_analysis.pipeline import open_volumes
from mesospim_analysis.projection import iter_slab_projections, planes_per_slab

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

ACQUISITION = Path("/Volumes/MarcBusche/James/Mesospim/2026-05-18/N027/001")

Z_START_UM = 4750.0
"""One depth, several thicknesses. Slab 95 at this depth carries ~500 objects against a median
of 43 and is what pulls the object distribution towards the midline."""

THICKNESSES_UM = [50.0, 20.0, 10.0, 5.0]
"""5 um is a single plane: no projection at all, and the control for the whole question."""

SHELL_UM = 1200.0  # depth from the tissue surface counted as "outer", a stand-in for cortex
PYRAMID_LEVEL = 0
SIGNAL_CHANNEL = "638 nm"
AF_CHANNEL = "561 nm"  # 488 was rejected: 234 false cells in a brain given no antibody
SAVE_IMAGES = True
OUT_DIR = Path("/Volumes/JR-cache/Mesospim/slab_inspection")

# ---------------------------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = resolve_h5(ACQUISITION)
    print(f"reading {h5_path}", flush=True)
    rows = []

    with h5py.File(h5_path, "r") as handle:
        volumes = open_volumes(
            handle, h5_path, PYRAMID_LEVEL, SIGNAL_CHANNEL, AF_CHANNEL
        )
        scale = volumes.scale
        print(f"{scale.pixel_size_um:.2f} um/px, {scale.z_step_um:.1f} um z step", flush=True)

        for thickness in THICKNESSES_UM:
            planes = planes_per_slab(thickness, scale.z_step_um)
            actual = planes * scale.z_step_um
            start = round(Z_START_UM / scale.z_step_um)
            projections = []
            for volume in (volumes.signal, volumes.af):
                [slab] = list(
                    iter_slab_projections(
                        volume, planes, start, start + planes, volumes.read_block
                    )
                )
                projections.append(np.asarray(slab.projection, dtype=np.float32))
            raw, af = projections

            result = correct_autofluorescence(raw, af, scale.pixel_size_um)
            tissue = result.tissue
            background = local_background(raw, scale.pixel_size_um)
            contrast = raw / np.maximum(background, 1.0)

            depth_um = ndi.distance_transform_edt(tissue) * scale.pixel_size_um
            outer = tissue & (depth_um <= SHELL_UM)
            inner = tissue & (depth_um > SHELL_UM)

            coloc = classify_by_autofluorescence(
                raw, af, tissue,
                af_ratio_cutoff=AF_RATIO_MULTIPLE * result.alpha,
                pixel_size_um=scale.pixel_size_um,
            )
            specific = ~coloc.is_autofluorescent
            points = coloc.centroids[specific].astype(int)
            depths = depth_um[points[:, 0], points[:, 1]] if len(points) else np.zeros(0)

            # per mm3, so thicknesses are comparable: a thinner slab holds less tissue
            area_mm2 = scale.pixel_size_um**2 / 1e6
            volumes_mm3 = {
                "outer": float(outer.sum()) * area_mm2 * actual / 1000.0,
                "inner": float(inner.sum()) * area_mm2 * actual / 1000.0,
            }
            counts = {
                "outer": int((depths <= SHELL_UM).sum()),
                "inner": int((depths > SHELL_UM).sum()),
            }
            rows.append({
                "thickness": actual,
                "planes": planes,
                "alpha": result.alpha,
                "bg_outer": float(np.median(background[outer])),
                "bg_inner": float(np.median(background[inner])),
                "contrast_outer": float(np.percentile(contrast[outer], 99.9)),
                "contrast_inner": float(np.percentile(contrast[inner], 99.9)),
                "outer_per_mm3": counts["outer"] / max(volumes_mm3["outer"], 1e-9),
                "inner_per_mm3": counts["inner"] / max(volumes_mm3["inner"], 1e-9),
                "detected": len(coloc.is_autofluorescent),
                "specific": int(specific.sum()),
            })
            print(f"  {actual:5.0f} um ({planes} planes): {rows[-1]['detected']} detected, "
                  f"{rows[-1]['specific']} specific", flush=True)

            if SAVE_IMAGES:
                tifffile.imwrite(
                    OUT_DIR / f"z{int(Z_START_UM)}_t{int(actual)}_raw.tif",
                    np.clip(raw, 0, np.iinfo(np.uint16).max).astype(np.uint16),
                )

    print(f"\n{'thick':>6s} {'planes':>7s} {'alpha':>6s} {'bg out':>7s} {'bg in':>7s} "
          f"{'contr out':>10s} {'contr in':>9s} {'out/mm3':>9s} {'in/mm3':>9s}")
    for row in rows:
        print(f"{row['thickness']:6.0f} {row['planes']:7d} {row['alpha']:6.2f} "
              f"{row['bg_outer']:7.0f} {row['bg_inner']:7.0f} {row['contrast_outer']:10.2f} "
              f"{row['contrast_inner']:9.2f} {row['outer_per_mm3']:9.1f} {row['inner_per_mm3']:9.1f}")
    print("\nIf the projection is the problem, background falls and contrast rises as the slab")
    print("thins, and outer/mm3 rises with it. If they are flat, thickness is not the cause.")

    if len(rows) > 1:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        thick = [r["thickness"] for r in rows]
        for ax, keys, title in (
            (axs[0], ("bg_outer", "bg_inner"), "median local background"),
            (axs[1], ("contrast_outer", "contrast_inner"), "99.9th pct contrast"),
            (axs[2], ("outer_per_mm3", "inner_per_mm3"), "specific objects per mm3"),
        ):
            for key, colour, label in zip(keys, ("#ff2d55", "#0a84ff"), ("outer", "inner")):
                ax.plot(thick, [r[key] for r in rows], "o-", color=colour, label=label)
            ax.set_xlabel("slab thickness (um)")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"z{int(Z_START_UM)}_thickness.png", dpi=110)
        print(f"\nwrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
