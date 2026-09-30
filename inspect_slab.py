"""Edit the settings below, then run:  uv run inspect_slab.py

Writes the raw, background-subtracted and AF-corrected projections for chosen slabs, with the
detections drawn on the RAW projection -- which is the image the detector actually works on.
run_slab saves only _bgsub and _afcorr, so the raw view is the one missing when asking why a
slab produced the objects it did.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from mesospim_analysis.acquisitions import resolve_h5
from mesospim_analysis.correction import correct_autofluorescence
from mesospim_analysis.detection import classify_by_autofluorescence
from mesospim_analysis.pipeline import open_volumes
from mesospim_analysis.projection import iter_slab_projections, planes_per_slab

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

ACQUISITION = Path("/Volumes/MarcBusche/James/Mesospim/2026-05-18/N027/001")

# Slab 95 sits in the band at z 4500-4950 that carries ~500 objects each against a median of 43,
# and is what pulls every object distribution towards the midline. Slab 76 is a median slab for
# comparison; it was already known to spread across the whole brain.
Z_STARTS_UM = [4750.0, 3800.0]

THICKNESS_UM = 50.0
PYRAMID_LEVEL = 0
SIGNAL_CHANNEL = "638 nm"
AF_CHANNEL = "561 nm"  # 488 was rejected as a reference: 234 false cells in a no-antibody control
OUT_DIR = Path("/Volumes/JR-cache/Mesospim/slab_inspection")

# ---------------------------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = resolve_h5(ACQUISITION)
    print(f"reading {h5_path}", flush=True)

    with h5py.File(h5_path, "r") as handle:
        volumes = open_volumes(
            handle, h5_path, PYRAMID_LEVEL, SIGNAL_CHANNEL, AF_CHANNEL
        )
        scale = volumes.scale
        planes = planes_per_slab(THICKNESS_UM, scale.z_step_um)
        print(f"{scale.pixel_size_um:.2f} um/px, {planes} planes per slab", flush=True)

        for z_start_um in Z_STARTS_UM:
            start = round(z_start_um / scale.z_step_um)
            tag = f"z{int(z_start_um)}"
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
            result = correct_autofluorescence(
                raw,
                np.asarray(af_slab.projection, dtype=np.float32),
                scale.pixel_size_um,
            )
            coloc = classify_by_autofluorescence(
                raw,
                np.asarray(af_slab.projection, dtype=np.float32),
                result.tissue,
                pixel_size_um=scale.pixel_size_um,
            )
            specific = ~coloc.is_autofluorescent
            print(
                f"{tag}: alpha {result.alpha:.2f}, {len(coloc.is_autofluorescent)} detected, "
                f"{int(specific.sum())} specific",
                flush=True,
            )

            for name, image in (
                ("raw", raw),
                ("bgsub", result.background_subtracted),
                ("afcorr", result.corrected),
            ):
                tifffile.imwrite(
                    OUT_DIR / f"{tag}_{name}.tif",
                    np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16),
                )

            fig, axs = plt.subplots(1, 2, figsize=(18, 9))
            top = float(np.percentile(raw[result.tissue], 99.5))
            for ax, mask, label in (
                (
                    axs[0],
                    slice(None),
                    f"all {len(coloc.is_autofluorescent)} detections",
                ),
                (axs[1], specific, f"{int(specific.sum())} specific"),
            ):
                ax.imshow(raw, cmap="gray", vmax=top)
                points = coloc.centroids[mask]
                ax.scatter(
                    points[:, 1],
                    points[:, 0],
                    s=10,
                    facecolors="none",
                    edgecolors="#ff2d55",
                    linewidths=0.5,
                )
                ax.set_title(f"{tag} raw projection — {label}", fontsize=10)
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{tag}_detections.png", dpi=110)
            plt.close(fig)

    print(f"\nwrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
