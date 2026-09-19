"""Entry points: correct a whole brain, one slab, or a pair of projection TIFFs.

Call these from main.py (edit the settings there) or from a notebook.
"""

import sys
from pathlib import Path

import h5py

from mesospim_analysis.acquisitions import Acquisition, parse_acquisition, resolve_h5
from mesospim_analysis.correction import (
    REFERENCE_PIXEL_SIZE_UM,
    AutofluorescenceCorrection,
    correct_autofluorescence,
)
from mesospim_analysis.detection import classify_by_autofluorescence, count_objects
from mesospim_analysis.io import load_channel_pair, save_uint16
from mesospim_analysis.pipeline import (
    COUNT_THRESHOLDS,
    OpenVolumes,
    SlabStackWriter,
    SlabSummary,
    open_volumes,
    process_slabs,
    write_summary_csv,
)
from mesospim_analysis.projection import planes_per_slab

METHOD = "local background subtracted (median filter, ~98 um window)"


def afcorr_description(alpha: float) -> str:
    return f"{METHOD}; autofluorescence removed as signal - {alpha:.3f} * softthresh(af, 3 sigma)"


def print_counts(before: dict[float, int], after: dict[float, int]) -> None:
    for t in COUNT_THRESHOLDS:
        kept = 100 * after[t] / before[t] if before[t] else 0.0
        print(f"    >{t:.0f}: bgsub {before[t]} -> afcorr {after[t]} ({kept:.1f}% kept)")


def print_summary(summary: SlabSummary) -> None:
    where = f"slab {summary.slab} (z {summary.z_start_um:g}-{summary.z_end_um:g} um)"
    if summary.status != "ok":
        print(f"{where}: skipped, {summary.status}")
        return
    print(
        f"{where}: alpha {summary.alpha:.3f} (fit on {summary.n_fit_pixels} px), "
        f"puncta {summary.puncta_total} = {summary.puncta_autofluorescent} autofluorescent "
        f"+ {summary.puncta_specific} specific"
    )
    print_counts(summary.objects_bgsub, summary.objects_afcorr)


def _open(
    f: h5py.File, h5_path: Path, level: int, signal: str, af: str, thickness_um: float
) -> tuple[OpenVolumes, int]:
    volumes = open_volumes(f, h5_path, level, signal, af)
    scale = volumes.scale
    planes = planes_per_slab(thickness_um, scale.z_step_um)
    if scale.too_coarse_for_counting:
        print(f"warning: {scale.pixel_size_um:.2f} um/px is too coarse to separate nuclei from "
              "noise; use pyramid level 0 for counts", file=sys.stderr)
    if planes * scale.z_step_um != thickness_um:
        print(f"note: slab thickness rounded to {planes} planes = "
              f"{planes * scale.z_step_um:g} um", file=sys.stderr)
    return volumes, planes


def _output_stem(acquisition: Acquisition, planes: int, volumes: OpenVolumes) -> str:
    return f"{acquisition.name}_{planes * volumes.scale.z_step_um:g}um"


def run_brain(
    path: Path,
    thickness_um: float = 50.0,
    pyramid_level: int = 0,
    signal: str = "638 nm",
    af: str = "561 nm",
    out_dir: Path | None = None,
    z_range_um: tuple[float, float] | None = None,
    save_bgsub: bool = False,
) -> list[SlabSummary]:
    """Correct every consecutive slab of a stitched.h5 into a BigTIFF stack plus a summary csv.

    `path` is stitched.h5 or its directory. Outputs go to `out_dir`, default afcorr/ next to
    the h5. Returns the per-slab summaries written to the csv.
    """
    h5_path = resolve_h5(path)
    acquisition = parse_acquisition(h5_path)
    out_dir = out_dir or h5_path.parent / "afcorr"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[SlabSummary] = []
    with h5py.File(h5_path, "r") as f:
        volumes, planes = _open(f, h5_path, pyramid_level, signal, af, thickness_um)
        z_step = volumes.scale.z_step_um
        z_start, z_end = 0, None
        if z_range_um is not None:
            z_start, z_end = (round(v / z_step) for v in z_range_um)
        stem = _output_stem(acquisition, planes, volumes)
        note = f"{planes * z_step:g} um slab max projections from plane {z_start}"
        stacks = {"afcorr": SlabStackWriter(
            out_dir / f"{stem}_afcorr.tif",
            f"{METHOD}; autofluorescence removed with alpha fitted per slab (see csv); {note}",
        )}
        if save_bgsub:
            stacks["bgsub"] = SlabStackWriter(out_dir / f"{stem}_bgsub.tif", f"{METHOD}; {note}")
        plane_shape = volumes.signal.shape[1:]
        try:
            for summary, result in process_slabs(
                volumes.signal, volumes.af, planes, volumes.scale,
                z_start, z_end, volumes.read_block,
            ):
                print_summary(summary)
                summaries.append(summary)
                stacks["afcorr"].write(None if result is None else result.corrected, plane_shape)
                if "bgsub" in stacks:
                    stacks["bgsub"].write(
                        None if result is None else result.background_subtracted, plane_shape
                    )
        finally:
            for stack in stacks.values():
                stack.close()

    write_summary_csv(out_dir / f"{stem}_summary.csv", summaries)
    n_ok = sum(s.status == "ok" for s in summaries)
    specific = sum(s.puncta_specific for s in summaries)
    print(f"{acquisition.name}: {n_ok}/{len(summaries)} slabs corrected, "
          f"{specific} specific puncta in total; outputs in {out_dir}")
    return summaries


def run_slab(
    path: Path,
    z_start_um: float,
    thickness_um: float = 50.0,
    pyramid_level: int = 0,
    signal: str = "638 nm",
    af: str = "561 nm",
    out_dir: Path | None = None,
) -> tuple[SlabSummary, AutofluorescenceCorrection]:
    """Correct the slab starting at `z_start_um` and write its _bgsub/_afcorr TIFFs."""
    h5_path = resolve_h5(path)
    acquisition = parse_acquisition(h5_path)
    out_dir = out_dir or h5_path.parent / "afcorr"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        volumes, planes = _open(f, h5_path, pyramid_level, signal, af, thickness_um)
        z_start = round(z_start_um / volumes.scale.z_step_um)
        [(summary, result)] = process_slabs(
            volumes.signal, volumes.af, planes, volumes.scale,
            z_start, z_start + planes, volumes.read_block,
        )
    print_summary(summary)
    if result is None:
        raise ValueError(f"slab could not be corrected: {summary.status}")
    name = f"{_output_stem(acquisition, planes, volumes)}_z{summary.z_start_um:g}"
    save_uint16(out_dir / f"{name}_bgsub.tif", result.background_subtracted, METHOD)
    save_uint16(out_dir / f"{name}_afcorr.tif", result.corrected, afcorr_description(result.alpha))
    print(f"wrote {name}_bgsub.tif and {name}_afcorr.tif to {out_dir}")
    return summary, result


def run_tiff(
    signal_path: Path,
    af_path: Path,
    out_dir: Path | None = None,
    prefix: str | None = None,
    pixel_size_um: float = REFERENCE_PIXEL_SIZE_UM,
) -> AutofluorescenceCorrection:
    """Correct a pair of existing projection TIFFs (signal channel, AF channel)."""
    out_dir = out_dir or signal_path.parent
    prefix = prefix or signal_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    signal, af = load_channel_pair(signal_path, af_path)
    result = correct_autofluorescence(signal, af, pixel_size_um)
    save_uint16(out_dir / f"{prefix}_bgsub.tif", result.background_subtracted, METHOD)
    save_uint16(out_dir / f"{prefix}_afcorr.tif", result.corrected, afcorr_description(result.alpha))

    print(f"alpha {result.alpha:.3f} (fit on {result.n_fit_pixels} px), "
          f"noise sigma signal {result.signal_sigma:.1f}, af {result.af_sigma:.1f}")
    coloc = classify_by_autofluorescence(signal, af, result.tissue, pixel_size_um=pixel_size_um)
    n, n_af = len(coloc.is_autofluorescent), int(coloc.is_autofluorescent.sum())
    print(f"puncta >2x local background: {n} = {n_af} autofluorescent + {n - n_af} specific")
    print("objects inside tissue:")
    print_counts(
        {t: count_objects(result.background_subtracted, t, result.tissue, pixel_size_um)
         for t in COUNT_THRESHOLDS},
        {t: count_objects(result.corrected, t, result.tissue, pixel_size_um)
         for t in COUNT_THRESHOLDS},
    )
    print(f"wrote {prefix}_bgsub.tif and {prefix}_afcorr.tif to {out_dir}")
    return result
