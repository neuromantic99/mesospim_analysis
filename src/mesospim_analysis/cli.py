import argparse
import sys
from pathlib import Path

import h5py

from mesospim_analysis.acquisitions import (
    Acquisition,
    find_acquisitions,
    parse_acquisition,
    resolve_h5,
)
from mesospim_analysis.correction import REFERENCE_PIXEL_SIZE_UM, correct_autofluorescence
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


def run_tiff(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir or args.signal.parent
    prefix: str = args.prefix or args.signal.stem
    pixel_size: float = args.pixel_size_um
    out_dir.mkdir(parents=True, exist_ok=True)

    signal, af = load_channel_pair(args.signal, args.af)
    result = correct_autofluorescence(signal, af, pixel_size)
    save_uint16(out_dir / f"{prefix}_bgsub.tif", result.background_subtracted, METHOD)
    save_uint16(out_dir / f"{prefix}_afcorr.tif", result.corrected, afcorr_description(result.alpha))

    print(f"alpha {result.alpha:.3f} (fit on {result.n_fit_pixels} px), "
          f"noise sigma signal {result.signal_sigma:.1f}, af {result.af_sigma:.1f}")
    coloc = classify_by_autofluorescence(signal, af, result.tissue, pixel_size_um=pixel_size)
    n, n_af = len(coloc.is_autofluorescent), int(coloc.is_autofluorescent.sum())
    print(f"puncta >2x local background: {n} = {n_af} autofluorescent + {n - n_af} specific")
    print("objects inside tissue:")
    print_counts(
        {t: count_objects(result.background_subtracted, t, result.tissue, pixel_size)
         for t in COUNT_THRESHOLDS},
        {t: count_objects(result.corrected, t, result.tissue, pixel_size)
         for t in COUNT_THRESHOLDS},
    )
    print(f"wrote {prefix}_bgsub.tif and {prefix}_afcorr.tif to {out_dir}")


def prepare_volume(
    args: argparse.Namespace, f: h5py.File, h5_path: Path
) -> tuple[OpenVolumes, int]:
    volumes = open_volumes(f, h5_path, args.level, args.signal, args.af)
    scale = volumes.scale
    planes = planes_per_slab(args.thickness_um, scale.z_step_um)
    if scale.too_coarse_for_counting:
        print(f"warning: {scale.pixel_size_um:.2f} um/px is too coarse to separate nuclei from "
              "noise; use pyramid level 0 for counts", file=sys.stderr)
    if planes * scale.z_step_um != args.thickness_um:
        print(f"note: slab thickness rounded to {planes} planes = "
              f"{planes * scale.z_step_um:g} um", file=sys.stderr)
    return volumes, planes


def output_stem(acquisition: Acquisition, planes: int, volumes: OpenVolumes) -> str:
    return f"{acquisition.name}_{planes * volumes.scale.z_step_um:g}um"


def run_slab(args: argparse.Namespace) -> None:
    h5_path = resolve_h5(args.path)
    acquisition = parse_acquisition(h5_path)
    out_dir: Path = args.out_dir or h5_path.parent / "afcorr"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        volumes, planes = prepare_volume(args, f, h5_path)
        z_start = round(args.z_start_um / volumes.scale.z_step_um)
        [(summary, result)] = process_slabs(
            volumes.signal, volumes.af, planes, volumes.scale,
            z_start, z_start + planes, volumes.read_block,
        )
    print_summary(summary)
    if result is None:
        raise SystemExit(f"slab could not be corrected: {summary.status}")
    name = f"{output_stem(acquisition, planes, volumes)}_z{summary.z_start_um:g}"
    save_uint16(out_dir / f"{name}_bgsub.tif", result.background_subtracted, METHOD)
    save_uint16(out_dir / f"{name}_afcorr.tif", result.corrected, afcorr_description(result.alpha))
    print(f"wrote {name}_bgsub.tif and {name}_afcorr.tif to {out_dir}")


def run_brain(args: argparse.Namespace) -> None:
    h5_path = resolve_h5(args.path)
    acquisition = parse_acquisition(h5_path)
    out_dir: Path = args.out_dir or h5_path.parent / "afcorr"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[SlabSummary] = []
    with h5py.File(h5_path, "r") as f:
        volumes, planes = prepare_volume(args, f, h5_path)
        z_step = volumes.scale.z_step_um
        z_start, z_end = 0, None
        if args.z_range_um is not None:
            z_start, z_end = (round(v / z_step) for v in args.z_range_um)
        stem = output_stem(acquisition, planes, volumes)
        note = f"{planes * z_step:g} um slab max projections from plane {z_start}"
        stacks = {"afcorr": SlabStackWriter(
            out_dir / f"{stem}_afcorr.tif",
            f"{METHOD}; autofluorescence removed with alpha fitted per slab (see csv); {note}",
        )}
        if args.save_bgsub:
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


def run_list(args: argparse.Namespace) -> None:
    for acquisition in find_acquisitions(args.root, args.mouse):
        print(f"{acquisition.name}\t{acquisition.h5_path}")


def add_volume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="stitched.h5, or the directory containing it")
    parser.add_argument("--thickness-um", type=float, default=50.0, help="slab thickness (default 50)")
    parser.add_argument("--level", type=int, default=0, help="pyramid level (default 0; count at 0)")
    parser.add_argument("--signal", default="638 nm", help='signal channel (default "638 nm")')
    parser.add_argument("--af", default="561 nm", help='autofluorescence channel (default "561 nm")')
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: afcorr/ next to the h5)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Background subtraction and autofluorescence removal for mesoSPIM data, "
        "using a registered off-target channel (e.g. AF647 at 638 nm, AF at 561 nm)."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    tiff = commands.add_parser("tiff", help="correct a pair of projection TIFFs")
    tiff.add_argument("signal", type=Path, help="signal-channel projection TIFF")
    tiff.add_argument("af", type=Path, help="autofluorescence-channel projection TIFF")
    tiff.add_argument("--out-dir", type=Path, default=None, help="default: next to signal")
    tiff.add_argument("--prefix", default=None, help="default: signal file stem")
    tiff.add_argument("--pixel-size-um", type=float, default=REFERENCE_PIXEL_SIZE_UM,
                      help=f"default {REFERENCE_PIXEL_SIZE_UM} (mesoSPIM 2x, level 0)")
    tiff.set_defaults(run=run_tiff)

    slab = commands.add_parser("slab", help="correct one slab of a stitched.h5")
    add_volume_args(slab)
    slab.add_argument("--z-start-um", type=float, required=True, help="top of the slab")
    slab.set_defaults(run=run_slab)

    brain = commands.add_parser(
        "brain", help="correct every consecutive slab of a stitched.h5 into a stack + csv"
    )
    add_volume_args(brain)
    brain.add_argument("--z-range-um", type=float, nargs=2, default=None, metavar=("START", "END"),
                       help="restrict to this depth range (default: whole volume)")
    brain.add_argument("--save-bgsub", action="store_true",
                       help="also write the background-subtracted (not AF-corrected) stack")
    brain.set_defaults(run=run_brain)

    list_ = commands.add_parser("list", help="list acquisitions under a root directory")
    list_.add_argument("root", type=Path)
    list_.add_argument("--mouse", default=None, help="only this mouse ID")
    list_.set_defaults(run=run_list)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
