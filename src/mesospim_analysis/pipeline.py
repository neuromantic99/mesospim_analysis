"""Autofluorescence correction of consecutive z-slab projections from a BigStitcher volume."""

import csv
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import tifffile

from mesospim_analysis.correction import (
    REFERENCE_PIXEL_SIZE_UM,
    AutofluorescenceCorrection,
    correct_autofluorescence,
)
from mesospim_analysis.detection import classify_by_autofluorescence, count_objects
from mesospim_analysis.projection import Slab, Volume, iter_slab_projections
from mesospim_analysis.types import FloatImage
from mesospim_analysis.utils import (
    open_channel_dataset,
    pyramid_downsampling,
    read_stitched_metadata,
)

COUNT_THRESHOLDS = (100.0, 150.0, 200.0)
"""ADU above local background. Only comparable between brains imaged with the same settings."""

COARSE_PIXEL_WARNING_FACTOR = 1.5


@dataclass(frozen=True)
class Scale:
    pixel_size_um: float
    z_step_um: float

    @property
    def too_coarse_for_counting(self) -> bool:
        """Nuclei are ~3 px across at the reference size; much coarser and noise passes as cells."""
        return self.pixel_size_um > COARSE_PIXEL_WARNING_FACTOR * REFERENCE_PIXEL_SIZE_UM


@dataclass(frozen=True)
class SlabSummary:
    slab: int
    z_start: int
    z_end: int
    z_start_um: float
    z_end_um: float
    status: str
    """"ok", or why the slab could not be corrected (e.g. no tissue above/below the brain)."""
    tissue_px: int = 0
    alpha: float = math.nan
    n_fit_pixels: int = 0
    signal_sigma: float = math.nan
    af_sigma: float = math.nan
    puncta_total: int = 0
    """Objects >2x local background in the signal channel."""
    puncta_autofluorescent: int = 0
    puncta_specific: int = 0
    objects_bgsub: dict[float, int] = field(default_factory=dict)
    objects_afcorr: dict[float, int] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        bgsub: dict[float, int] = row.pop("objects_bgsub")
        afcorr: dict[float, int] = row.pop("objects_afcorr")
        for threshold in COUNT_THRESHOLDS:
            row[f"objects_bgsub_gt{threshold:.0f}"] = bgsub.get(threshold, 0)
            row[f"objects_afcorr_gt{threshold:.0f}"] = afcorr.get(threshold, 0)
        return row


def summarize(
    slab: Slab,
    af_projection: FloatImage,
    result: AutofluorescenceCorrection,
    scale: Scale,
) -> SlabSummary:
    coloc = classify_by_autofluorescence(
        slab.projection, af_projection, result.tissue, pixel_size_um=scale.pixel_size_um
    )
    n_af = int(coloc.is_autofluorescent.sum())
    return SlabSummary(
        slab=slab.index,
        z_start=slab.z_start,
        z_end=slab.z_end,
        z_start_um=slab.z_start * scale.z_step_um,
        z_end_um=slab.z_end * scale.z_step_um,
        status="ok",
        tissue_px=int(result.tissue.sum()),
        alpha=result.alpha,
        n_fit_pixels=result.n_fit_pixels,
        signal_sigma=result.signal_sigma,
        af_sigma=result.af_sigma,
        puncta_total=len(coloc.is_autofluorescent),
        puncta_autofluorescent=n_af,
        puncta_specific=len(coloc.is_autofluorescent) - n_af,
        objects_bgsub={
            t: count_objects(result.background_subtracted, t, result.tissue, scale.pixel_size_um)
            for t in COUNT_THRESHOLDS
        },
        objects_afcorr={
            t: count_objects(result.corrected, t, result.tissue, scale.pixel_size_um)
            for t in COUNT_THRESHOLDS
        },
    )


def process_slabs(
    signal: Volume,
    af: Volume,
    planes: int,
    scale: Scale,
    z_start: int = 0,
    z_end: int | None = None,
    read_block: int = 32,
) -> Iterator[tuple[SlabSummary, AutofluorescenceCorrection | None]]:
    """Correct each slab independently, fitting alpha per slab.

    Alpha is refitted per slab because 561 and 638 nm light attenuate differently with depth,
    so the AF ratio need not be constant through the brain. Check its stability in the summary.
    """
    if signal.shape != af.shape:
        raise ValueError(f"channel volumes differ in shape: {signal.shape} vs {af.shape}")
    signal_slabs = iter_slab_projections(signal, planes, z_start, z_end, read_block)
    af_slabs = iter_slab_projections(af, planes, z_start, z_end, read_block)
    for signal_slab, af_slab in zip(signal_slabs, af_slabs, strict=True):
        try:
            result = correct_autofluorescence(
                signal_slab.projection, af_slab.projection, scale.pixel_size_um
            )
        except ValueError as e:
            yield (
                SlabSummary(
                    slab=signal_slab.index,
                    z_start=signal_slab.z_start,
                    z_end=signal_slab.z_end,
                    z_start_um=signal_slab.z_start * scale.z_step_um,
                    z_end_um=signal_slab.z_end * scale.z_step_um,
                    status=str(e),
                ),
                None,
            )
            continue
        yield summarize(signal_slab, af_slab.projection, result, scale), result


@dataclass(frozen=True)
class OpenVolumes:
    signal: h5py.Dataset
    af: h5py.Dataset
    scale: Scale
    read_block: int


def open_volumes(
    h5_file: h5py.File, h5_path: Path, pyramid_level: int, signal_channel: str, af_channel: str
) -> OpenVolumes:
    metadata = read_stitched_metadata(h5_path)
    signal_setup = metadata.setup_id(signal_channel)
    af_setup = metadata.setup_id(af_channel)
    factors = pyramid_downsampling(h5_file, signal_setup)
    if pyramid_level >= len(factors):
        raise ValueError(f"pyramid level {pyramid_level} not available (0-{len(factors) - 1})")
    if pyramid_downsampling(h5_file, af_setup)[pyramid_level] != factors[pyramid_level]:
        raise ValueError("signal and af channels have different pyramid downsampling")
    fz, fy, _ = factors[pyramid_level]
    vz, vy, _ = metadata.voxel_size_zyx_um
    signal = open_channel_dataset(h5_file, signal_setup, pyramid_level)
    af = open_channel_dataset(h5_file, af_setup, pyramid_level)
    chunks = signal.chunks
    return OpenVolumes(
        signal=signal,
        af=af,
        scale=Scale(pixel_size_um=vy * fy, z_step_um=vz * fz),
        read_block=int(chunks[0]) if chunks else 32,
    )


class SlabStackWriter:
    """Streams slab images into a BigTIFF stack; failed slabs are written as blank frames."""

    def __init__(self, path: Path, description: str) -> None:
        self._writer = tifffile.TiffWriter(path, bigtiff=True)
        self._description: str | None = description

    def write(self, image: FloatImage | None, shape: tuple[int, ...]) -> None:
        frame = (
            np.zeros(shape, np.uint16)
            if image is None
            else np.clip(image, 0, 65535).astype(np.uint16)
        )
        self._writer.write(frame, contiguous=True, description=self._description)
        self._description = None

    def close(self) -> None:
        self._writer.close()


def write_summary_csv(path: Path, summaries: list[SlabSummary]) -> None:
    rows = [s.as_row() for s in summaries]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
