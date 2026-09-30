"""Frozen decision rule: does a brain contain specific labelling above its background?

Every threshold here was chosen on brains N021, N024, N027, N028, N029, N030, N032 and N041
(see FROZEN_ON). Those brains are the training set and no accuracy claimed on them means
anything. The rule is written as code so that later brains are judged mechanically, without
the thresholds drifting to fit whatever arrives.

What it answers: whether a brain has labelling above the background of its own staining round.
What it cannot answer: whether primary antibody was applied. A real brain whose stain failed is
indistinguishable from a secondary-only control here, and trying to tell them apart by eye from
these numbers produced several wrong calls during development. Take the label from the record.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mesospim_analysis.detection import (
    AF_RATIO_MULTIPLE,
    DEFAULT_ALPHA,
    DETECTION_CONTRAST,
    MIN_ALPHA_FIT_PIXELS,
)
from mesospim_analysis.filters import DEFAULT_FILTER, ObjectRow, read_objects_csv
from mesospim_analysis.pipeline import SlabSummary, read_summary_csv

FROZEN_ON = "2026-09-22"

WORKED_TIMES_FLOOR = 20.0
LIKELY_TIMES_FLOOR = 5.0
WEAK_TIMES_FLOOR = 2.0
"""Multiples of the staining round's secondary-only floor. In training, brains with a
same-round control sat at 89x (N027) and 19x (N028); the controls themselves at 1x."""

CALIBRATED_AT_CONTRAST = 2.0
"""The detection contrast the density thresholds below were set at.

They are absolute counts, so they do not survive a change to the contrast: lowering it to 1.5
roughly triples the objects a brain yields, and a control that measured 222/MP would measure
far more. The multiples of a same-round floor are ratios and are less affected, but they have
not been re-checked either. Judging a run made at a different contrast against these numbers
would call weak brains strong."""

STRONG_DENSITY_PER_MP = 1000.0
POSSIBLE_DENSITY_PER_MP = 250.0
"""Used only when no same-round control exists. Below ~250/MP nothing can be concluded: a
secondary-only control measured 222/MP and a brain whose stain worked measured 247/MP."""

STRICT_MULTIPLE = 8.0
LOW_RETENTION = 0.3
"""Added after the freeze, and deliberately not part of the call: the share of objects
surviving a much stricter cutoff says whether a brain's counted objects form a population
distinct from its autofluorescence, or merely sit on top of it. It does NOT separate real
brains from controls -- a secondary-only control retained 80% because nonspecific antibody is
also AF647 -- so it flags an untrustworthy count, not a control."""

TISSUE_FRACTION_OF_FRAME = 0.6
"""The tissue mask always selects this share of the frame, so it measures the frame, not the
tissue. Densities are per megapixel of frame and are only comparable at equal tissue filling."""

PLAUSIBLE_ALPHA = (0.02, 3.0)
MIN_ALPHA_FIT_SLABS = 5
"""A slab's fitted alpha is only believed inside PLAUSIBLE_ALPHA, and a brain's median is only
believed when at least MIN_ALPHA_FIT_SLABS slabs contribute. Both guards exist because N052
fitted alpha 82 from 2 slabs out of 190: `correction.min_af_for_fit` is an absolute threshold,
so a slab with bright label and almost no autofluorescence yields a huge signal/AF median. That
alpha set the cutoff for all 190 slabs and discarded 90% of the brain's objects."""

ROBUSTNESS_ALPHA = (0.3, 2.0)
LOW_ROBUSTNESS = 0.5
"""Added after the freeze alongside LOW_RETENTION, and likewise not part of the call: the
density at ROBUSTNESS_ALPHA[1] over the density at ROBUSTNESS_ALPHA[0], i.e. how much of the
count survives assuming alpha is at the top rather than the bottom of its range. Near 1 the
count does not depend on alpha at all; near 0 it is entirely an artifact of where the cutoff
sits. Measured 0.99 in the best brain and 0.00 in a no-antibody control. This matters because
most brains after the training rounds fit no alpha at all and fall back to DEFAULT_ALPHA."""


@dataclass(frozen=True)
class BrainMeasurement:
    name: str
    n_specific: int
    density_per_mp: float
    geometry_score: float
    """Mean object score from DEFAULT_FILTER; real brains scored 0.67-0.72, controls 0.33-0.48."""
    median_alpha: float
    frame_mp: float
    retained_at_strict_cutoff: float
    """Fraction of the counted objects that survive STRICT_MULTIPLE x alpha. Near 1 means a
    clean population well clear of autofluorescence; near 0 means the count is an artifact of
    where the cutoff sits. Measured 0.97 in the best brain and 0.17 in the least reliable."""
    alpha_robustness: float = float("nan")
    """See LOW_ROBUSTNESS. Unlike retention, this is independent of the alpha actually used."""
    n_alpha_slabs: int = 0
    """Slabs whose fitted alpha was believed. 0 means median_alpha is DEFAULT_ALPHA, assumed."""


@dataclass(frozen=True)
class Verdict:
    measurement: BrainMeasurement
    floor_per_mp: float | None
    floor_source: str
    times_floor: float | None
    call: str
    confidence: str
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class SlabAlphas:
    per_slab: dict[int, float]
    fallback: float
    n_believed: int

    @property
    def assumed(self) -> bool:
        """True when too few slabs fitted for the brain's median to mean anything, so slabs
        without a believable fit of their own were judged against DEFAULT_ALPHA."""
        return self.n_believed < MIN_ALPHA_FIT_SLABS


def _believable(summary: SlabSummary) -> bool:
    low, high = PLAUSIBLE_ALPHA
    return (
        summary.n_fit_pixels >= MIN_ALPHA_FIT_PIXELS
        and bool(np.isfinite(summary.alpha))
        and low <= summary.alpha <= high
    )


def _slab_alphas(summaries: list[SlabSummary]) -> SlabAlphas:
    """Each slab's alpha, with implausible fits and under-supported medians replaced.

    A slab's own fit is used only when it is believable; otherwise the slab takes the brain's
    median. When too few slabs contribute to that median it is not a measurement of this brain,
    so those slabs take DEFAULT_ALPHA instead and the result says so via `assumed`.
    """
    usable = [s for s in summaries if s.status == "ok"]
    believed = [s.alpha for s in usable if _believable(s)]
    fallback = (
        float(np.median(believed)) if len(believed) >= MIN_ALPHA_FIT_SLABS else DEFAULT_ALPHA
    )
    per_slab = {s.slab: (s.alpha if _believable(s) else fallback) for s in usable}
    return SlabAlphas(per_slab, fallback, len(believed))


def select_specific(objects: list[ObjectRow], summaries: list[SlabSummary]) -> list[ObjectRow]:
    """Objects whose ratio exceeds AF_RATIO_MULTIPLE x their slab's alpha.

    Recomputes the classification from a finished run, so summaries produced before the
    alpha-relative cutoff can be corrected without reprocessing any image data.
    """
    alphas = _slab_alphas(summaries)
    return [
        row for row in objects
        if float(row["ratio"])
        > AF_RATIO_MULTIPLE * alphas.per_slab.get(int(row["slab"]), alphas.fallback)
    ]


def _density_at_alpha(objects: list[ObjectRow], alpha: float, frame_mp: float) -> float:
    """Density this brain would report if every slab's alpha were `alpha`."""
    cutoff = AF_RATIO_MULTIPLE * alpha
    return sum(1 for row in objects if float(row["ratio"]) > cutoff) / frame_mp


def measure_brain(objects_csv: Path, summary_csv: Path, name: str | None = None) -> BrainMeasurement:
    summaries = read_summary_csv(summary_csv)
    usable = [s for s in summaries if s.status == "ok"]
    if not usable:
        raise ValueError(f"{summary_csv}: no slabs were corrected")
    with open(objects_csv, newline="") as f:
        objects = list(csv.DictReader(f))
    specific = select_specific(objects, summaries)
    alphas = _slab_alphas(summaries)
    strict = [
        row for row in specific
        if float(row["ratio"])
        > STRICT_MULTIPLE * alphas.per_slab.get(int(row["slab"]), alphas.fallback)
    ]
    frame_mp = float(np.median([s.tissue_px for s in usable])) / TISSUE_FRACTION_OF_FRAME / 1e6
    permissive = _density_at_alpha(objects, ROBUSTNESS_ALPHA[0], frame_mp)
    strictest = _density_at_alpha(objects, ROBUSTNESS_ALPHA[1], frame_mp)
    return BrainMeasurement(
        name=name or summary_csv.name.split("_50um")[0],
        n_specific=len(specific),
        density_per_mp=len(specific) / frame_mp,
        geometry_score=DEFAULT_FILTER.brain_score(specific) if specific else float("nan"),
        median_alpha=alphas.fallback,
        frame_mp=frame_mp,
        retained_at_strict_cutoff=len(strict) / len(specific) if specific else float("nan"),
        alpha_robustness=strictest / permissive if permissive > 0 else float("nan"),
        n_alpha_slabs=alphas.n_believed,
    )


def judge(
    measurement: BrainMeasurement,
    floor_per_mp: float | None = None,
    floor_source: str = "",
) -> Verdict:
    """Call a brain against the secondary-only floor of its own staining round, if there is one."""
    caveats: list[str] = []
    if floor_per_mp is not None and floor_per_mp > 0:
        times = measurement.density_per_mp / floor_per_mp
        if times >= WORKED_TIMES_FLOOR:
            call, confidence = "stain worked", "high"
        elif times >= LIKELY_TIMES_FLOOR:
            call, confidence = "stain worked", "medium"
        elif times >= WEAK_TIMES_FLOOR:
            call, confidence = "weak labelling", "low"
        else:
            call, confidence = "at floor", "medium"
            caveats.append("at floor: a failed stain and a secondary-only control look identical")
    else:
        times = None
        caveats.append("no same-round control; floors varied 17-fold between rounds in training")
        if measurement.density_per_mp >= STRONG_DENSITY_PER_MP:
            call, confidence = "stain worked", "medium"
        elif measurement.density_per_mp >= POSSIBLE_DENSITY_PER_MP:
            call, confidence = "probably worked", "low"
        else:
            call, confidence = "cannot call", "none"
            caveats.append(
                f"below {POSSIBLE_DENSITY_PER_MP:.0f}/MP a working stain and a control overlap"
            )
    if measurement.retained_at_strict_cutoff < LOW_RETENTION:
        caveats.append(
            f"only {100 * measurement.retained_at_strict_cutoff:.0f}% of objects survive a "
            "stricter cutoff; the count sits close to this brain's autofluorescence"
        )
    if measurement.alpha_robustness < LOW_ROBUSTNESS:
        caveats.append(
            f"only {100 * measurement.alpha_robustness:.0f}% of the count survives assuming "
            f"alpha {ROBUSTNESS_ALPHA[1]:.1f} rather than {ROBUSTNESS_ALPHA[0]:.1f}; the count "
            "depends on where the cutoff sits, not on the objects"
        )
    if measurement.n_alpha_slabs < MIN_ALPHA_FIT_SLABS:
        caveats.append(
            f"alpha assumed: only {measurement.n_alpha_slabs} slab(s) fitted, so "
            f"{DEFAULT_ALPHA} was used rather than a measurement of this brain"
        )
    elif measurement.median_alpha > 2.0:
        caveats.append(f"alpha {measurement.median_alpha:.2f} is above the trained range (0.49-2.13)")
    if not 0.3 <= measurement.frame_mp / 15.3 <= 1.2:
        caveats.append(f"frame {measurement.frame_mp:.1f} MP differs from the training frames")
    if DETECTION_CONTRAST != CALIBRATED_AT_CONTRAST:
        caveats.append(
            f"detection contrast is {DETECTION_CONTRAST}, but these thresholds were set at "
            f"{CALIBRATED_AT_CONTRAST}; densities are not comparable and the call may be wrong"
        )
    return Verdict(measurement, floor_per_mp, floor_source, times, call, confidence, tuple(caveats))


def format_verdict(verdict: Verdict) -> str:
    m = verdict.measurement
    times = f"{verdict.times_floor:.0f}x {verdict.floor_source or 'floor'}" if verdict.times_floor else "no floor"
    lines = [
        f"{m.name}: {verdict.call} ({verdict.confidence} confidence)",
        f"  {m.n_specific} specific objects, {m.density_per_mp:.0f}/MP, {times}",
        f"  geometry {m.geometry_score:.3f}, alpha {m.median_alpha:.2f}"
        f"{' (assumed)' if m.n_alpha_slabs < MIN_ALPHA_FIT_SLABS else f' from {m.n_alpha_slabs} slabs'}"
        f", frame {m.frame_mp:.1f} MP",
        f"  {100 * m.retained_at_strict_cutoff:.0f}% retained at strict cutoff, "
        f"{100 * m.alpha_robustness:.0f}% robust to alpha",
    ]
    lines += [f"  caveat: {c}" for c in verdict.caveats]
    return "\n".join(lines)
