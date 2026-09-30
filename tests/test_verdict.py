from pathlib import Path

import pytest

from conftest import WriteAcquisition
from mesospim_analysis.detection import DEFAULT_ALPHA
from mesospim_analysis.pipeline import SlabSummary, read_summary_csv
from mesospim_analysis.run import run_brain
from mesospim_analysis.verdict import (
    CALIBRATED_AT_CONTRAST,
    MIN_ALPHA_FIT_SLABS,
    PLAUSIBLE_ALPHA,
    BrainMeasurement,
    Verdict,
    _slab_alphas,
    judge,
    measure_brain,
    select_specific,
)
from test_pipeline import _synthetic_brain


def _measurement(
    density: float,
    alpha: float = 0.6,
    frame_mp: float = 15.3,
    retained: float = 0.9,
    robustness: float = 0.9,
    n_alpha_slabs: int = 50,
) -> BrainMeasurement:
    return BrainMeasurement(
        "test", int(density * frame_mp), density, 0.7, alpha, frame_mp, retained,
        robustness, n_alpha_slabs,
    )


def _other_caveats(verdict: Verdict) -> tuple[str, ...]:
    """Caveats other than the standing one about the detection contrast.

    That caveat fires on every judgement while the contrast differs from the one the density
    thresholds were set at, which is the point of it. These tests are about the others.
    """
    return tuple(c for c in verdict.caveats if "detection contrast" not in c)


def _slab(index: int, alpha: float, n_fit_pixels: int) -> SlabSummary:
    return SlabSummary(
        slab=index, z_start=index * 10, z_end=index * 10 + 10,
        z_start_um=index * 50.0, z_end_um=index * 50.0 + 50.0, status="ok",
        tissue_px=9_000_000, alpha=alpha, n_fit_pixels=n_fit_pixels,
        signal_sigma=6.0, af_sigma=6.0,
        puncta_total=0, puncta_autofluorescent=0, puncta_specific=0,
    )


@pytest.mark.parametrize(
    ("density", "floor", "call", "confidence"),
    [
        (1300, 13, "stain worked", "high"),      # N027-like: 100x
        (250, 13, "stain worked", "medium"),     # N028-like: 19x
        (30, 13, "weak labelling", "low"),       # N021-like against a borrowed floor
        (13, 13, "at floor", "medium"),          # a control against itself
    ],
)
def test_calls_against_a_same_round_floor(
    density: float, floor: float, call: str, confidence: str
) -> None:
    verdict = judge(_measurement(density), floor, "round X")
    assert (verdict.call, verdict.confidence) == (call, confidence)
    assert verdict.times_floor == pytest.approx(density / floor)


@pytest.mark.parametrize(
    ("density", "call"),
    [(1135, "stain worked"), (279, "probably worked"), (94, "cannot call")],
)
def test_calls_without_a_floor_are_hedged(density: float, call: str) -> None:
    verdict = judge(_measurement(density))
    assert verdict.call == call
    assert any("no same-round control" in c for c in verdict.caveats)


def test_untrained_conditions_are_flagged() -> None:
    assert any("above the trained range" in c for c in judge(_measurement(1300, alpha=2.5), 13).caveats)
    assert any("differs from the training frames" in c for c in
               judge(_measurement(1300, frame_mp=2.0), 13).caveats)
    assert not _other_caveats(judge(_measurement(1300, frame_mp=8.0), 13))  # hemibrain ok


def test_at_floor_states_it_cannot_identify_a_control() -> None:
    caveats = judge(_measurement(13), 13, "round X").caveats
    assert any("failed stain and a secondary-only control look identical" in c for c in caveats)


def test_select_specific_uses_each_slab_alpha(
    write_acquisition: WriteAcquisition, tmp_path: Path
) -> None:
    volumes, _, n_specific = _synthetic_brain()
    out = tmp_path / "out"
    summaries = run_brain(write_acquisition(volumes).parent, out_dir=out)
    stem = "2026-05-18_N027_001_50um"
    import csv as _csv

    with open(out / f"{stem}_objects.csv", newline="") as f:
        objects = list(_csv.DictReader(f))
    selected = select_specific(objects, read_summary_csv(out / f"{stem}_summary.csv"))
    assert len(selected) == sum(s.puncta_specific for s in summaries)


def test_measure_brain_reads_a_finished_run(
    write_acquisition: WriteAcquisition, tmp_path: Path
) -> None:
    volumes, _, n_specific = _synthetic_brain()
    out = tmp_path / "out"
    run_brain(write_acquisition(volumes).parent, out_dir=out)
    stem = "2026-05-18_N027_001_50um"
    m = measure_brain(out / f"{stem}_objects.csv", out / f"{stem}_summary.csv")
    assert m.n_specific == 2 * n_specific  # two tissue slabs in the synthetic brain
    assert m.density_per_mp > 0 and m.frame_mp > 0


def test_low_retention_is_flagged_without_changing_the_call() -> None:
    clean = judge(_measurement(1300, retained=0.9), 13)
    marginal = judge(_measurement(1300, retained=0.17), 13)
    assert (marginal.call, marginal.confidence) == (clean.call, clean.confidence)
    assert any("stricter cutoff" in c for c in marginal.caveats)
    assert not _other_caveats(clean)


def test_a_wild_alpha_from_a_few_slabs_cannot_set_the_cutoff() -> None:
    """N052 fitted alpha 82 from 2 slabs of 190 and lost 90% of its objects to the cutoff."""
    degenerate = PLAUSIBLE_ALPHA[1] * 30
    summaries = [_slab(i, degenerate, 5000) for i in range(2)]
    summaries += [_slab(i, float("nan"), 100) for i in range(2, 190)]
    alphas = _slab_alphas(summaries)
    assert alphas.fallback == DEFAULT_ALPHA
    assert alphas.n_believed == 0 and alphas.assumed
    assert set(alphas.per_slab.values()) == {DEFAULT_ALPHA}


def test_a_median_from_too_few_slabs_is_not_trusted_for_the_rest() -> None:
    believable = 1.4
    summaries = [_slab(i, believable, 5000) for i in range(MIN_ALPHA_FIT_SLABS - 1)]
    summaries += [_slab(i, float("nan"), 10) for i in range(MIN_ALPHA_FIT_SLABS - 1, 100)]
    alphas = _slab_alphas(summaries)
    assert alphas.assumed and alphas.fallback == DEFAULT_ALPHA
    assert alphas.per_slab[0] == believable  # a slab keeps its own measurement
    assert alphas.per_slab[50] == DEFAULT_ALPHA  # an unfitted slab does not inherit it


def test_enough_slabs_are_trusted() -> None:
    summaries = [_slab(i, 1.4, 5000) for i in range(MIN_ALPHA_FIT_SLABS)]
    summaries += [_slab(i, float("nan"), 10) for i in range(MIN_ALPHA_FIT_SLABS, 100)]
    alphas = _slab_alphas(summaries)
    assert not alphas.assumed
    assert alphas.fallback == pytest.approx(1.4)
    assert alphas.per_slab[50] == pytest.approx(1.4)


def test_assumed_alpha_is_flagged_without_changing_the_call() -> None:
    measured = judge(_measurement(1300, n_alpha_slabs=50), 13)
    assumed = judge(_measurement(1300, n_alpha_slabs=1), 13)
    assert (assumed.call, assumed.confidence) == (measured.call, measured.confidence)
    assert any("alpha assumed" in c for c in assumed.caveats)
    assert not _other_caveats(measured)


def test_alpha_dependent_counts_are_flagged_without_changing_the_call() -> None:
    solid = judge(_measurement(1300, robustness=0.99), 13)
    fragile = judge(_measurement(1300, robustness=0.05), 13)
    assert (fragile.call, fragile.confidence) == (solid.call, solid.confidence)
    assert any("depends on where the cutoff sits" in c for c in fragile.caveats)
    assert not _other_caveats(solid)


def test_robustness_separates_a_real_brain_from_a_control(
    write_acquisition: WriteAcquisition, tmp_path: Path
) -> None:
    volumes, _, _ = _synthetic_brain()
    out = tmp_path / "out"
    run_brain(write_acquisition(volumes).parent, out_dir=out)
    stem = "2026-05-18_N027_001_50um"
    m = measure_brain(out / f"{stem}_objects.csv", out / f"{stem}_summary.csv")
    assert 0.0 <= m.alpha_robustness <= 1.0
    assert m.alpha_robustness > 0.5  # synthetic cells sit well clear of the autofluorescence


def test_a_changed_detection_contrast_is_flagged() -> None:
    """The density thresholds are absolute counts, so they do not survive a contrast change:
    lowering it to 1.5 roughly triples what a brain yields."""
    from mesospim_analysis.detection import DETECTION_CONTRAST

    verdict = judge(_measurement(1300), 13)
    flagged = any("detection contrast" in c for c in verdict.caveats)
    assert flagged == (DETECTION_CONTRAST != CALIBRATED_AT_CONTRAST)
