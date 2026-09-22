"""Separating labelled cells from nonspecific antibody, using object geometry.

The autofluorescence correction removes autofluorescence; it cannot remove secondary antibody
bound where it should not be, because that is genuine AF647. A secondary-only control measures
that background directly: in N041 it was ~18% of a real brain's object density. Those objects
are not randomly placed — they accumulate on vessel, ventricle and tissue surfaces, and sit in
long thin structures, whereas labelled nuclei are compact and at chance distance from such
boundaries. This module scores objects on that difference.

Deliberately geometric: object brightness separates the populations within one brain but
reflects that brain's exposure, and including it made the model worse on a held-out brain.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

ObjectRow = dict[str, str]

GEOMETRY_FEATURES = ("distance_to_dark_um", "context_length_um", "context_area_px")


def feature_matrix(
    rows: Sequence[ObjectRow], features: Sequence[str]
) -> npt.NDArray[np.float64]:
    """log1p of each feature; all are distances, lengths or areas, so positive and heavy-tailed."""
    values = np.array([[float(r[k]) for k in features] for r in rows], dtype=np.float64)
    return np.log1p(np.clip(values, 0.0, None))


@dataclass(frozen=True)
class NonspecificFilter:
    """Logistic model scoring each object 0 (nonspecific) to 1 (cell-like)."""

    features: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    provenance: str
    """What it was fitted on. Nonspecific binding is batch dependent, so this matters."""

    def score(self, rows: Sequence[ObjectRow]) -> npt.NDArray[np.float64]:
        if len(rows) == 0:
            return np.array([], dtype=np.float64)
        x = (feature_matrix(rows, self.features) - np.array(self.mean)) / np.array(self.scale)
        logit = x @ np.array(self.coefficients) + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-logit)), dtype=np.float64)

    def threshold_for_recall(self, real_rows: Sequence[ObjectRow], recall: float) -> float:
        """The score threshold keeping `recall` of the objects of a real brain."""
        if not 0.0 < recall <= 1.0:
            raise ValueError(f"recall must be in (0, 1]; got {recall}")
        return float(np.quantile(self.score(real_rows), 1.0 - recall))

    def brain_score(self, rows: Sequence[ObjectRow]) -> float:
        """Mean object score. Real brains scored ~0.67-0.70, a secondary-only control ~0.33."""
        scores = self.score(rows)
        return float(scores.mean()) if len(scores) else float("nan")

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "NonspecificFilter":
        data = json.loads(path.read_text())
        return cls(
            features=tuple(data["features"]),
            mean=tuple(data["mean"]),
            scale=tuple(data["scale"]),
            coefficients=tuple(data["coefficients"]),
            intercept=float(data["intercept"]),
            provenance=str(data["provenance"]),
        )


def fit_nonspecific_filter(
    real_rows: Sequence[ObjectRow],
    control_rows: Sequence[ObjectRow],
    features: Sequence[str] = GEOMETRY_FEATURES,
    provenance: str = "",
) -> NonspecificFilter:
    """Fit on a real brain's objects against a secondary-only control's.

    Refit per staining batch when you have a control for it. Note that a real brain's objects
    include its own nonspecific fraction, so the fitted separation is a lower bound.
    """
    from sklearn.linear_model import LogisticRegression  # fitting only; scoring is numpy
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([feature_matrix(real_rows, features), feature_matrix(control_rows, features)])
    y = np.r_[np.ones(len(real_rows)), np.zeros(len(control_rows))]
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    ).fit(x, y)
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    return NonspecificFilter(
        features=tuple(features),
        mean=tuple(float(v) for v in scaler.mean_),
        scale=tuple(float(v) for v in scaler.scale_),
        coefficients=tuple(float(v) for v in logistic.coef_[0]),
        intercept=float(logistic.intercept_[0]),
        provenance=provenance,
    )


DEFAULT_FILTER = NonspecificFilter(
    features=GEOMETRY_FEATURES,
    mean=(6.234, 3.1837, 3.0662),
    scale=(1.1717, 0.8208, 1.1117),
    coefficients=(0.546, -0.614, -0.195),
    intercept=0.5647,
    provenance="N032 (real, 2026-07-30) vs N041 (secondary-only, 2026-08-27); AUC 0.83",
)
"""Fitted on one real brain against one secondary-only control. Refit for other batches."""


def read_objects_csv(path: Path, specific_only: bool = True) -> list[ObjectRow]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["is_autofluorescent"] == "0"] if specific_only else rows


def write_scored_objects(
    path: Path,
    out_path: Path,
    threshold: float,
    filter_: NonspecificFilter = DEFAULT_FILTER,
) -> tuple[int, int]:
    """Copy an objects csv with a `cell_score` column, keeping objects scoring above `threshold`.

    Returns (kept, removed).
    """
    rows = read_objects_csv(path, specific_only=False)
    specific = [r for r in rows if r["is_autofluorescent"] == "0"]
    kept: list[ObjectRow] = []
    removed = 0
    for row, score in zip(specific, filter_.score(specific), strict=True):
        if score >= threshold:
            kept.append({**row, "cell_score": f"{score:.4f}"})
        else:
            removed += 1
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], "cell_score"])
        writer.writeheader()
        writer.writerows(kept)
    return len(kept), removed
