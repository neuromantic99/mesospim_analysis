"""Depth profiles of a processed brain, for checking a run at a glance."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display (e.g. over ssh)
import matplotlib.pyplot as plt  # noqa: E402

from mesospim_analysis.pipeline import SlabSummary  # noqa: E402

RELIABLE_FIT_PIXELS = 5000
"""Below this many autofluorescent pixels the fitted alpha is noise; those slabs are
effectively outside the brain."""


def plot_depth_profiles(summaries: Mapping[str, Sequence[SlabSummary]], out_path: Path) -> Path:
    """Plot cells, autofluorescence load and fitted alpha against depth, one line per brain."""
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for label, brain in summaries.items():
        ok = [s for s in brain if s.status == "ok"]
        if not ok:
            continue
        depth_mm = [s.z_start_um / 1000 for s in ok]
        specific = [s.puncta_specific for s in ok]
        autofluorescent = [s.puncta_autofluorescent for s in ok]
        axes[0].plot(depth_mm, specific, label=f"{label} ({sum(specific)} total)")
        axes[1].plot(depth_mm, autofluorescent, label=f"{label} ({sum(autofluorescent)} total)")
        colour = axes[0].get_lines()[-1].get_color()
        solid = [(s.z_start_um / 1000, s.alpha) for s in ok if s.n_fit_pixels > RELIABLE_FIT_PIXELS]
        weak = [(s.z_start_um / 1000, s.alpha) for s in ok if s.n_fit_pixels <= RELIABLE_FIT_PIXELS]
        if solid:
            axes[2].plot(*zip(*solid, strict=True), color=colour, label=label)
        if weak:
            axes[2].plot(*zip(*weak, strict=True), ".", color=colour, ms=4, alpha=0.5)

    axes[0].set(ylabel="specific puncta / slab", title="labelled cells by depth")
    axes[1].set(ylabel="autofluorescent puncta", yscale="log", title="autofluorescence load")
    axes[2].set(ylabel="alpha (signal/AF)", xlabel="depth (mm)",
                title=f"fitted alpha; dots = slabs with <{RELIABLE_FIT_PIXELS} fit pixels")
    for axis in axes:
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=9)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return out_path
