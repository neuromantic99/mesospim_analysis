import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import tifffile

from conftest import WriteAcquisition
from mesospim_analysis.acquisitions import find_acquisitions
from mesospim_analysis.run import run_brain, run_slab

SIDE = 260
PLANES = 30  # three 50 um slabs at 5 um steps
TISSUE_RADIUS = 110
OUT_OF_TISSUE, TISSUE = 120.0, 200.0
BLEED_THROUGH = 0.07  # label intensity appearing in the 561 nm channel


def _spot_grid() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    centre = SIDE // 2
    spots = [
        (r, c)
        for r in range(centre - 80, centre + 81, 20)
        for c in range(centre - 80, centre + 81, 20)
        if (r - centre) ** 2 + (c - centre) ** 2 < 80**2
    ]
    return spots[::2], spots[1::2]


def _synthetic_brain() -> tuple[dict[str, npt.NDArray[np.uint16]], int, int]:
    """Slab 0 is above the brain (camera noise only); slabs 1-2 hold a disc of tissue with
    autofluorescent spots (in both channels) and labelled cells (638 nm, bleeding into 561 nm)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:SIDE, :SIDE]
    disc = (yy - SIDE // 2) ** 2 + (xx - SIDE // 2) ** 2 < TISSUE_RADIUS**2
    base = np.where(disc, TISSUE, OUT_OF_TISSUE)
    af_spots, specific_spots = _spot_grid()

    volumes: dict[str, npt.NDArray[np.uint16]] = {}
    for channel in ("638 nm", "561 nm", "488 nm"):
        volume = np.empty((PLANES, SIDE, SIDE), dtype=np.float64)
        volume[:10] = 100.0  # above the brain
        volume[10:] = base
        volume += rng.normal(0, 5, volume.shape)
        for slab_start in (10, 20):
            for i, (r, c) in enumerate(af_spots):
                z = slab_start + 2 + i % 6
                volume[z, r - 1 : r + 2, c - 1 : c + 2] += {"638 nm": 300, "561 nm": 400}.get(channel, 0)
            for i, (r, c) in enumerate(specific_spots):
                amplitude = 3000 if i % 4 == 0 else 300  # some bright enough to show at 561
                scale = {"638 nm": 1.0, "561 nm": BLEED_THROUGH}.get(channel, 0.0)
                volume[slab_start + 1 + i % 7, r - 1 : r + 2, c - 1 : c + 2] += scale * amplitude
        volumes[channel] = np.clip(volume, 0, 65535).astype(np.uint16)
    return volumes, len(af_spots), len(specific_spots)


def test_run_brain_corrects_each_slab(write_acquisition: WriteAcquisition, tmp_path: Path) -> None:
    volumes, n_af, n_specific = _synthetic_brain()
    h5 = write_acquisition(volumes)
    out = tmp_path / "out"
    summaries = run_brain(h5.parent, out_dir=out, save_bgsub=True)
    assert [s.puncta_specific for s in summaries] == [0, n_specific, n_specific]

    with open(out / "2026-05-18_N027_001_50um_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["slab"] for r in rows] == ["0", "1", "2"]
    assert rows[0]["status"] != "ok"
    for row in rows[1:]:
        assert row["status"] == "ok"
        assert (row["z_start_um"], row["z_end_um"]) in [("50.0", "100.0"), ("100.0", "150.0")]
        assert int(row["puncta_autofluorescent"]) == n_af
        assert int(row["puncta_specific"]) == n_specific
        assert int(row["objects_bgsub_gt100"]) == n_af + n_specific
        assert int(row["objects_afcorr_gt100"]) == n_specific

    afcorr = tifffile.imread(out / "2026-05-18_N027_001_50um_afcorr.tif")
    bgsub = tifffile.imread(out / "2026-05-18_N027_001_50um_bgsub.tif")
    assert afcorr.shape == bgsub.shape == (3, SIDE, SIDE)
    assert not afcorr[0].any()  # failed slab kept as a blank frame so frame index == slab index


def test_run_slab_writes_one_projection_pair(
    write_acquisition: WriteAcquisition, tmp_path: Path
) -> None:
    volumes, _, n_specific = _synthetic_brain()
    h5 = write_acquisition(volumes, imaging=None)
    out = tmp_path / "out"
    summary, _ = run_slab(h5, z_start_um=50, out_dir=out)
    assert (summary.z_start_um, summary.puncta_specific) == (50, n_specific)
    for kind in ("bgsub", "afcorr"):
        image = tifffile.imread(out / f"2026-05-18_N027_50um_z50_{kind}.tif")
        assert image.shape == (SIDE, SIDE) and image.dtype == np.uint16


def test_coarse_pyramid_level_warns(
    write_acquisition: WriteAcquisition, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    volumes, _, _ = _synthetic_brain()
    h5 = write_acquisition(volumes, extra_level=True)
    run_slab(h5, z_start_um=50, pyramid_level=1, out_dir=tmp_path)
    assert "too coarse" in capsys.readouterr().err


def test_find_written_acquisition(write_acquisition: WriteAcquisition, tmp_path: Path) -> None:
    volumes, _, _ = _synthetic_brain()
    h5 = write_acquisition(volumes, mouse="N030")
    [acquisition] = find_acquisitions(tmp_path)
    assert (acquisition.name, acquisition.h5_path) == ("2026-05-18_N030_001", h5)
