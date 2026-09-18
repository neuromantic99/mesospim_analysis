import datetime
from pathlib import Path

import pytest

from mesospim_analysis.acquisitions import find_acquisitions, parse_acquisition, resolve_h5


def test_parse_full_layout() -> None:
    a = parse_acquisition(Path("/data/2026-05-18/N030/001/stitched.h5"))
    assert (a.date, a.mouse_id, a.imaging_number) == (datetime.date(2026, 5, 18), "N030", "001")
    assert a.name == "2026-05-18_N030_001"


def test_parse_without_imaging_number() -> None:
    a = parse_acquisition(Path("/data/2026-05-18/N027/stitched.h5"))
    assert (a.mouse_id, a.imaging_number) == ("N027", None)
    assert a.name == "2026-05-18_N027"


def test_parse_rejects_other_layouts() -> None:
    with pytest.raises(ValueError, match="yyyy-mm-dd"):
        parse_acquisition(Path("/data/N027/001/stitched.h5"))


def test_resolve_h5_accepts_directory(tmp_path: Path) -> None:
    assert resolve_h5(tmp_path) == tmp_path / "stitched.h5"
    assert resolve_h5(tmp_path / "x.h5") == tmp_path / "x.h5"


def test_find_acquisitions_sorts_and_filters(tmp_path: Path) -> None:
    for rel in [
        "2026-05-19/N030/001", "2026-05-18/N030/002", "2026-05-18/N030/001",
        "2026-05-18/N027", "notes/N030/001",
    ]:
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "stitched.h5").touch()
    assert [a.name for a in find_acquisitions(tmp_path)] == [
        "2026-05-18_N027", "2026-05-18_N030_001", "2026-05-18_N030_002", "2026-05-19_N030_001",
    ]
    assert [a.name for a in find_acquisitions(tmp_path, mouse_id="N027")] == ["2026-05-18_N027"]
