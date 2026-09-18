import numpy as np
import pytest

from mesospim_analysis.projection import iter_slab_projections, planes_per_slab


def test_planes_per_slab() -> None:
    assert planes_per_slab(50, 5.0) == 10
    assert planes_per_slab(48, 5.0) == 10
    with pytest.raises(ValueError, match="thinner than one"):
        planes_per_slab(2, 5.0)


@pytest.mark.parametrize("read_block", [1, 3, 7, 10, 64])
def test_slabs_match_numpy_max_whatever_the_read_block(read_block: int) -> None:
    volume = np.random.default_rng(0).integers(0, 1000, (37, 8, 9)).astype(np.uint16)
    slabs = list(iter_slab_projections(volume, planes=10, read_block=read_block))
    assert [(s.index, s.z_start, s.z_end) for s in slabs] == [(0, 0, 10), (1, 10, 20), (2, 20, 30)]
    for s in slabs:  # the partial slab at 30-37 is dropped
        np.testing.assert_array_equal(s.projection, volume[s.z_start : s.z_end].max(axis=0))


def test_z_range() -> None:
    volume = np.arange(40, dtype=np.uint16)[:, None, None] * np.ones((1, 2, 2), np.uint16)
    slabs = list(iter_slab_projections(volume, planes=5, z_start=12, z_end=30))
    assert [(s.z_start, s.z_end) for s in slabs] == [(12, 17), (17, 22), (22, 27)]
    assert [float(s.projection[0, 0]) for s in slabs] == [16, 21, 26]


def test_int16_storage_does_not_hide_bright_pixels() -> None:
    volume = np.full((10, 4, 4), 100, dtype=np.uint16)
    volume[4, 1, 1] = 40000
    [slab] = iter_slab_projections(volume.view(np.int16), planes=10)
    assert slab.projection[1, 1] == 40000
