from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import SETUP_OF_CHANNEL, WriteAcquisition
from mesospim_analysis.utils import (
    get_wavelengths_of_channels,
    load_bigstitched_data,
    pyramid_downsampling,
    read_stitched_metadata,
)

SHAPE = (12, 32, 48)


def _channel_volumes() -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.uint16]]]:
    # each channel filled with a distinct value so a wrong lookup is obvious
    return {
        name: np.full(SHAPE, value, dtype=np.uint16)
        for name, value in [("638 nm", 638), ("488 nm", 488), ("561 nm", 561)]
    }


def test_channel_names_map_to_ids(write_acquisition: WriteAcquisition) -> None:
    h5 = write_acquisition(_channel_volumes())
    assert get_wavelengths_of_channels(h5.with_suffix(".xml")) == {
        "638 nm": 0, "488 nm": 1, "561 nm": 2,
    }


def test_metadata_maps_channels_through_setup_attributes(write_acquisition: WriteAcquisition) -> None:
    metadata = read_stitched_metadata(write_acquisition(_channel_volumes()))
    assert metadata.shape_zyx == SHAPE
    assert metadata.voxel_size_zyx_um == (5.0, 3.26, 3.26)
    assert metadata.channel_setups == SETUP_OF_CHANNEL
    with pytest.raises(KeyError, match="405 nm"):
        metadata.setup_id("405 nm")


def test_load_reads_the_requested_channel_and_z_range(write_acquisition: WriteAcquisition) -> None:
    h5 = write_acquisition(_channel_volumes())
    for channel, value in [("638 nm", 638), ("561 nm", 561)]:
        stack = np.asarray(load_bigstitched_data(h5, 0, channel, 2, 7))  # type: ignore[arg-type]
        assert stack.shape == (5, *SHAPE[1:])
        assert np.all(stack == value)


def test_load_undoes_int16_storage(write_acquisition: WriteAcquisition) -> None:
    volumes = _channel_volumes()
    volumes["638 nm"][3, 10, 10] = 40000  # stored as a negative int16
    stack = load_bigstitched_data(write_acquisition(volumes), 0, "638 nm", 0, SHAPE[0])
    assert stack.dtype == np.uint16
    assert int(np.asarray(stack).max()) == 40000


def test_pyramid_downsampling_is_returned_zyx(write_acquisition: WriteAcquisition) -> None:
    h5 = write_acquisition(_channel_volumes(), extra_level=True)
    with h5py.File(h5, "r") as f:
        assert pyramid_downsampling(f, SETUP_OF_CHANNEL["638 nm"]) == [(1, 1, 1), (1, 2, 2)]


def test_rejects_channel_with_several_setups(tmp_path: Path, write_acquisition: WriteAcquisition) -> None:
    h5 = write_acquisition(_channel_volumes())
    xml = h5.with_suffix(".xml")
    # unfused data: the 488 setup now also claims channel 0 (638 nm)
    xml.write_text(xml.read_text().replace("<channel>1</channel>", "<channel>0</channel>"))
    with pytest.raises(ValueError, match="expected exactly one"):
        read_stitched_metadata(h5)
